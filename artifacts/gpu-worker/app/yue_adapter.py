"""
YuE inference adapter.

This module handles the YuE-specific inference pipeline, isolating all
model-family and checkpoint-format concerns away from the HTTP layer.

Supported model families:
  - causal  (Llama-based, e.g. HKUSTAudio/YuE-s1-7B-anneal-en-icl)
  - seq2seq (encoder-decoder, e.g. hypothetical AudioCraft-style checkpoints)

Architecture detection
----------------------
Call detect_model_family(model_path) before loading to resolve which loader
and inference path to use. This reads only the config file (fast, no weights).

Inference
---------
run_yue_inference(body, output_path, state) is the single entry-point used by
generate.py. It dispatches to the correct path based on the loaded model family.
"""

from __future__ import annotations

import logging
import random
import struct
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.models import GenerateRequest

logger = logging.getLogger(__name__)

# ── Known Llama-family config class names ─────────────────────────────────────
_CAUSAL_CONFIG_CLASSES = {
    "LlamaConfig",
    "MistralConfig",
    "QwenConfig",
    "Qwen2Config",
    "FalconConfig",
    "GPTNeoXConfig",
    "GPT2Config",
    "GPTJConfig",
    "OPTConfig",
    "BloomConfig",
    "MPTConfig",
    "RWConfig",
    "InternLMConfig",
}

_SEQ2SEQ_CONFIG_CLASSES = {
    "T5Config",
    "BartConfig",
    "PegasusConfig",
    "MBartConfig",
    "MarianConfig",
    "MusicgenConfig",
    "EncoderDecoderConfig",
}


# ─── Architecture detection ───────────────────────────────────────────────────

def detect_model_family(model_path: str, trust_remote_code: bool = False) -> str:
    """
    Load only the model config (no weights) and return the resolved family string:
      'causal'  — Llama / decoder-only models
      'seq2seq' — encoder-decoder models
      'unknown' — unrecognised; caller should raise an error

    Raises RuntimeError with actionable message on any load failure.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers accelerate"
        ) from exc

    logger.info("Detecting model architecture | path=%s", model_path)
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read model config from '{model_path}'. "
            f"Ensure YUE_MODEL_PATH is a valid local directory or HuggingFace repo id. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error reading model config from '{model_path}': {exc}"
        ) from exc

    config_class = type(config).__name__
    logger.info("Detected config class: %s", config_class)

    if config_class in _CAUSAL_CONFIG_CLASSES:
        logger.info("Resolved model family: causal (Llama/decoder-only path)")
        return "causal"

    if config_class in _SEQ2SEQ_CONFIG_CLASSES:
        logger.info("Resolved model family: seq2seq (encoder-decoder path)")
        return "seq2seq"

    # Heuristic fallback: check is_decoder / model_type fields
    model_type = getattr(config, "model_type", "").lower()
    if model_type in {"llama", "mistral", "qwen2", "qwen", "falcon", "gpt_neox", "gpt2", "gptj", "opt", "bloom"}:
        logger.info("Resolved model family via model_type field: causal (%s)", model_type)
        return "causal"

    if model_type in {"t5", "bart", "pegasus", "mbart", "marian", "musicgen"}:
        logger.info("Resolved model family via model_type field: seq2seq (%s)", model_type)
        return "seq2seq"

    logger.warning(
        "Unrecognised config class '%s' (model_type='%s'). "
        "Set YUE_MODEL_FAMILY=causal or YUE_MODEL_FAMILY=seq2seq to override.",
        config_class,
        model_type,
    )
    return "unknown"


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model_and_processor(
    model_path: str,
    family: str,
    device: Any,
    torch_dtype: Any,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """
    Load model + tokenizer/processor for the given family.

    Returns (model, processor) where processor is a tokenizer for causal
    models or an AutoProcessor for seq2seq models.

    Raises RuntimeError with actionable messages on failure.
    """
    try:
        import torch  # noqa: F401 — ensure torch is available
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers accelerate"
        ) from exc

    common_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    try:
        if family == "causal":
            logger.info("Loading causal LM (AutoModelForCausalLM) | path=%s", model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)
            model = model.to(device)
            model.eval()
            processor = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Causal model + tokenizer loaded successfully")

        elif family == "seq2seq":
            logger.info("Loading seq2seq LM (AutoModelForSeq2SeqLM) | path=%s", model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common_kwargs)
            model = model.to(device)
            model.eval()
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Seq2seq model + processor loaded successfully")

        else:
            raise RuntimeError(
                f"Unsupported model family '{family}'. "
                f"Set YUE_MODEL_FAMILY=causal or YUE_MODEL_FAMILY=seq2seq explicitly. "
                f"Detected config class was not in any known family mapping."
            )

    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"Could not load checkpoint from '{model_path}'. "
            f"Verify YUE_MODEL_PATH points to a valid directory or HuggingFace repo id. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error loading model (family={family}, path={model_path}): {exc}"
        ) from exc

    return model, processor


# ─── Inference ────────────────────────────────────────────────────────────────

def run_yue_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    processor: Any,
    device: Any,
    family: str,
    sample_rate: int,
    cfg_scale: float,
    num_steps: int,
) -> None:
    """
    Blocking inference entry-point — called inside a ThreadPoolExecutor.

    Dispatches to the appropriate inference path based on model family,
    then writes the result as a WAV file at output_path.
    """
    if family == "causal":
        _run_causal_inference(
            body=body,
            output_path=output_path,
            model=model,
            tokenizer=processor,
            device=device,
            sample_rate=sample_rate,
            cfg_scale=cfg_scale,
        )
    elif family == "seq2seq":
        _run_seq2seq_inference(
            body=body,
            output_path=output_path,
            model=model,
            processor=processor,
            device=device,
            sample_rate=sample_rate,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
        )
    else:
        raise RuntimeError(
            f"Cannot run inference: unsupported model family '{family}'."
        )


# ─── Causal (Llama-based YuE) inference ──────────────────────────────────────

def _run_causal_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    tokenizer: Any,
    device: Any,
    sample_rate: int,
    cfg_scale: float,
) -> None:
    """
    YuE causal / Llama inference path.

    HKUSTAudio/YuE-s1-7B-anneal-en-icl is a decoder-only model that generates
    interleaved audio codec tokens. The pipeline:
      1. Format prompt from genre description + lyrics.
      2. Tokenize with the model's tokenizer.
      3. Run causal generation to produce audio codec tokens.
      4. Decode tokens to a waveform via the model's codec (if available)
         or fall back to writing a silent WAV for stub-compatible testing.
      5. Save WAV to output_path.
    """
    import torch

    # ── Reproducibility ────────────────────────────────────────────────────────
    _apply_seed(body.seed, device)

    # ── Build prompt ───────────────────────────────────────────────────────────
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)
    full_prompt = _format_yue_prompt(genre_prompt, body.lyrics)

    logger.debug(
        "Causal inference | job=%s | prompt_len=%d | duration=%ds",
        body.job_id,
        len(full_prompt),
        body.duration_sec,
    )

    # ── Tokenise ───────────────────────────────────────────────────────────────
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    # ── Estimate max new tokens for target duration ─────────────────────────
    # YuE-s1 uses EnCodec at ~75 tokens/second (codec frame rate).
    # Each "frame" packs 8 codebook entries, so the raw token count is:
    #   duration_sec * codec_fps * num_codebooks
    # For a simpler estimate we use 75 tokens/sec as a conservative cap.
    max_new_tokens = body.duration_sec * 75

    # ── Generate ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
        )

    # ── Decode audio ───────────────────────────────────────────────────────────
    # Attempt to use the model's built-in codec decoder (available when
    # trust_remote_code=True loads the full YuE pipeline class).
    # Fall back to a silent WAV so the HTTP contract is always satisfied.
    try:
        waveform = _decode_audio_tokens(model, output_ids, inputs["input_ids"], device, sample_rate)
        _save_waveform(waveform, output_path, sample_rate)
        logger.info("Causal inference complete | job=%s | waveform saved", body.job_id)
    except Exception as exc:
        logger.warning(
            "Audio token decoding failed (job=%s): %s. "
            "Writing silent WAV — set YUE_TRUST_REMOTE_CODE=true and ensure "
            "the YuE codec weights are downloaded to enable real audio output.",
            body.job_id,
            exc,
        )
        _write_silent_wav(output_path, body.duration_sec, sample_rate)


def _decode_audio_tokens(
    model: Any,
    output_ids: Any,
    input_ids: Any,
    device: Any,
    sample_rate: int,
) -> Any:
    """
    Attempt to extract and decode audio codec tokens from the generated ids.

    YuE models loaded with trust_remote_code=True expose a decode_audio()
    method or similar.  We try the known interface patterns and raise if
    none succeed so the caller can fall back gracefully.
    """
    import torch

    # Strip the prompt from the output to get only generated tokens
    new_tokens = output_ids[:, input_ids.shape[-1]:]

    # Pattern A — model.decode_audio() (YuE native pipeline)
    if hasattr(model, "decode_audio"):
        waveform = model.decode_audio(new_tokens)
        return waveform

    # Pattern B — model.codec.decode() (HF audio model pattern)
    if hasattr(model, "codec"):
        # codec tokens are typically [batch, num_codebooks, seq_len]
        # YuE interleaves them so we need to reshape
        codec = model.codec
        audio_codes = new_tokens.unsqueeze(0)  # add codebook dim
        waveform = codec.decode(audio_codes)[0]
        return waveform

    # Pattern C — model.audio_encoder.decode() (MusicGen-style)
    if hasattr(model, "audio_encoder"):
        audio_codes = new_tokens.unsqueeze(1)  # [batch, 1, seq_len]
        waveform = model.audio_encoder.decode(audio_codes, [None])[0]
        return waveform

    raise NotImplementedError(
        "Model does not expose a known audio decoding interface "
        "(decode_audio, codec, or audio_encoder). "
        "Ensure the checkpoint is loaded with trust_remote_code=True."
    )


def _save_waveform(waveform: Any, output_path: Path, sample_rate: int) -> None:
    """Save a torch waveform tensor to a WAV file."""
    import torch

    # Normalise shape to [channels, samples]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)  # [1, channels, samples] -> [channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]

    waveform_cpu = waveform.cpu().float()

    try:
        import torchaudio
        torchaudio.save(str(output_path), waveform_cpu, sample_rate)
        return
    except ImportError:
        pass

    # Fallback: soundfile
    import numpy as np
    import soundfile as sf
    audio_np = waveform_cpu.numpy().T  # soundfile expects [samples, channels]
    sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")


# ─── Seq2seq inference ────────────────────────────────────────────────────────

def _run_seq2seq_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    processor: Any,
    device: Any,
    sample_rate: int,
    cfg_scale: float,
    num_steps: int,
) -> None:
    """
    Seq2seq / encoder-decoder inference path.

    Used for MusicGen-style or other encoder-decoder checkpoints.
    If outputs.audio_values is present (MusicGen API), uses that directly.
    """
    import torch

    _apply_seed(body.seed, device)

    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)

    inputs = processor(
        text=genre_prompt,
        lyrics=body.lyrics if hasattr(processor, "lyrics") else None,
        return_tensors="pt",
        padding=True,
    )
    # Remove None values before moving to device
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    max_new_tokens = _duration_to_tokens(body.duration_sec, sample_rate)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
        )

    # MusicGen / AudioGen output structure
    if hasattr(outputs, "audio_values"):
        audio_values = outputs.audio_values
        out_sample_rate = getattr(outputs, "sample_rate", sample_rate)
        _save_waveform(audio_values[0], output_path, out_sample_rate)
    else:
        # Treat raw output ids as audio codec tokens
        _save_waveform(outputs.float(), output_path, sample_rate)

    logger.info("Seq2seq inference complete | job=%s", body.job_id)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _apply_seed(seed: int | None, device: Any) -> None:
    if seed is None:
        return
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _build_genre_prompt(style_preset: str, user_prompt: str) -> str:
    style_labels = {
        "pop":        "pop, contemporary, radio-friendly",
        "rock":       "rock, electric guitar, drums",
        "hip_hop":    "hip-hop, rap, urban, trap",
        "rnb":        "r&b, soul, smooth, groove",
        "electronic": "electronic, synthesizer, EDM, dance",
        "jazz":       "jazz, improvisation, swing, acoustic",
        "classical":  "classical, orchestral, symphonic, strings",
        "country":    "country, acoustic guitar, folk, storytelling",
        "folk":       "folk, acoustic, singer-songwriter, organic",
        "metal":      "metal, heavy, distorted guitar, aggressive",
        "custom":     "",
    }
    base = style_labels.get(style_preset, style_preset)
    parts = [p.strip() for p in [base, user_prompt] if p.strip()]
    return ", ".join(parts)


def _format_yue_prompt(genre_prompt: str, lyrics: str) -> str:
    """
    Format the full input prompt for YuE causal models.

    YuE-s1-7B uses a structured prompt format with explicit section markers.
    Adjust the template if your checkpoint uses a different convention.
    """
    return (
        f"[INST] Generate a song with the following style and lyrics.\n\n"
        f"Genre/Style: {genre_prompt}\n\n"
        f"Lyrics:\n{lyrics}\n [/INST]"
    )


def _duration_to_tokens(duration_sec: int, sample_rate: int) -> int:
    """Approximate token count for target duration (EnCodec @ 75 tokens/sec)."""
    codec_frame_rate = 75
    return duration_sec * codec_frame_rate


def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
    """Write a silent mono PCM WAV — stub fallback."""
    num_samples = sample_rate * duration_sec
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<h", 0) * num_samples)
