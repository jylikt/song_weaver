"""
YuE inference adapter.

This module handles the YuE-specific inference pipeline, isolating all
model-family and checkpoint-format concerns away from the HTTP layer.

How YuE causal inference works
-------------------------------
HKUSTAudio/YuE-s1-7B-anneal-en-icl is a Llama model fine-tuned to generate
*audio codec tokens* rather than text.  The pipeline has two distinct stages:

  Stage 1 — LLM generation (this file, _run_causal_inference):
    Formatted prompt (genre + lyrics) is tokenised with the Llama tokenizer.
    model.generate() produces a flat sequence of token IDs.
    Token IDs >= text_vocab_size are audio codec tokens; the rest are text.

  Stage 2 — Codec decoding (_decode_with_xcodec2):
    Audio token IDs are shifted down by text_vocab_size to get raw code indices.
    Codes are reshaped to (1, n_codebooks, seq_len) — xcodec2 uses 8 codebooks.
    The xcodec2 codec model decodes these to a waveform tensor.
    The waveform is saved as a WAV file.

Codec model
-----------
Load via load_codec(codec_path, device).  The default codec path is
"m-a-p/xcodec2" (HuggingFace repo id).  Set YUE_CODEC_PATH to a local
directory if you've already downloaded the codec weights.
"""

from __future__ import annotations

import logging
import random
import struct
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.models import GenerateRequest

logger = logging.getLogger(__name__)

# ── Known config class → model family mappings ────────────────────────────────
_CAUSAL_CONFIG_CLASSES = {
    "LlamaConfig", "MistralConfig", "QwenConfig", "Qwen2Config",
    "FalconConfig", "GPTNeoXConfig", "GPT2Config", "GPTJConfig",
    "OPTConfig", "BloomConfig", "MPTConfig", "RWConfig", "InternLMConfig",
}
_SEQ2SEQ_CONFIG_CLASSES = {
    "T5Config", "BartConfig", "PegasusConfig", "MBartConfig",
    "MarianConfig", "MusicgenConfig", "EncoderDecoderConfig",
}


# ─── Architecture detection ───────────────────────────────────────────────────

def detect_model_family(model_path: str, trust_remote_code: bool = False) -> str:
    """
    Inspect the model config (no weights loaded) and return the family string:
      'causal'  — Llama / decoder-only
      'seq2seq' — encoder-decoder
      'unknown' — unrecognised; caller should raise an error
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers accelerate"
        ) from exc

    logger.info("Detecting model architecture | path=%s", model_path)
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
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

    # Heuristic via model_type field
    model_type = getattr(config, "model_type", "").lower()
    if model_type in {"llama", "mistral", "qwen2", "qwen", "falcon", "gpt_neox", "gpt2", "gptj", "opt", "bloom"}:
        logger.info("Resolved model family via model_type: causal (%s)", model_type)
        return "causal"
    if model_type in {"t5", "bart", "pegasus", "mbart", "marian", "musicgen"}:
        logger.info("Resolved model family via model_type: seq2seq (%s)", model_type)
        return "seq2seq"

    logger.warning(
        "Unrecognised config class '%s' (model_type='%s'). "
        "Set YUE_MODEL_FAMILY=causal or YUE_MODEL_FAMILY=seq2seq to override.",
        config_class, model_type,
    )
    return "unknown"


# ─── Model + codec loaders ────────────────────────────────────────────────────

def load_model_and_processor(
    model_path: str,
    family: str,
    device: Any,
    torch_dtype: Any,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """
    Load LLM + tokenizer/processor for the given family.
    Returns (model, processor).
    """
    try:
        from transformers import (
            AutoModelForCausalLM, AutoModelForSeq2SeqLM,
            AutoProcessor, AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: pip install transformers accelerate"
        ) from exc

    common = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code)

    try:
        if family == "causal":
            logger.info("Loading causal LM (AutoModelForCausalLM) | path=%s", model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, **common)
            model = model.to(device).eval()
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            logger.info("Causal model + tokenizer loaded successfully")

        elif family == "seq2seq":
            logger.info("Loading seq2seq LM (AutoModelForSeq2SeqLM) | path=%s", model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common)
            model = model.to(device).eval()
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            logger.info("Seq2seq model + processor loaded successfully")

        else:
            raise RuntimeError(
                f"Unsupported model family '{family}'. "
                f"Set YUE_MODEL_FAMILY=causal or YUE_MODEL_FAMILY=seq2seq explicitly."
            )

    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"Could not load checkpoint from '{model_path}'. "
            f"Verify YUE_MODEL_PATH is a valid directory or HuggingFace repo id. "
            f"Original error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error loading model (family={family}, path={model_path}): {exc}"
        ) from exc

    return model, processor


def load_codec(codec_path: str, device: Any) -> Optional[Any]:
    """
    Load the xcodec2 codec model used to decode YuE audio tokens to waveforms.

    The codec is loaded with trust_remote_code=True because xcodec2 ships its
    own modeling code on HuggingFace (m-a-p/xcodec2).

    Returns the codec model on success, or None if loading fails (in which case
    inference will fall back to a silent WAV with a clear log message).
    """
    if not codec_path:
        logger.warning(
            "YUE_CODEC_PATH is empty — codec loading skipped. "
            "Generated audio will be a silent WAV. "
            "Set YUE_CODEC_PATH=m-a-p/xcodec2 (or a local path) to enable real audio."
        )
        return None

    logger.info("Loading xcodec2 codec | path=%s | device=%s", codec_path, device)

    # Try the xcodec2 package first (pip install xcodec2), then fall back to
    # AutoModel with trust_remote_code which works directly from the HF repo.
    try:
        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore[import]
            codec = XCodec2Model.from_pretrained(codec_path)
        except (ImportError, ModuleNotFoundError):
            from transformers import AutoModel
            codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True)

        codec = codec.to(device).eval()
        logger.info("xcodec2 codec loaded successfully | device=%s", device)
        return codec

    except Exception as exc:
        logger.error(
            "Failed to load xcodec2 codec from '%s': %s. "
            "Generations will fall back to silent WAV. "
            "Fix: ensure the codec path is correct and the model files are present.",
            codec_path, exc,
        )
        return None


# ─── Public inference entry-point ─────────────────────────────────────────────

def run_yue_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    processor: Any,
    device: Any,
    family: str,
    codec: Optional[Any],
    n_codebooks: int,
    sample_rate: int,
    cfg_scale: float,
    num_steps: int,
) -> None:
    """
    Blocking inference dispatcher — called inside a ThreadPoolExecutor.
    Routes to the correct path based on model family.
    """
    if family == "causal":
        _run_causal_inference(
            body=body,
            output_path=output_path,
            model=model,
            tokenizer=processor,
            device=device,
            codec=codec,
            n_codebooks=n_codebooks,
            sample_rate=sample_rate,
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
        raise RuntimeError(f"Cannot run inference: unsupported model family '{family}'.")


# ─── Causal (Llama-based YuE + xcodec2) inference ────────────────────────────

def _run_causal_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    tokenizer: Any,
    device: Any,
    codec: Optional[Any],
    n_codebooks: int,
    sample_rate: int,
) -> None:
    """
    Full YuE causal inference pipeline:

      1. Seed RNG for reproducibility.
      2. Format prompt (genre description + lyrics).
      3. Tokenise — set pad_token = eos_token for Llama tokenizers.
      4. Run model.generate() to produce audio codec token IDs.
      5. Extract tokens with ID >= text_vocab_size (those are audio codes).
      6. Shift down by text_vocab_size to get raw codec indices.
      7. Reshape to (1, n_codebooks, seq_len) and decode with xcodec2.
      8. Save waveform as WAV.

    If the codec is not loaded, logs a clear error and writes a silent WAV
    so the HTTP contract is always fulfilled.
    """
    import torch

    _apply_seed(body.seed, device)

    # ── Prompt ────────────────────────────────────────────────────────────────
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)
    full_prompt = _format_yue_prompt(genre_prompt, body.lyrics)

    logger.info(
        "Causal inference start | job=%s | prompt_chars=%d | duration=%ds | codec_loaded=%s",
        body.job_id, len(full_prompt), body.duration_sec, codec is not None,
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    # Llama tokenizers have no pad token by default — use eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    # ── Token budget ──────────────────────────────────────────────────────────
    # xcodec2 runs at 75 frames/sec; each frame needs n_codebooks tokens.
    # So 30 s @ 8 codebooks = 30 * 75 * 8 = 18 000 tokens.
    max_new_tokens = body.duration_sec * 75 * n_codebooks
    logger.info(
        "Generating %d tokens (%d s × 75 fps × %d codebooks)",
        max_new_tokens, body.duration_sec, n_codebooks,
    )

    # ── LLM generation ────────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # ── Extract audio codec tokens ────────────────────────────────────────────
    # Strip the prompt prefix; keep only the newly generated tokens.
    new_ids = output_ids[0, prompt_len:]  # shape: (n_generated,)

    text_vocab_size = tokenizer.vocab_size  # e.g. 32000 for Llama-2
    audio_mask = new_ids >= text_vocab_size
    audio_codes_flat = new_ids[audio_mask] - text_vocab_size  # shift to codec index space

    n_audio = audio_codes_flat.shape[0]
    logger.info(
        "Token extraction | total_generated=%d | audio_tokens=%d | text_vocab_size=%d",
        new_ids.shape[0], n_audio, text_vocab_size,
    )

    if n_audio == 0:
        raise RuntimeError(
            f"No audio codec tokens found in model output (all {new_ids.shape[0]} tokens "
            f"are below text_vocab_size={text_vocab_size}). "
            f"The prompt format or model checkpoint may be incompatible."
        )

    # ── Codec decoding ────────────────────────────────────────────────────────
    if codec is None:
        raise RuntimeError(
            f"xcodec2 codec is not loaded (YUE_CODEC_PATH is empty or loading failed). "
            f"Extracted {n_audio} audio codec tokens but cannot decode them. "
            f"Set YUE_CODEC_PATH=m-a-p/xcodec2 (or a local path) and restart the worker."
        )

    waveform = _decode_with_xcodec2(audio_codes_flat, codec, n_codebooks, device)
    _save_waveform(waveform, output_path, sample_rate)
    logger.info(
        "Causal inference complete | job=%s | audio_tokens=%d | waveform_shape=%s",
        body.job_id, n_audio, list(waveform.shape),
    )


def _decode_with_xcodec2(
    audio_codes_flat: Any,
    codec: Any,
    n_codebooks: int,
    device: Any,
) -> Any:
    """
    Decode a flat sequence of codec token indices to a waveform.

    audio_codes_flat — 1-D tensor of codec code indices (already shifted, i.e.
                       the text vocab offset has been subtracted).

    xcodec2 uses RVQ with n_codebooks (default 8) codebooks.
    YuE interleaves codes: for each time step the model generates one code per
    codebook in order, so the flat sequence looks like:
        [t0_cb0, t0_cb1, ..., t0_cb7, t1_cb0, t1_cb1, ..., t1_cb7, ...]

    We reshape to (1, n_codebooks, seq_len) and call the codec decoder.
    """
    import torch

    n_total = audio_codes_flat.shape[0]

    # Truncate to the largest multiple of n_codebooks
    remainder = n_total % n_codebooks
    if remainder:
        logger.warning(
            "Audio token count %d is not divisible by n_codebooks=%d — "
            "dropping last %d tokens.",
            n_total, n_codebooks, remainder,
        )
        audio_codes_flat = audio_codes_flat[: n_total - remainder]

    seq_len = audio_codes_flat.shape[0] // n_codebooks

    # Reshape: (seq_len * n_codebooks,) → (1, n_codebooks, seq_len)
    codes = audio_codes_flat.reshape(seq_len, n_codebooks).T.unsqueeze(0)
    # codes shape: (1, n_codebooks, seq_len)
    codes = codes.to(device)

    logger.info(
        "Decoding codec tokens | shape=%s | n_codebooks=%d | seq_len=%d",
        list(codes.shape), n_codebooks, seq_len,
    )

    with torch.no_grad():
        # Try the known xcodec2 decoding interfaces in order of preference.

        # Interface A — XCodec2Model.decode_code() (xcodec2 package / HF repo)
        if hasattr(codec, "decode_code"):
            waveform = codec.decode_code(codes)
            return waveform

        # Interface B — codec.decode() (some xcodec2 versions use this name)
        if hasattr(codec, "decode"):
            waveform = codec.decode(codes)
            # Some implementations return (waveform, ...) tuple
            if isinstance(waveform, (tuple, list)):
                waveform = waveform[0]
            return waveform

        # Interface C — codec.quantizer.decode() + codec.decoder()
        if hasattr(codec, "quantizer") and hasattr(codec, "decoder"):
            embeddings = codec.quantizer.decode(codes)
            waveform = codec.decoder(embeddings)
            return waveform

        raise RuntimeError(
            "xcodec2 codec does not expose a known decoding interface "
            "(decode_code, decode, or quantizer+decoder). "
            "Check that the loaded model at YUE_CODEC_PATH is actually xcodec2."
        )


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
    """Encoder-decoder inference path (MusicGen-style checkpoints)."""
    import torch

    _apply_seed(body.seed, device)
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)

    inputs = processor(text=genre_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    max_new_tokens = body.duration_sec * 75

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=cfg_scale,
        )

    if hasattr(outputs, "audio_values"):
        out_sr = getattr(outputs, "sample_rate", sample_rate)
        _save_waveform(outputs.audio_values[0], output_path, out_sr)
    else:
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
    Format the full input prompt for YuE-s1-7B-anneal-en-icl.

    YuE-s1 uses a genre+lyrics conditioning format. The model was trained with
    the genre description followed by formatted lyrics. Adjust if your
    checkpoint uses a different template.
    """
    return (
        f"Generate music.\n"
        f"Genre: {genre_prompt}\n\n"
        f"Lyrics:\n{lyrics}\n"
    )


def _save_waveform(waveform: Any, output_path: Path, sample_rate: int) -> None:
    """Save a torch waveform tensor to a WAV file."""
    import torch  # noqa: F401

    # Normalise to [channels, samples]
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform_cpu = waveform.cpu().float()

    try:
        import torchaudio
        torchaudio.save(str(output_path), waveform_cpu, sample_rate)
        return
    except ImportError:
        pass

    import numpy as np
    import soundfile as sf
    audio_np = waveform_cpu.numpy().T
    sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")


def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
    """Write a silent mono PCM WAV — stub/fallback only."""
    num_samples = sample_rate * duration_sec
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<h", 0) * num_samples)
