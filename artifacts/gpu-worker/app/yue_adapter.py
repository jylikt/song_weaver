"""
YuE inference adapter.

Two inference paths are supported, selected by YUE_REPO_PATH:

  Path A — Native subprocess (recommended, most accurate)
    Set YUE_REPO_PATH to a cloned HKUSTAudio/YuE repository.
    The worker writes genre/lyrics to temp files and calls infer.py as a
    subprocess.  This uses the official YuE pipeline and codec loading, so
    all token-budget, prompt-format, and codec concerns are handled correctly.

  Path B — Built-in transformers (fallback, YUE_REPO_PATH not set)
    Loads the LLM with AutoModelForCausalLM and xcodec2 separately.
    Critical correctness notes for this path:
      • text_vocab_size MUST be the BASE Llama-2 vocab (32 000), NOT
        tokenizer.vocab_size which returns the full extended vocab (83 734).
        Audio tokens occupy IDs [32000, 83734) in YuE's unified vocabulary.
      • max_new_tokens is capped at model.config.max_position_embeddings
        minus the prompt length.  Exceeding this causes context overflow,
        ~14-minute generation times, and empty audio output.

Architecture detection
----------------------
detect_model_family() reads only the model config (no weights) and maps
the config class to 'causal' or 'seq2seq'.

Codec loading
-------------
load_codec() loads the xcodec2 neural audio codec from a local path or
HuggingFace repo id (m-a-p/xcodec2).  Pass yue_hf_token for gated repos.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.models import GenerateRequest

logger = logging.getLogger(__name__)

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
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("transformers not installed. Run: pip install transformers accelerate") from exc

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
        raise RuntimeError(f"Unexpected error reading model config from '{model_path}': {exc}") from exc

    config_class = type(config).__name__
    logger.info("Detected config class: %s", config_class)

    if config_class in _CAUSAL_CONFIG_CLASSES:
        logger.info("Resolved model family: causal (Llama/decoder-only path)")
        return "causal"
    if config_class in _SEQ2SEQ_CONFIG_CLASSES:
        logger.info("Resolved model family: seq2seq (encoder-decoder path)")
        return "seq2seq"

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
    hf_token: str = "",
) -> tuple[Any, Any]:
    """Load LLM + tokenizer for the given family. Returns (model, processor)."""
    try:
        from transformers import (
            AutoModelForCausalLM, AutoModelForSeq2SeqLM,
            AutoProcessor, AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError("transformers not installed. Run: pip install transformers accelerate") from exc

    token = hf_token or None
    common = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                  trust_remote_code=trust_remote_code, token=token)

    try:
        if family == "causal":
            logger.info("Loading causal LM (AutoModelForCausalLM) | path=%s", model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, **common)
            model = model.to(device).eval()
            processor = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, token=token,
            )
            logger.info("Causal model + tokenizer loaded successfully")

        elif family == "seq2seq":
            logger.info("Loading seq2seq LM (AutoModelForSeq2SeqLM) | path=%s", model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **common)
            model = model.to(device).eval()
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, token=token,
            )
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
        raise RuntimeError(f"Unexpected error loading model (family={family}, path={model_path}): {exc}") from exc

    return model, processor


def load_codec(
    codec_path: str,
    device: Any,
    hf_token: str = "",
) -> Optional[Any]:
    """
    Load the xcodec2 audio codec used to decode YuE audio tokens to waveforms.

    Tries in order:
      1. xcodec2 package  (pip install xcodec2)
      2. AutoModel with trust_remote_code=True (works from the HF repo directly)

    Returns None on failure — caller should treat this as a degraded mode.
    """
    if not codec_path:
        logger.warning(
            "YUE_CODEC_PATH is empty — codec loading skipped. "
            "Set YUE_CODEC_PATH to a local xcodec2 directory or 'm-a-p/xcodec2' "
            "to enable real audio output."
        )
        return None

    logger.info("Loading xcodec2 codec | path=%s | device=%s", codec_path, device)
    token = hf_token or None

    try:
        try:
            from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore[import]
            codec = XCodec2Model.from_pretrained(codec_path, token=token)
        except (ImportError, ModuleNotFoundError):
            from transformers import AutoModel
            codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True, token=token)

        codec = codec.to(device).eval()
        logger.info("xcodec2 codec loaded successfully | device=%s", device)
        return codec

    except Exception as exc:
        logger.error(
            "Failed to load xcodec2 codec from '%s': %s\n"
            "Generations will fall back to silent WAV.\n"
            "Fix options:\n"
            "  • Set YUE_CODEC_PATH to a local directory with pre-downloaded codec weights.\n"
            "  • If the repo is gated, set YUE_HF_TOKEN to your HuggingFace token.\n"
            "  • Run: pip install xcodec2  (if the xcodec2 package is available).\n"
            "  • Set YUE_REPO_PATH to a cloned HKUSTAudio/YuE repo for native inference.",
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
    text_vocab_size: int,
    sample_rate: int,
    cfg_scale: float,
    num_steps: int,
    yue_repo_path: str = "",
    yue_model_path: str = "",
) -> None:
    """
    Blocking inference dispatcher — called inside a ThreadPoolExecutor.

    Prefers native subprocess (yue_repo_path set), falls back to built-in path.
    """
    # ── Path A: Native YuE subprocess (most accurate) ─────────────────────────
    if yue_repo_path and yue_model_path:
        try:
            _run_yue_subprocess(
                body=body,
                output_path=output_path,
                repo_path=yue_repo_path,
                model_path=yue_model_path,
            )
            return
        except Exception as exc:
            logger.warning(
                "Native YuE subprocess failed (job=%s): %s — "
                "falling back to built-in transformers path.",
                body.job_id, exc,
            )

    # ── Path B: Built-in transformers ─────────────────────────────────────────
    if family == "causal":
        _run_causal_inference(
            body=body,
            output_path=output_path,
            model=model,
            tokenizer=processor,
            device=device,
            codec=codec,
            n_codebooks=n_codebooks,
            text_vocab_size=text_vocab_size,
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


# ─── Path A: Native YuE subprocess ───────────────────────────────────────────

def _run_yue_subprocess(
    body: "GenerateRequest",
    output_path: Path,
    repo_path: str,
    model_path: str,
) -> None:
    """
    Call the official HKUSTAudio/YuE infer.py as a subprocess.

    The YuE repo must be cloned at yue_repo_path.  This path handles the
    codec, prompt format, and token budget correctly by delegating entirely
    to the official pipeline.

    Output: the generated WAV is moved to output_path.
    """
    repo = Path(repo_path)
    infer_script = _find_infer_script(repo)

    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)

    with tempfile.TemporaryDirectory(prefix="yue_") as tmp:
        tmp_path = Path(tmp)
        genre_file = tmp_path / "genre.txt"
        lyrics_file = tmp_path / "lyrics.txt"
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        genre_file.write_text(genre_prompt, encoding="utf-8")
        lyrics_file.write_text(body.lyrics, encoding="utf-8")

        cmd = _build_infer_cmd(
            script=infer_script,
            model_path=model_path,
            genre_file=genre_file,
            lyrics_file=lyrics_file,
            out_dir=out_dir,
            duration_sec=body.duration_sec,
            seed=body.seed,
        )

        logger.info(
            "Native YuE subprocess | job=%s | cmd=%s",
            body.job_id, " ".join(str(c) for c in cmd),
        )

        env = {**os.environ}
        result = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min hard cap
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"infer.py exited with code {result.returncode}.\n"
                f"STDOUT: {result.stdout[-2000:]}\n"
                f"STDERR: {result.stderr[-2000:]}"
            )

        # Find the generated WAV and move it to output_path
        wavs = list(out_dir.rglob("*.wav"))
        if not wavs:
            raise RuntimeError(
                f"infer.py exited 0 but no WAV found under {out_dir}.\n"
                f"STDOUT: {result.stdout[-1000:]}"
            )

        shutil.move(str(wavs[0]), str(output_path))
        logger.info(
            "Native YuE subprocess complete | job=%s | wav=%s | size=%d bytes",
            body.job_id, wavs[0].name, output_path.stat().st_size,
        )


def _find_infer_script(repo: Path) -> Path:
    """Locate the YuE inference entry-point within the cloned repo."""
    candidates = [
        repo / "infer.py",
        repo / "inference.py",
        repo / "src" / "infer.py",
        repo / "yue" / "infer.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(
        f"Cannot find infer.py in YUE_REPO_PATH='{repo}'. "
        f"Searched: {[str(c) for c in candidates]}. "
        f"Ensure YUE_REPO_PATH points to the root of the cloned HKUSTAudio/YuE repository."
    )


def _build_infer_cmd(
    script: Path,
    model_path: str,
    genre_file: Path,
    lyrics_file: Path,
    out_dir: Path,
    duration_sec: int,
    seed: Optional[int],
) -> list:
    """
    Build the subprocess command for the official YuE infer.py.

    The YuE repo has evolved over multiple versions with different argument
    names.  We cover the two most common CLI patterns and let the subprocess
    fail with a clear error if neither matches.
    """
    cmd = [
        "python", str(script),
        # Pattern A (newer YuE versions)
        "--stage1_model", model_path,
        # Pattern B fallback names are added as extras; argparse ignores unknowns
        # only if the script uses parse_known_args — this is common in research code.
        "--model_name_or_path", model_path,
        "--genre_txt", str(genre_file),
        "--lyrics_txt", str(lyrics_file),
        "--output_dir", str(out_dir),
        "--max_new_tokens", str(min(duration_sec * 100, 3000)),
        "--cuda_idx", "0",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd


# ─── Path B: Causal (Llama + xcodec2) inference ──────────────────────────────

def _run_causal_inference(
    body: "GenerateRequest",
    output_path: Path,
    model: Any,
    tokenizer: Any,
    device: Any,
    codec: Optional[Any],
    n_codebooks: int,
    text_vocab_size: int,
    sample_rate: int,
) -> None:
    """
    Built-in causal inference path for YuE-s1-7B-anneal-en-icl.

    Pipeline:
      1. Seed RNG.
      2. Format prompt (genre + lyrics).
      3. Tokenise (set pad_token=eos_token for Llama tokenizers).
      4. Calculate max_new_tokens respecting the model's context window.
      5. model.generate() → flat sequence of token IDs.
      6. Extract audio tokens: IDs in [text_vocab_size, full_vocab).
         CRITICAL: text_vocab_size must be 32000 (Llama-2 base), NOT
         tokenizer.vocab_size which returns 83734 for YuE.
      7. Subtract text_vocab_size → raw codec indices.
      8. Reshape to (1, n_codebooks, seq_len) and decode with xcodec2.
      9. Save WAV.
    """
    import torch

    _apply_seed(body.seed, device)

    # ── Prompt ────────────────────────────────────────────────────────────────
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)
    full_prompt = _format_yue_prompt(genre_prompt, body.lyrics)

    logger.info(
        "Causal inference start | job=%s | prompt_chars=%d | duration=%ds | "
        "codec_loaded=%s | text_vocab_size=%d",
        body.job_id, len(full_prompt), body.duration_sec,
        codec is not None, text_vocab_size,
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    # ── Token budget ──────────────────────────────────────────────────────────
    # Respect the model's hard context window limit.
    # xcodec2 @ 75 fps × n_codebooks tokens per frame.
    model_max_ctx = getattr(model.config, "max_position_embeddings", 16384)
    available = model_max_ctx - prompt_len - 16   # safety margin
    ideal = body.duration_sec * 75 * n_codebooks  # e.g. 30 s × 75 × 8 = 18 000
    max_new_tokens = max(1, min(ideal, available))

    if ideal > available:
        effective_sec = available // (75 * n_codebooks)
        logger.warning(
            "Requested %d tokens (%d s) but model context allows only %d tokens "
            "(~%d s) with a %d-token prompt.  Capping to %d tokens.",
            ideal, body.duration_sec, available, effective_sec,
            prompt_len, max_new_tokens,
        )

    logger.info(
        "Token budget | prompt=%d | max_new=%d | ctx_limit=%d | ideal=%d",
        prompt_len, max_new_tokens, model_max_ctx, ideal,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
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

    # ── Extract audio tokens ──────────────────────────────────────────────────
    # Strip the prompt prefix; only the newly generated tokens matter.
    new_ids = output_ids[0, prompt_len:]     # shape: (n_generated,)

    # Audio tokens are IDs in [text_vocab_size, full_vocab).
    # text_vocab_size = 32000 (Llama-2 base), NOT tokenizer.vocab_size (83734).
    audio_mask = new_ids >= text_vocab_size
    audio_codes_flat = new_ids[audio_mask] - text_vocab_size

    n_audio = audio_codes_flat.shape[0]
    logger.info(
        "Token extraction | generated=%d | audio=%d | text_vocab_size=%d",
        new_ids.shape[0], n_audio, text_vocab_size,
    )

    if n_audio == 0:
        raise RuntimeError(
            f"No audio codec tokens in model output "
            f"(generated={new_ids.shape[0]}, all < text_vocab_size={text_vocab_size}).\n"
            f"Possible causes:\n"
            f"  • YUE_TEXT_VOCAB_SIZE is wrong (check model card for base vocab size).\n"
            f"  • Prompt format mismatch — model didn't enter audio generation mode.\n"
            f"  • Set YUE_REPO_PATH to the cloned HKUSTAudio/YuE repo for native inference."
        )

    # ── Codec decode ──────────────────────────────────────────────────────────
    if codec is None:
        raise RuntimeError(
            f"xcodec2 codec not loaded ({n_audio} audio tokens extracted but cannot decode).\n"
            f"Fix: set YUE_CODEC_PATH to a local xcodec2 directory, "
            f"or set YUE_REPO_PATH to the HKUSTAudio/YuE repo for native inference."
        )

    waveform = _decode_with_xcodec2(audio_codes_flat, codec, n_codebooks, device)
    _save_waveform(waveform, output_path, sample_rate)
    logger.info(
        "Causal inference complete | job=%s | audio_tokens=%d | waveform=%s",
        body.job_id, n_audio, list(waveform.shape),
    )


def _decode_with_xcodec2(
    audio_codes_flat: Any,
    codec: Any,
    n_codebooks: int,
    device: Any,
) -> Any:
    """
    Decode a flat sequence of codec indices to a waveform tensor.

    YuE interleaves codebooks: [t0_cb0, t0_cb1…t0_cb7, t1_cb0, t1_cb1…]
    Reshape to (1, n_codebooks, seq_len) before passing to the codec.
    """
    import torch

    n_total = audio_codes_flat.shape[0]
    remainder = n_total % n_codebooks
    if remainder:
        logger.warning(
            "Audio token count %d not divisible by n_codebooks=%d — "
            "dropping last %d tokens.",
            n_total, n_codebooks, remainder,
        )
        audio_codes_flat = audio_codes_flat[: n_total - remainder]

    seq_len = audio_codes_flat.shape[0] // n_codebooks
    # (seq_len * n_codebooks,) → (1, n_codebooks, seq_len)
    codes = audio_codes_flat.reshape(seq_len, n_codebooks).T.unsqueeze(0).to(device)

    logger.info(
        "Decoding | codes_shape=%s | n_codebooks=%d | seq_len=%d",
        list(codes.shape), n_codebooks, seq_len,
    )

    with torch.no_grad():
        if hasattr(codec, "decode_code"):
            return codec.decode_code(codes)
        if hasattr(codec, "decode"):
            out = codec.decode(codes)
            return out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(codec, "quantizer") and hasattr(codec, "decoder"):
            embeddings = codec.quantizer.decode(codes)
            return codec.decoder(embeddings)

    raise RuntimeError(
        "xcodec2 codec does not expose decode_code / decode / quantizer+decoder. "
        "Check that YUE_CODEC_PATH points to a real xcodec2 model."
    )


# ─── Path B: Seq2seq inference ────────────────────────────────────────────────

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
    import torch
    _apply_seed(body.seed, device)
    genre_prompt = _build_genre_prompt(body.style_preset, body.prompt)
    inputs = processor(text=genre_prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=body.duration_sec * 75, do_sample=True)
    if hasattr(outputs, "audio_values"):
        _save_waveform(outputs.audio_values[0], output_path, getattr(outputs, "sample_rate", sample_rate))
    else:
        _save_waveform(outputs.float(), output_path, sample_rate)
    logger.info("Seq2seq inference complete | job=%s", body.job_id)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _apply_seed(seed: Optional[int], device: Any) -> None:
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
    """Prompt format for YuE-s1-7B-anneal-en-icl."""
    return f"Generate music.\nGenre: {genre_prompt}\n\nLyrics:\n{lyrics}\n"


def _save_waveform(waveform: Any, output_path: Path, sample_rate: int) -> None:
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
    sf.write(str(output_path), waveform_cpu.numpy().T, sample_rate, subtype="PCM_16")


def _write_silent_wav(path: Path, duration_sec: int, sample_rate: int = 44100) -> None:
    num_samples = sample_rate * duration_sec
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<h", 0) * num_samples)
