"""
YuE inference adapter.

Two inference paths are supported, selected by YUE_REPO_PATH:

  Path A — Native subprocess (recommended, most accurate)
    Set YUE_REPO_PATH to a cloned HKUSTAudio/YuE repository.
    The worker writes genre/lyrics to temp files and calls infer.py as a
    subprocess.  This uses the official YuE pipeline and codec loading, so
    all token-budget, prompt-format, and codec concerns are handled correctly.

  Path B — Built-in transformers (fallback, YUE_REPO_PATH not set)
    Loads the LLM with AutoModelForCausalLM and the audio codec separately.
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
load_codec() loads the xcodec_mini_infer neural audio codec from a local
path or HuggingFace repo id (m-a-p/xcodec_mini_infer).  This is the codec
that YuE was designed for — it has 8 RVQ codebooks and a 24 kHz sample rate.
Pass yue_hf_token for gated repos.
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
    common = dict(dtype=torch_dtype, low_cpu_mem_usage=True,
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
    Load the xcodec_mini_infer audio codec used to decode YuE audio tokens to waveforms.

    This is the codec YuE was trained with (m-a-p/xcodec_mini_infer).
    It has 8 RVQ codebooks, a 24 kHz sample rate, and requires trust_remote_code=True.

    Loading strategy (tried in order until one succeeds):

      1. Direct module import + model_cls.from_pretrained() (primary path for
         local directories).  Scans the codec directory for known model module
         files (model.py, modeling_xcodec.py, modeling_xcodec2.py) and known
         class names (XCodecModel, XCodec2Model, CodecModel, SoundCodec).
         Completely bypasses AutoConfig, avoiding "does not recognize
         architecture" errors.  Falls back to manual JSON config + weight
         loading if from_pretrained itself raises an exception.

      2. AutoModel with trust_remote_code + explicit CPU device_map (primary
         path for HF repo ids such as 'm-a-p/xcodec_mini_infer').
         Passing device_map={"": "cpu"} prevents the meta-device path.
         The codec is moved to the target device after loading.

      3. xcodec2 pip package (legacy; pip install xcodec2).
         Kept for compatibility with old local xcodec2 checkpoints.

    Returns None on failure — caller logs the error and continues in degraded mode.
    """
    if not codec_path:
        logger.warning(
            "YUE_CODEC_PATH is empty — codec loading skipped. "
            "Set YUE_CODEC_PATH to a local xcodec_mini_infer directory or "
            "'m-a-p/xcodec_mini_infer' to enable real audio output."
        )
        return None

    logger.info("Loading audio codec | path=%s | device=%s", codec_path, device)
    token = hf_token or None

    # ── Reset meta-device global state ────────────────────────────────────────
    # AutoModelForCausalLM.from_pretrained(..., low_cpu_mem_usage=True) uses
    # accelerate's init_empty_weights() which calls torch.set_default_device('meta').
    # On some accelerate versions this global state is NOT restored after the
    # context exits, causing every subsequent from_pretrained() call to fail with
    # "from_pretrained with a meta device context manager".
    # Explicitly resetting it here makes codec loading immune to that leak.
    _reset_default_device()

    last_exc: Optional[Exception] = None

    # ── Strategy 1: local directory loaders ───────────────────────────────────
    if os.path.isdir(codec_path):

        # 1a — Original m-a-p/xcodec research repo layout:
        #      models/soundstream_hubert_new.py + final_ckpt/config.yaml + ckpt_*.pth
        #      FileNotFoundError is raised (not logged) when layout doesn't match,
        #      so we silently skip to 1b.
        try:
            codec = _try_xcodec_local_repo(codec_path, device)
            logger.info("Audio codec loaded via xcodec repo loader | device=%s", device)
            return codec
        except FileNotFoundError:
            pass  # not an xcodec repo dir — try HF-style next
        except Exception as exc:
            logger.debug("xcodec repo load failed: %s — trying next strategy.", exc)
            last_exc = exc

        # 1b — HF-style directory: model.py / modeling_xcodec*.py + safetensors/bin weights.
        #      Used for m-a-p/xcodec_mini_infer and HKUSTAudio/xcodec2 local snapshots.
        try:
            codec = _load_codec_manual_weights(codec_path, device)
            logger.info("Audio codec loaded via manual weights | device=%s", device)
            return codec
        except Exception as exc:
            logger.debug("Manual weight load failed: %s — trying next strategy.", exc)
            last_exc = exc

    # ── Strategy 2: AutoModel with trust_remote_code (primary for HF repos) ──
    # m-a-p/xcodec_mini_infer ships custom model code in the HF repo and
    # requires trust_remote_code=True.  Explicit CPU device_map prevents the
    # meta-device path; we move to the target device after loading.
    try:
        codec = _try_automodel_cpu(codec_path, token)
        codec = codec.to(device).eval()
        logger.info("Audio codec loaded via AutoModel | device=%s", device)
        return codec
    except Exception as exc:
        logger.debug("AutoModel load failed: %s — trying next strategy.", exc)
        last_exc = exc

    # ── Strategy 3: xcodec2 pip package (legacy fallback) ─────────────────────
    # Kept for compatibility with old HKUSTAudio/xcodec2 local checkpoints that
    # were installed via `pip install xcodec2`.
    try:
        codec = _try_xcodec2_pkg(codec_path, token)
        codec = codec.to(device).eval()
        logger.info("Audio codec loaded via xcodec2 package | device=%s", device)
        return codec
    except Exception as exc:
        logger.debug("xcodec2-pkg load failed: %s", exc)
        last_exc = exc

    logger.error(
        "Failed to load audio codec from '%s': %s\n"
        "Generations will fall back to silent WAV.\n"
        "Fix options:\n"
        "  • Set YUE_CODEC_PATH='m-a-p/xcodec_mini_infer' (recommended HF repo).\n"
        "  • Download the m-a-p/xcodec_mini_infer snapshot and set YUE_CODEC_PATH\n"
        "    to that local directory (must contain model.py + weight files).\n"
        "  • Set YUE_REPO_PATH to a cloned HKUSTAudio/YuE repo for native inference.",
        codec_path, last_exc,
    )
    return None


# ─── Codec loading helpers ────────────────────────────────────────────────────

def detect_codec_n_quantizers(codec: Any) -> Optional[int]:
    """
    Inspect a loaded xcodec-family codec and return the actual number of
    RVQ/FSQ/VQ quantizer stages it uses.

    Structural paths searched (in priority order):

      xcodec_mini_infer (m-a-p) — RVQ layout:
        codec.quantizer.quantizers           — list of VQ stages, len == n_q
        codec.quantizer.codebooks            — shape (n_q, codebook_size, dim)
        codec.model.quantizer.quantizers/codebooks
        codec.vq_model.quantizer.*

      xcodec2 (HKUSTAudio) — ResidualFSQ layout:
        codec.generator.quantizer.codebooks  — shape (n_q, codebook_size, dim)
        codec.generator.quantizer.fsqs       — list of FSQ stages, len == n_q

    Returns None when the structure cannot be determined so callers fall
    back gracefully to the configured yue_codec_n_codebooks value.
    """
    def _n_from_quantizer(q_obj: Any) -> Optional[int]:
        """Extract n_quantizers from any quantizer-like object."""
        if q_obj is None:
            return None
        # Codebooks tensor: shape (n_q, codebook_size, dim) — most reliable
        codebooks = getattr(q_obj, "codebooks", None)
        if codebooks is not None:
            try:
                return int(codebooks.shape[0])
            except Exception:
                pass
        # List of quantizer stages: quantizers / fsqs / layers
        for list_attr in ("quantizers", "fsqs", "layers", "rvq_layers"):
            stages = getattr(q_obj, list_attr, None)
            if stages is not None and hasattr(stages, "__len__"):
                return len(stages)
        # num_quantizers scalar attribute
        for scalar_attr in ("num_quantizers", "n_q", "num_q", "n_codebooks"):
            val = getattr(q_obj, scalar_attr, None)
            if val is not None:
                try:
                    return int(val)
                except Exception:
                    pass
        return None

    try:
        # ── xcodec_mini_infer paths (m-a-p) ───────────────────────────────────
        # Direct quantizer on the model root (most common for xcodec_mini)
        for root_attr in (None, "model", "vq_model", "codec"):
            root = codec if root_attr is None else getattr(codec, root_attr, None)
            if root is None:
                continue
            for q_attr in ("quantizer", "vq", "residual_vq", "residual_fsq"):
                q_obj = getattr(root, q_attr, None)
                n = _n_from_quantizer(q_obj)
                if n is not None:
                    return n

        # ── xcodec2 paths (HKUSTAudio) ────────────────────────────────────────
        generator = getattr(codec, "generator", None)
        if generator is not None:
            for q_attr in ("quantizer", "vq", "residual_vq"):
                q_obj = getattr(generator, q_attr, None)
                n = _n_from_quantizer(q_obj)
                if n is not None:
                    return n

    except Exception:
        pass

    return None


def _reset_default_device() -> None:
    """
    Clear any global torch default device left by prior model loading.

    accelerate's init_empty_weights() (used by from_pretrained with
    low_cpu_mem_usage=True) sets torch.set_default_device('meta').  On some
    accelerate versions the cleanup after the context exits does not fully
    restore the global state, causing all subsequent from_pretrained() calls
    to fail.  This function resets it to a safe state.
    """
    import torch
    try:
        torch.set_default_device(None)
        logger.debug("Reset torch default device to None.")
    except (AttributeError, TypeError):
        pass  # PyTorch < 2.0 does not have set_default_device


def _load_codec_manual_weights(codec_path: str, device: Any) -> Any:
    """
    Load an xcodec-family codec from a local directory without going through AutoConfig.

    Supports m-a-p/xcodec_mini_infer (primary) and HKUSTAudio/xcodec2 (legacy).

    Steps:
      1. sys.path injection — scan the codec directory for known model modules
         and class names.  Checked in priority order:
           model.py          → XCodecModel, CodecModel, SoundCodec  (xcodec_mini)
           modeling_xcodec.py → XCodecModel, XCodec2Model           (generic xcodec)
           modeling_xcodec2.py → XCodec2Model                       (xcodec2 legacy)
      2. Call model_cls.from_pretrained(codec_dir) directly — the class carries
         its own config_class, so AutoConfig is never consulted.
      3. If from_pretrained fails, fall back to manual JSON config parsing +
         safetensors/bin weight loading (keeps device off meta throughout).
    """
    import json
    import sys
    import torch
    from pathlib import Path as _Path

    codec_dir = os.path.abspath(codec_path)
    codec_dir_path = _Path(codec_dir)

    inserted = codec_dir not in sys.path
    if inserted:
        sys.path.insert(0, codec_dir)

    try:
        # Step 1 — find model class from local directory.
        # Priority: xcodec_mini_infer (model.py) > generic xcodec > xcodec2 legacy.
        # Each entry: (module_filename_stem, [candidate_class_names])
        model_cls = None
        for module_name, class_names in [
            ("model",            ["XCodecModel", "CodecModel", "SoundCodec", "XCodec2Model"]),
            ("modeling_xcodec",  ["XCodecModel", "XCodec2Model"]),
            ("modeling_xcodec2", ["XCodec2Model", "XCodecModel"]),
            ("xcodec",           ["XCodecModel", "CodecModel"]),
        ]:
            # Only attempt if the module file actually exists in the codec dir
            # (avoids polluting sys.modules with failed imports from other packages)
            if not (codec_dir_path / f"{module_name}.py").exists():
                continue
            try:
                # Force reimport from the injected sys.path entry
                if module_name in sys.modules:
                    mod = sys.modules[module_name]
                else:
                    mod = __import__(module_name)
                for cls_name in class_names:
                    if hasattr(mod, cls_name):
                        model_cls = getattr(mod, cls_name)
                        logger.debug(
                            "Found codec class %s in %s.py", cls_name, module_name
                        )
                        break
                if model_cls:
                    break
            except ImportError:
                continue

        if model_cls is None:
            raise ImportError(
                f"No recognisable codec model class found in {codec_dir}. "
                f"Expected one of: model.py (XCodecModel/CodecModel/SoundCodec), "
                f"modeling_xcodec.py (XCodecModel), modeling_xcodec2.py (XCodec2Model)."
            )

        # Step 2 — use the locally-imported class's own from_pretrained.
        # This bypasses AutoConfig entirely: the class carries a config_class
        # attribute pointing to its own config (e.g. XCodec2Config), so
        # Transformers never needs to look up "xcodec2" in its global registry.
        # Reset meta-device state first to neutralise any accelerate leak.
        _reset_default_device()
        try:
            codec = model_cls.from_pretrained(
                codec_dir,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            codec = codec.to(device).eval()
            return codec
        except Exception as fp_exc:
            logger.debug(
                "model_cls.from_pretrained failed: %s — falling back to manual weight load.",
                fp_exc,
            )

        # Step 3 — manual config + weight loading (last resort).
        # Derive config class name from model class name (XCodec2Model → XCodec2Config).
        mod_obj = sys.modules.get(model_cls.__module__)
        config_cls_name = model_cls.__name__.replace("Model", "Config")
        config_cls = getattr(mod_obj, config_cls_name, None) if mod_obj else None

        config_json = codec_dir_path / "config.json"
        if not config_json.exists():
            raise FileNotFoundError(f"No config.json found in {codec_dir}")

        with open(config_json, encoding="utf-8") as f:
            cfg_dict = json.load(f)

        if config_cls is not None:
            # Strip Transformers bookkeeping keys that aren't __init__ params.
            skip = {"model_type", "transformers_version", "_name_or_path",
                    "architectures", "auto_map", "torch_dtype"}
            try:
                config = config_cls(**{k: v for k, v in cfg_dict.items() if k not in skip})
            except Exception:
                from transformers import PretrainedConfig
                config = PretrainedConfig.from_pretrained(codec_dir)
        else:
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_pretrained(codec_dir)

        with torch.device("cpu"):
            model = model_cls(config)

        # Step 4 — find weight files
        state_dict: dict = {}
        sf_files = sorted(codec_dir_path.glob("model*.safetensors"))
        if not sf_files:
            sf_files = sorted(codec_dir_path.glob("*.safetensors"))

        bin_files = sorted(codec_dir_path.glob("pytorch_model*.bin"))
        if not bin_files:
            bin_files = sorted(codec_dir_path.glob("*.bin"))

        if sf_files:
            try:
                from safetensors.torch import load_file as _sf_load
                for sf in sf_files:
                    state_dict.update(_sf_load(str(sf), device="cpu"))
                logger.debug("Loaded %d safetensors weight files.", len(sf_files))
            except ImportError:
                logger.debug("safetensors not installed — falling back to .bin files.")
                sf_files = []

        if not state_dict and bin_files:
            for bf in bin_files:
                state_dict.update(
                    torch.load(str(bf), map_location="cpu", weights_only=True)
                )
            logger.debug("Loaded %d .bin weight files.", len(bin_files))

        if not state_dict:
            raise FileNotFoundError(
                f"No .safetensors or .bin weight files found in {codec_dir}. "
                f"Check that the codec was downloaded completely."
            )

        # Step 5 — load state dict and move to device
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.debug("Missing keys in codec state_dict: %s", missing[:5])
        if unexpected:
            logger.debug("Unexpected keys in codec state_dict: %s", unexpected[:5])

        return model.to(device).eval()

    finally:
        if inserted and codec_dir in sys.path:
            sys.path.remove(codec_dir)


class _XCodecRepoWrapper:
    """
    Thin wrapper around the original m-a-p/xcodec SoundStream model.

    Exposes the standard decode_code(codes) interface expected by
    _decode_with_codec(), plus to() / eval() so load_codec() can move
    and freeze it like any other model.

    codec.decode_code(codes) tries:
      1. model.decode(codes)                         — combined decode
      2. model.quantizer.from_codes(codes) → decoder — two-step
      3. model.quantizer.decode(codes) → decoder     — alt two-step API
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    # ── Primary interface ──────────────────────────────────────────────────

    def decode_code(self, codes: Any) -> Any:
        """
        Decode VQ codes (batch, n_q, time) to waveform.

        SoundStream.decode() takes continuous embeddings, NOT integer codes, so the
        correct pipeline is two-step: dequantize → decode.  Direct decode(codes) is
        tried last as a fallback for models that combine both steps.
        """
        import torch
        with torch.no_grad():
            # ── Step 1 (primary): dequantize then decode ───────────────────
            # SoundStream architecture: codes → quantizer.from_codes → embeddings
            # → decoder/decode → waveform
            quant = getattr(self._model, "quantizer", None)
            dec_fn = (
                getattr(self._model, "decoder", None)
                or getattr(self._model, "decode", None)
            )
            if quant is not None and dec_fn is not None:
                for dequant_name in ("from_codes", "decode", "get_output_from_indices"):
                    dequant_fn = getattr(quant, dequant_name, None)
                    if dequant_fn is None:
                        continue
                    try:
                        emb = dequant_fn(codes)
                        # dec_fn may be a nn.Module (decoder) or a method (decode)
                        out = dec_fn(emb) if callable(dec_fn) else dec_fn
                        if isinstance(out, (tuple, list)):
                            return out[0]
                        return out
                    except Exception as e:
                        logger.debug(
                            "Two-step decode via quantizer.%s failed: %s",
                            dequant_name, e,
                        )
                        continue

            # ── Step 2 (fallback): model.decode(codes) ────────────────────
            # For models that combine quantization + decoding in one call.
            if hasattr(self._model, "decode"):
                try:
                    out = self._model.decode(codes)
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out
                except Exception as e:
                    logger.debug("SoundStream.decode(codes) failed: %s", e)

        raise RuntimeError(
            "SoundStream model does not expose a usable decode API. "
            "Expected model.quantizer.from_codes + model.decoder, "
            "or a combined model.decode(codes)."
        )

    # ── Torch-compatible helpers ───────────────────────────────────────────

    def to(self, device: Any) -> "_XCodecRepoWrapper":
        self._model = self._model.to(device)
        return self

    def eval(self) -> "_XCodecRepoWrapper":
        self._model.eval()
        return self

    def __getattr__(self, name: str) -> Any:
        # Transparent attribute delegation for quantizer/generator inspection
        return getattr(self._model, name)


def _resolve_codec_paths(obj: Any, codec_dir: str) -> Any:
    """
    Recursively walk a config dict/list and make path strings absolute.

    Handles two cases from infer.py's argument defaults:
      ./xcodec_mini_infer/<rel>  →  {codec_dir}/<rel>   (YuE repo layout)
      ./<rel>                    →  {codec_dir}/<rel>    (paths relative to codec dir itself)

    Strings that are not path-like or already absolute are returned unchanged.
    Raises nothing — worst case the original string is preserved and the model
    constructor will raise a clear error about the path.
    """
    if isinstance(obj, dict):
        return {k: _resolve_codec_paths(v, codec_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_codec_paths(v, codec_dir) for v in obj]
    if isinstance(obj, str) and not os.path.isabs(obj):
        # Strip known YuE-layout prefix first
        for prefix in ("./xcodec_mini_infer/", "xcodec_mini_infer/"):
            if obj.startswith(prefix):
                resolved = os.path.join(codec_dir, obj[len(prefix):])
                logger.debug("Config path resolved: %s → %s", obj, resolved)
                return resolved
        # Strip generic ./ relative-to-codec-dir paths
        if obj.startswith("./"):
            candidate = os.path.join(codec_dir, obj[2:])
            if os.path.exists(candidate):
                logger.debug("Config path resolved: %s → %s", obj, candidate)
                return candidate
    return obj


def _try_xcodec_local_repo(codec_path: str, device: Any) -> "_XCodecRepoWrapper":
    """
    Load xcodec_mini_infer from its research repository directory.

    Mirrors the codec loading logic in YuE's infer.py exactly:
      • sys.path: codec_dir AND codec_dir/descriptaudiocodec
      • Config: OmegaConf.load(final_ckpt/config.yaml)
      • Instantiate: eval(cfg.generator.name)(**cfg.generator.config)
      • Weights: parameter_dict['codec_model']  (then model / state_dict fallbacks)
      • Paths: resolve ./xcodec_mini_infer/ and ./ prefixes to absolute paths

    Raises FileNotFoundError when the directory does not match the expected
    layout, so callers can silently skip to the next strategy.
    """
    import sys
    import torch
    from pathlib import Path as _Path

    codec_dir = os.path.abspath(codec_path)
    codec_dir_path = _Path(codec_dir)

    # ── Layout detection (fast, no imports) ───────────────────────────────
    model_file = codec_dir_path / "models" / "soundstream_hubert_new.py"
    ckpt_dir = codec_dir_path / "final_ckpt"
    if not model_file.exists():
        raise FileNotFoundError(
            f"Not an xcodec repo dir (missing models/soundstream_hubert_new.py): {codec_dir}"
        )
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Not an xcodec repo dir (missing final_ckpt/): {codec_dir}"
        )

    # ── Checkpoint discovery ───────────────────────────────────────────────
    pth_files = sorted(ckpt_dir.glob("ckpt_*.pth"))
    if not pth_files:
        pth_files = sorted(ckpt_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoint found in {ckpt_dir}")
    ckpt_file = pth_files[-1]

    # ── Config ────────────────────────────────────────────────────────────
    config_yaml = ckpt_dir / "config.yaml"
    if not config_yaml.exists():
        raise FileNotFoundError(f"No config.yaml found in {ckpt_dir}")

    # ── sys.path (mirrors infer.py) ────────────────────────────────────────
    # infer.py appends two paths:
    #   sys.path.append(.../xcodec_mini_infer/)              ← codec_dir
    #   sys.path.append(.../xcodec_mini_infer/descriptaudiocodec/)  ← for dac
    # We keep them permanently (not removed after load) because the model's
    # forward pass may re-import from these locations at inference time.
    dac_dir = str(codec_dir_path / "descriptaudiocodec")
    for p in (codec_dir, dac_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
            logger.debug("sys.path += %s", p)

    # ── OmegaConf config load (same as infer.py: OmegaConf.load(...)) ─────
    try:
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError(
            "omegaconf is required to load xcodec config. "
            "Run: pip install omegaconf"
        )

    try:
        model_config = OmegaConf.load(str(config_yaml))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse xcodec config.yaml at {config_yaml}: {exc}"
        ) from exc

    generator_name = str(model_config.generator.name)
    logger.debug("xcodec config loaded | generator.name=%s", generator_name)

    # ── Path resolution in generator.config ───────────────────────────────
    # Convert OmegaConf DictConfig → plain dict so we can walk and patch paths.
    try:
        gen_cfg: dict = OmegaConf.to_container(model_config.generator.config, resolve=True)  # type: ignore[assignment]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse generator.config from {config_yaml}: {exc}"
        ) from exc

    gen_cfg = _resolve_codec_paths(gen_cfg, codec_dir)

    # ── Import model class ─────────────────────────────────────────────────
    # Flush stale cached module entries so a previous failed import doesn't
    # shadow the freshly sys.path-patched version.
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("models.") or mod_name == "models":
            del sys.modules[mod_name]

    try:
        from models.soundstream_hubert_new import SoundStream  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            f"Cannot import SoundStream from {codec_dir}/models/soundstream_hubert_new.py: {exc}"
        ) from exc

    # ── Instantiate (mirrors: eval(model_config.generator.name)(**...config)) ──
    # We resolve the class by name from the imported module to avoid bare eval().
    import importlib as _importlib
    _mod = _importlib.import_module("models.soundstream_hubert_new")
    cls = getattr(_mod, generator_name, None)
    if cls is None:
        raise ImportError(
            f"Class '{generator_name}' not found in models/soundstream_hubert_new.py "
            f"(available: {[n for n in dir(_mod) if not n.startswith('_')]})"
        )

    try:
        codec_model = cls(**gen_cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to instantiate {generator_name}(**generator.config): {exc}\n"
            f"  config keys: {list(gen_cfg.keys())}"
        ) from exc

    # ── Load checkpoint weights ────────────────────────────────────────────
    # infer.py: parameter_dict = torch.load(resume_path, ...)
    #           codec_model.load_state_dict(parameter_dict['codec_model'])
    logger.debug("xcodec repo: loading checkpoint %s", ckpt_file.name)
    try:
        parameter_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_file}: {exc}"
        ) from exc

    if isinstance(parameter_dict, dict):
        state_dict = (
            parameter_dict.get("codec_model")   # canonical key per infer.py
            or parameter_dict.get("model")
            or parameter_dict.get("state_dict")
            or parameter_dict.get("generator")
            or parameter_dict                    # bare state dict
        )
    else:
        state_dict = parameter_dict

    missing, unexpected = codec_model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("xcodec repo: missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.debug("xcodec repo: unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    codec_model = codec_model.to(device).eval()
    logger.info(
        "xcodec repo loaded | cls=%s | ckpt=%s | device=%s",
        generator_name, ckpt_file.name, device,
    )
    return _XCodecRepoWrapper(codec_model)


def _try_xcodec2_pkg(codec_path: str, token: Optional[str]) -> Any:
    """
    Legacy fallback: load via the xcodec2 pip package.
    Tries the xcodec2 package first, then a generic xcodec package.
    """
    # Reset meta-device state before calling from_pretrained to avoid
    # accelerate's init_empty_weights leak causing NoneType device errors.
    _reset_default_device()
    # Try xcodec2 package (HKUSTAudio/xcodec2 variant)
    try:
        from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore[import]
        return XCodec2Model.from_pretrained(codec_path, token=token)
    except ImportError:
        pass
    # Try generic xcodec package (m-a-p xcodec variant)
    try:
        from xcodec.modeling_xcodec import XCodecModel  # type: ignore[import]
        return XCodecModel.from_pretrained(codec_path, token=token)
    except ImportError:
        pass
    raise ImportError(
        "Neither 'xcodec2' nor 'xcodec' package is installed. "
        "Install one or use YUE_CODEC_PATH='m-a-p/xcodec_mini_infer' instead."
    )


def _try_automodel_cpu(codec_path: str, token: Optional[str]) -> Any:
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(
        codec_path,
        trust_remote_code=True,
        token=token,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )


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


# ─── Path B: Causal (Llama + xcodec_mini_infer) inference ────────────────────

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
            f"Audio codec not loaded ({n_audio} audio tokens extracted but cannot decode).\n"
            f"Fix: set YUE_CODEC_PATH='m-a-p/xcodec_mini_infer' or a local snapshot, "
            f"or set YUE_REPO_PATH to the HKUSTAudio/YuE repo for native inference."
        )

    waveform = _decode_with_codec(audio_codes_flat, codec, n_codebooks, device)
    _save_waveform(waveform, output_path, sample_rate)
    logger.info(
        "Causal inference complete | job=%s | audio_tokens=%d | waveform=%s",
        body.job_id, n_audio, list(waveform.shape),
    )


def _decode_with_codec(
    audio_codes_flat: Any,
    codec: Any,
    n_codebooks: int,
    device: Any,
) -> Any:
    """
    Decode a flat sequence of codec token indices to a waveform tensor.

    YuE interleaves codebooks in the token stream:
        [t0_cb0, t0_cb1 … t0_cb{N-1}, t1_cb0, t1_cb1 …]

    This function reshapes the flat sequence to (1, n_codebooks, seq_len)
    and then calls the codec's decode API.

    Supported codec APIs (tried in order):
      • codec.decode_code(codes)                  — xcodec_mini_infer and xcodec2
      • codec.decode_code(codes[0])               — unbatched variant (n_q, T)
      • codec.decode(codes)                       — generic decode
      • codec.quantizer.decode + codec.decoder    — manual two-step decode
    """
    import torch

    n_total = audio_codes_flat.shape[0]

    logger.info(
        "Decode | total_audio_tokens=%d | n_codebooks=%d (from codec auto-detect or config)",
        n_total, n_codebooks,
    )

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
        # Primary path: decode_code(codes) — used by both xcodec_mini_infer and xcodec2
        if hasattr(codec, "decode_code"):
            try:
                return codec.decode_code(codes)
            except (TypeError, RuntimeError, ValueError) as e:
                # Some xcodec variants expect (n_q, T) without batch dim
                logger.debug(
                    "decode_code(codes) failed (%s) — retrying without batch dim.", e
                )
                try:
                    return codec.decode_code(codes[0])
                except Exception:
                    raise

        # Generic decode(codes) fallback
        if hasattr(codec, "decode"):
            out = codec.decode(codes)
            return out[0] if isinstance(out, (tuple, list)) else out

        # Two-step manual decode for models without a combined decode_code
        if hasattr(codec, "quantizer") and hasattr(codec, "decoder"):
            embeddings = codec.quantizer.decode(codes)
            return codec.decoder(embeddings)

    raise RuntimeError(
        "Audio codec does not expose decode_code / decode / quantizer+decoder. "
        "Check that YUE_CODEC_PATH points to a valid xcodec_mini_infer or xcodec2 model."
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
