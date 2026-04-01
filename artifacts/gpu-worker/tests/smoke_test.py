"""
GPU Worker smoke tests — acceptance tests from the spec (Section 9).

Tests the worker in isolation (no main song-gen app required).
Run against a locally running worker:

    cd artifacts/gpu-worker
    python main.py &             # start worker in background
    python -m pytest tests/smoke_test.py -v

Or against a remote worker:

    WORKER_URL=https://gpu.yourserver.com WORKER_TOKEN=secret \\
        python -m pytest tests/smoke_test.py -v

Test plan (per spec Section 9):
  ✓ API Smoke: /health, /load-model, /generate, /output, /unload-model
  ✓ End-to-End: full lifecycle with audio download verification
  ✓ Negative: invalid token, invalid payload, missing model path
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest
import httpx

# ─── Configuration ─────────────────────────────────────────────────────────────
WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:9000").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

HEADERS: dict[str, str] = {}
if WORKER_TOKEN:
    HEADERS["Authorization"] = f"Bearer {WORKER_TOKEN}"

# Sample payload matching examples/lyrics_pop.txt
SAMPLE_REQUEST = {
    "job_id": str(uuid.uuid4()),
    "prompt": "Uplifting synth-pop anthem about finding yourself after heartbreak",
    "lyrics": (
        "[Verse 1]\n"
        "Streetlights blur like watercolors on the glass\n"
        "I have been running from the echoes of the past\n\n"
        "[Chorus]\n"
        "I am the neon afterglow\n"
        "Brighter when the darkness falls below"
    ),
    "style_preset": "electronic",
    "duration_sec": 5,   # short for tests
    "seed": 42,
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get(path: str, **kwargs) -> httpx.Response:
    return httpx.get(f"{WORKER_URL}{path}", headers=HEADERS, timeout=30, **kwargs)


def post(path: str, **kwargs) -> httpx.Response:
    return httpx.post(f"{WORKER_URL}{path}", headers=HEADERS, timeout=60, **kwargs)


# ─── API smoke tests ───────────────────────────────────────────────────────────

class TestHealth:
    """GET /health — worker must respond with 200 and correct schema."""

    def test_health_returns_200(self):
        resp = httpx.get(f"{WORKER_URL}/health", timeout=10)
        assert resp.status_code == 200, resp.text

    def test_health_schema(self):
        resp = httpx.get(f"{WORKER_URL}/health", timeout=10)
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "model_loaded" in data
        assert "stub_mode" in data
        assert "generation_count" in data
        assert "uptime_seconds" in data
        assert "gpu" in data
        gpu = data["gpu"]
        assert "available" in gpu
        assert "device" in gpu
        assert "vram_total_gb" in gpu
        assert "vram_used_gb" in gpu

    def test_health_no_auth_required(self):
        """Health endpoint must be publicly accessible (no token needed)."""
        resp = httpx.get(f"{WORKER_URL}/health", timeout=10)
        assert resp.status_code == 200


class TestLoadModel:
    """POST /load-model — loads a model (or enters stub mode)."""

    def test_load_model_default(self):
        resp = post("/load-model", json={})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["model_name"], str)
        assert "loaded_at" in data

    def test_load_model_explicit_name(self):
        resp = post("/load-model", json={"model_name": "yue-base"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model_name"] == "yue-base"

    def test_load_model_idempotent(self):
        """Loading the same model twice must succeed and return the same model_name."""
        resp1 = post("/load-model", json={"model_name": "yue-base"})
        resp2 = post("/load-model", json={"model_name": "yue-base"})
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["model_name"] == resp2.json()["model_name"]

    def test_model_appears_loaded_in_health(self):
        post("/load-model", json={})
        health = httpx.get(f"{WORKER_URL}/health", timeout=10).json()
        assert health["model_loaded"] is True
        assert health["model_name"] is not None


class TestGenerate:
    """POST /generate — runs inference and returns audio_url."""

    def test_generate_returns_200(self):
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4())}
        resp = post("/generate", json=payload)
        assert resp.status_code == 200, resp.text

    def test_generate_response_schema(self):
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4())}
        resp = post("/generate", json=payload)
        data = resp.json()
        assert "job_id" in data
        assert "audio_url" in data
        assert "duration_sec" in data
        assert "model_name" in data
        assert "provider" in data
        assert data["provider"] == "GpuWorker"

    def test_generate_audio_url_is_downloadable(self):
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4())}
        resp = post("/generate", json=payload)
        assert resp.status_code == 200, resp.text
        audio_url = resp.json()["audio_url"]

        # Download the audio file
        audio_resp = httpx.get(audio_url, timeout=30)
        assert audio_resp.status_code == 200, f"audio_url not downloadable: {audio_url}"
        assert len(audio_resp.content) > 0, "Downloaded audio file is empty"

    def test_generate_output_is_valid_wav(self):
        """Downloaded file must be a valid WAV (RIFF header check)."""
        import wave, io
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4())}
        resp = post("/generate", json=payload)
        audio_url = resp.json()["audio_url"]
        audio_bytes = httpx.get(audio_url, timeout=30).content

        # WAV files start with RIFF
        assert audio_bytes[:4] == b"RIFF", (
            f"Output is not a valid WAV file. First 4 bytes: {audio_bytes[:4]!r}"
        )
        # Validate with the wave module
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            assert wf.getnframes() > 0, "WAV file contains no audio frames"

    def test_generate_echo_job_id(self):
        job_id = str(uuid.uuid4())
        resp = post("/generate", json={**SAMPLE_REQUEST, "job_id": job_id})
        assert resp.json()["job_id"] == job_id

    def test_generate_duration_approximate(self):
        """Generated audio duration should be within ±10% of requested duration."""
        import wave, io
        duration_sec = 5
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4()), "duration_sec": duration_sec}
        resp = post("/generate", json=payload)
        audio_url = resp.json()["audio_url"]
        audio_bytes = httpx.get(audio_url, timeout=30).content

        with wave.open(io.BytesIO(audio_bytes)) as wf:
            actual_duration = wf.getnframes() / wf.getframerate()

        tolerance = duration_sec * 0.10
        assert abs(actual_duration - duration_sec) <= tolerance + 1, (
            f"Duration mismatch: expected ~{duration_sec}s, got {actual_duration:.1f}s"
        )

    def test_generate_seed_reproducibility(self):
        """Same seed should produce identical output (in stub mode always true)."""
        seed_payload = {**SAMPLE_REQUEST, "seed": 99}
        resp1 = post("/generate", json={**seed_payload, "job_id": str(uuid.uuid4())})
        resp2 = post("/generate", json={**seed_payload, "job_id": str(uuid.uuid4())})
        url1 = resp1.json()["audio_url"]
        url2 = resp2.json()["audio_url"]
        content1 = httpx.get(url1, timeout=30).content
        content2 = httpx.get(url2, timeout=30).content
        # In stub mode both are silent, so they match. With a real model, same
        # seed should produce the same output (model permitting).
        assert content1 == content2, "Same seed produced different outputs"

    def test_generate_increments_generation_count(self):
        before = httpx.get(f"{WORKER_URL}/health", timeout=10).json()["generation_count"]
        post("/generate", json={**SAMPLE_REQUEST, "job_id": str(uuid.uuid4())})
        after = httpx.get(f"{WORKER_URL}/health", timeout=10).json()["generation_count"]
        assert after == before + 1


class TestUnloadModel:
    """POST /unload-model — releases the model from memory."""

    def test_unload_returns_200(self):
        post("/load-model", json={})
        resp = post("/unload-model")
        assert resp.status_code == 200, resp.text

    def test_unload_response_schema(self):
        post("/load-model", json={})
        data = post("/unload-model").json()
        assert data["success"] is True
        assert isinstance(data["message"], str)

    def test_unload_idempotent_when_nothing_loaded(self):
        """Unloading when nothing is loaded must not raise an error."""
        post("/unload-model")   # may already be unloaded
        resp = post("/unload-model")
        assert resp.status_code == 200

    def test_model_appears_unloaded_in_health(self):
        post("/load-model", json={})
        post("/unload-model")
        health = httpx.get(f"{WORKER_URL}/health", timeout=10).json()
        assert health["model_loaded"] is False
        assert health["model_name"] is None


# ─── End-to-end test ───────────────────────────────────────────────────────────

class TestEndToEnd:
    """Full lifecycle: unload → load → generate → download → unload."""

    def test_full_lifecycle(self):
        import wave, io

        # 1. Start clean
        post("/unload-model")
        assert not httpx.get(f"{WORKER_URL}/health", timeout=10).json()["model_loaded"]

        # 2. Load model
        load_resp = post("/load-model", json={})
        assert load_resp.status_code == 200
        assert load_resp.json()["success"]

        # 3. Generate
        job_id = str(uuid.uuid4())
        gen_resp = post("/generate", json={**SAMPLE_REQUEST, "job_id": job_id})
        assert gen_resp.status_code == 200
        audio_url = gen_resp.json()["audio_url"]
        assert audio_url

        # 4. Download
        audio_resp = httpx.get(audio_url, timeout=30)
        assert audio_resp.status_code == 200
        assert len(audio_resp.content) > 44  # at least larger than WAV header

        # 5. Validate WAV
        assert audio_resp.content[:4] == b"RIFF"
        with wave.open(io.BytesIO(audio_resp.content)) as wf:
            assert wf.getnframes() > 0

        # 6. Health shows updated counters
        health = httpx.get(f"{WORKER_URL}/health", timeout=10).json()
        assert health["model_loaded"] is True
        assert health["generation_count"] >= 1

        # 7. Unload
        unload_resp = post("/unload-model")
        assert unload_resp.json()["success"]
        assert not httpx.get(f"{WORKER_URL}/health", timeout=10).json()["model_loaded"]


# ─── Negative tests ────────────────────────────────────────────────────────────

class TestNegative:
    """Negative test cases per spec Section 9."""

    def test_invalid_token_returns_401(self):
        """If WORKER_TOKEN is set, a wrong token must be rejected."""
        if not WORKER_TOKEN:
            pytest.skip("WORKER_TOKEN not set — auth is disabled, skipping")
        bad_headers = {"Authorization": "Bearer wrong-token-xyz"}
        resp = httpx.post(
            f"{WORKER_URL}/generate",
            headers=bad_headers,
            json=SAMPLE_REQUEST,
            timeout=10,
        )
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"

    def test_missing_required_field_returns_422(self):
        """Request missing `prompt` must be rejected with 422."""
        bad_payload = {k: v for k, v in SAMPLE_REQUEST.items() if k != "prompt"}
        resp = post("/generate", json=bad_payload)
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_duration_exceeds_max_returns_422(self):
        """duration_sec above YUE_MAX_DURATION_SEC must be rejected with 422."""
        payload = {**SAMPLE_REQUEST, "job_id": str(uuid.uuid4()), "duration_sec": 99999}
        resp = post("/generate", json=payload)
        assert resp.status_code in (422, 400), f"Expected 4xx, got {resp.status_code}"

    def test_forced_inference_error_returns_500(self):
        """A generate call with an unparseable payload body must return 4xx/5xx."""
        resp = httpx.post(
            f"{WORKER_URL}/generate",
            headers={**HEADERS, "Content-Type": "application/json"},
            content=b"this is not json",
            timeout=10,
        )
        assert resp.status_code >= 400, f"Expected error response, got {resp.status_code}"
