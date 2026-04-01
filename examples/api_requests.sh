#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SongGen — example API requests
# Run the service first:  cd artifacts/song-gen && python main.py
# GPU worker (optional):  cd artifacts/gpu-worker && python main.py
# ─────────────────────────────────────────────────────────────────────────────

BASE="http://localhost:8000/api/v1"
WORKER="http://localhost:9000"


# ── 1. Health check ───────────────────────────────────────────────────────────

echo "=== Main service health ==="
curl -s "$BASE/health" | python3 -m json.tool

echo ""
echo "=== GPU worker health ==="
curl -s "$WORKER/health" | python3 -m json.tool


# ── 2. List available providers ───────────────────────────────────────────────

echo ""
echo "=== Available providers ==="
curl -s "$BASE/providers" | python3 -m json.tool


# ── 3. Generate a song — local mode (no GPU required) ────────────────────────

echo ""
echo "=== Generate (local mode) ==="
RESPONSE=$(curl -s -X POST "$BASE/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Uplifting synth-pop anthem about finding yourself after heartbreak",
    "lyrics": "[Verse 1]\nStreetlights blur like watercolors on the glass\nI have been running from the echoes of the past\n\n[Chorus]\nI am the neon afterglow\nBrighter when the darkness falls below",
    "style_preset": "electronic",
    "duration_sec": 10,
    "seed": 42,
    "mode": "local"
  }')

echo "$RESPONSE" | python3 -m json.tool
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"


# ── 4. Poll job status ────────────────────────────────────────────────────────

echo ""
echo "=== Poll status ==="
sleep 2
curl -s "$BASE/jobs/$JOB_ID" | python3 -m json.tool


# ── 5. Get full result (metadata + download URL) ──────────────────────────────

echo ""
echo "=== Full result ==="
curl -s "$BASE/jobs/$JOB_ID/result" | python3 -m json.tool


# ── 6. Download the generated audio ──────────────────────────────────────────

echo ""
echo "=== Download audio ==="
curl -s "$BASE/jobs/$JOB_ID/download" -o "local_output.wav"
echo "Saved: local_output.wav ($(wc -c < local_output.wav) bytes)"


# ── 7. Generate using remote GPU worker ──────────────────────────────────────
# Make sure the GPU worker is running: cd artifacts/gpu-worker && python main.py

echo ""
echo "=== Generate (remote_gpu mode) ==="
GPU_RESPONSE=$(curl -s -X POST "$BASE/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Dark atmospheric synthwave, pulsing bassline, retrofuturistic",
    "lyrics": "[Intro]\nstatic. signal. static.\n\n[Verse 1]\nI am a pulse inside a machine\nA ghost inside the circuit unseen\n\n[Chorus]\nSignal or noise — which one am I?\nFind me in the static\nFind me in the dark",
    "style_preset": "electronic",
    "duration_sec": 15,
    "seed": 7,
    "mode": "remote_gpu"
  }')

echo "$GPU_RESPONSE" | python3 -m json.tool
GPU_JOB_ID=$(echo "$GPU_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

sleep 3
echo ""
echo "=== GPU job result ==="
curl -s "$BASE/jobs/$GPU_JOB_ID/result" | python3 -m json.tool


# ── 8. GPU worker — manual model lifecycle ────────────────────────────────────

echo ""
echo "=== Load a specific model on the GPU worker ==="
curl -s -X POST "$WORKER/load-model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "musicgen-large"}' | python3 -m json.tool

echo ""
echo "=== Unload model when done ==="
curl -s -X POST "$WORKER/unload-model" | python3 -m json.tool
