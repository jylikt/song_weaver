# SongGen — AI Song Generation Service

A production-style Python service for **lyrics-based AI song generation**, built with FastAPI, clean layered architecture, and a pluggable provider backend system. Designed as a technical demonstration of how to scaffold, orchestrate, and extend an ML inference pipeline — with or without a real model attached.

> **Replit is used as the primary development and orchestration layer.**  
> The main API service runs here. Heavy model inference (AudioCraft, Bark, Suno-style models) is designed to run on an external GPU machine, represented by the separate `gpu-worker` service. Both communicate over HTTP with a clearly defined contract.

---

## Overview

SongGen accepts a song prompt, full lyrics, style preset, duration, and optional seed — then routes the job to one of two backends:

| Mode | Backend | Use Case |
|---|---|---|
| `local` | `LocalMusicGenerator` | Fast, no GPU. Stub outputs silent WAV. Swap in a CPU-capable model here. |
| `remote_gpu` | `RemoteGpuMusicGenerator` → gpu-worker | For large models (AudioCraft, Bark, custom fine-tunes) on separate GPU hardware. |

Jobs flow through an async pipeline: **queued → running → completed / failed**. Results include a downloadable WAV file, full metadata, and a copy of the lyrics used.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client (Browser / curl)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HTTP
┌────────────────────────────▼────────────────────────────────────────┐
│              Song Generation API  (port 8000)                        │
│                                                                      │
│  Web UI  (/index.html)           Jinja2 + Tailwind + vanilla JS     │
│  REST API (/api/v1/*)            FastAPI + Pydantic v2              │
│                                                                      │
│  ┌─────────────┐  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  Job Store  │  │  Generation Service  │  │  Provider Factory│   │
│  │ (in-memory) │  │  (async task queue)  │  │  (lru_cache)     │   │
│  └─────────────┘  └──────────┬───────────┘  └────────┬─────────┘   │
│                               │                       │             │
│              ┌────────────────┴──────────┐            │             │
│              │                           │            │             │
│    ┌─────────▼──────────┐   ┌───────────▼────────────▼──────────┐  │
│    │ LocalMusicGenerator│   │       RemoteGpuMusicGenerator      │  │
│    │  (runs in-process) │   │  GET /health → POST /load-model    │  │
│    │  [stub → real model│   │  → POST /generate → GET audio_url  │  │
│    │   goes here]       │   │                                    │  │
│    └────────────────────┘   └───────────────────┬───────────────┘  │
└────────────────────────────────────────────────  │  ────────────────┘
                                                   │  HTTP
┌──────────────────────────────────────────────────▼──────────────────┐
│              GPU Worker Service  (port 9000)                         │
│                                                                      │
│  GET  /health          Model state + GPU telemetry                  │
│  POST /load-model      Load model into VRAM                         │
│  POST /generate        Run inference → return audio_url             │
│  POST /unload-model    Free VRAM                                    │
│  GET  /output/{file}   Serve generated WAV                          │
│                                                                      │
│  WorkerState singleton: model lifecycle + simulated GPU metrics     │
│  [_run_inference() → replace with real model call]                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── README.md
├── examples/
│   ├── lyrics_pop.txt            Sample pop song lyrics
│   ├── lyrics_electronic.txt     Sample synthwave lyrics
│   ├── lyrics_orchestral.txt     Sample orchestral/cinematic lyrics
│   └── api_requests.sh           Ready-to-run curl examples
│
├── artifacts/
│   ├── song-gen/                 Main FastAPI service (port 8000)
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   ├── generated/            Output audio files
│   │   └── app/
│   │       ├── config.py         Settings (pydantic-settings)
│   │       ├── models/
│   │       │   └── generation.py Job, request, response schemas
│   │       ├── providers/
│   │       │   ├── base.py       BaseMusicGenerator ABC
│   │       │   ├── local_generator.py   ← stub + real model goes here
│   │       │   ├── remote_gpu_generator.py  ← HTTP client for gpu-worker
│   │       │   └── factory.py    Provider registry + lru_cache singletons
│   │       ├── services/
│   │       │   ├── job_store.py  In-memory async job store
│   │       │   └── generation_service.py  Job orchestration
│   │       ├── routes/
│   │       │   ├── generation.py  POST /generate, GET /jobs/{id}, etc.
│   │       │   ├── health.py      GET /api/v1/health
│   │       │   └── providers.py   GET /api/v1/providers
│   │       └── templates/
│   │           └── index.html    Jinja2 web UI
│   │
│   └── gpu-worker/               Separate inference worker (port 9000)
│       ├── main.py
│       ├── requirements.txt
│       ├── .env.example
│       ├── output/               Generated WAVs served as static files
│       └── app/
│           ├── config.py
│           ├── state.py          WorkerState — model lifecycle + GPU telemetry
│           ├── models.py         All request/response schemas
│           ├── auth.py           Optional Bearer-token auth
│           └── routes/
│               ├── health.py
│               ├── model.py      POST /load-model, POST /unload-model
│               └── generate.py  ← real inference code goes here
```

---

## Running Locally

### Prerequisites

- Python 3.11+
- pip

### 1. Start the main service

```bash
cd artifacts/song-gen
pip install -r requirements.txt
cp .env.example .env          # edit if needed
python main.py
# → http://localhost:8000
```

### 2. Start the GPU worker (optional — needed for `mode: remote_gpu`)

```bash
cd artifacts/gpu-worker
pip install -r requirements.txt
cp .env.example .env          # set WORKER_TOKEN if desired
python main.py
# → http://localhost:9000
```

### 3. Open Swagger UI

- Main API: http://localhost:8000/docs  
- GPU Worker: http://localhost:9000/docs

---

## Demo Flow

### Web UI (recommended for video demos)

1. Open **http://localhost:8000** in your browser.
2. Fill in the **Prompt** — e.g. `"Uplifting synth-pop anthem about finding yourself after heartbreak"`.
3. Paste lyrics from `examples/lyrics_pop.txt` into the **Lyrics** field.
4. Choose a **Style Preset** (e.g. Electronic) and set **Duration** to 10s.
5. Select **Mode**: `Local` (instant) or `GPU Worker` (exercises the worker integration).
6. Click **Generate Song** — the job panel appears immediately with a status badge.
7. Watch the progress bar animate while the job runs.
8. When complete, the result panel shows:
   - HTML5 audio player (plays the generated WAV).
   - Metadata card (provider, style, seed, output path).
   - Collapsible lyrics section.
9. Click **Download WAV** to save the file.

### curl Demo

```bash
# 1. Submit a generation job
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Uplifting synth-pop anthem, driving at night, neon lights",
    "lyrics": "[Verse 1]\nStreetlights blur like watercolors on the glass\nI have been running from the echoes of the past\n\n[Chorus]\nI am the neon afterglow\nBrighter when the darkness falls below",
    "style_preset": "electronic",
    "duration_sec": 10,
    "seed": 42,
    "mode": "local"
  }'

# → { "job_id": "abc-123", "status": "queued", "mode": "local" }

# 2. Poll job status
curl -s http://localhost:8000/api/v1/jobs/abc-123

# 3. Get result + metadata (once status is "completed")
curl -s http://localhost:8000/api/v1/jobs/abc-123/result

# 4. Download generated audio
curl -s http://localhost:8000/api/v1/jobs/abc-123/download -o song.wav
```

> All example requests are available as a ready-to-run script in `examples/api_requests.sh`.

---

## API Reference

### Song Generation API (port 8000)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/generate` | Submit a generation job. Returns `job_id` immediately. |
| `GET` | `/api/v1/jobs/{id}` | Poll job status: `queued / running / completed / failed`. |
| `GET` | `/api/v1/jobs/{id}/result` | Full result: download URL, metadata, lyrics used. |
| `GET` | `/api/v1/jobs/{id}/download` | Stream the generated WAV file. |
| `GET` | `/api/v1/health` | Service health, uptime, active provider. |
| `GET` | `/api/v1/providers` | List backends and their current availability. |
| `GET` | `/docs` | Interactive Swagger UI. |

#### `POST /api/v1/generate` — request body

```json
{
  "prompt":       "Uplifting synth-pop anthem about finding yourself",
  "lyrics":       "[Verse 1]\n...\n[Chorus]\n...",
  "style_preset": "electronic",
  "duration_sec": 30,
  "seed":         42,
  "mode":         "local"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | Yes | Natural-language song description. |
| `lyrics` | string | Yes | Full lyrics with `[Verse]` / `[Chorus]` / `[Bridge]` markers. |
| `style_preset` | enum | No | `pop / rock / hip_hop / rnb / electronic / jazz / classical / country / folk / metal / custom` |
| `duration_sec` | int | No | 5–300 seconds. Default: 30. |
| `seed` | int | No | Seed for reproducible output (same seed + same prompt = same output). |
| `mode` | enum | No | `local` or `remote_gpu`. Overrides `GENERATOR_PROVIDER` env var. |

### GPU Worker API (port 9000)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Model state, GPU telemetry, generation count. |
| `POST` | `/load-model` | Bearer | Load a named model. Idempotent. Swaps if different model is active. |
| `POST` | `/unload-model` | Bearer | Free the model from VRAM. |
| `POST` | `/generate` | Bearer | Run inference. Auto-loads default model if none is loaded. |
| `GET` | `/output/{filename}` | No | Download generated WAV. |

---

## Local vs Remote GPU Mode

```
mode: "local"                          mode: "remote_gpu"
─────────────────────────────────      ─────────────────────────────────────────
Runs inside the main FastAPI process   Sends work to gpu-worker over HTTP

Good for:                              Good for:
  · Fast iteration / prototyping         · Large models (AudioCraft, Bark)
  · CI / testing without GPU             · Models that need 8–80 GB VRAM
  · CPU-capable models                   · Running worker on separate GPU cloud

Integration point:                     Integration point:
  app/providers/local_generator.py       artifacts/gpu-worker/app/routes/generate.py
  → _run_model()                         → _run_inference()

Model is loaded once on startup        Model is loaded on /load-model call
(via lru_cache singleton)              and cached in WorkerState until unloaded
```

Per-request mode selection is always available via the `mode` field, regardless of the `GENERATOR_PROVIDER` server default.

---

## Integrating a Real Lyrics-to-Song Model

The system is designed so that plugging in a real open-source model requires changes to **exactly two functions** with no other code changes needed.

### Option A — Local Provider (small / CPU models)

**File:** `artifacts/song-gen/app/providers/local_generator.py`  
**Function:** `LocalMusicGenerator._run_model()`

```python
def _run_model(self, request: GenerationRequest, output_path: Path) -> None:
    # ── REPLACE THIS STUB ─────────────────────────────────────────────────
    # Example using AudioCraft MusicGen:
    #
    #   import torch, torchaudio
    #   from audiocraft.models import MusicGen
    #
    #   if request.seed is not None:
    #       torch.manual_seed(request.seed)
    #
    #   # self._model is loaded once in __init__
    #   self._model.set_generation_params(duration=request.duration_sec)
    #
    #   # Lyrics-conditioned generation (requires a fine-tuned model):
    #   wav = self._model.generate_with_lyrics(
    #       descriptions=[request.prompt],
    #       lyrics=[request.lyrics],
    #   )
    #
    #   # Prompt-only (base MusicGen, no lyrics conditioning):
    #   wav = self._model.generate([request.prompt])
    #
    #   torchaudio.save(str(output_path), wav[0].cpu(), self._model.sample_rate)
    # ──────────────────────────────────────────────────────────────────────
    self._write_silent_wav(output_path, duration_sec=request.duration_sec)
```

Also update `__init__` to load the model once:
```python
def __init__(self) -> None:
    from audiocraft.models import MusicGen
    self._model = MusicGen.get_pretrained("medium")   # or "large", "melody"
```

### Option B — GPU Worker (large models, separate hardware)

**File:** `artifacts/gpu-worker/app/routes/generate.py`  
**Function:** `_run_inference()`

Full step-by-step guidance is in the docstring. The key lines to replace:

```python
def _run_inference(body: GenerateRequest, output_path: Path) -> None:
    # ── REPLACE THIS STUB with real model inference ───────────────────────
    #
    # The model instance lives in worker_state (loaded via /load-model).
    # Access it as: worker_state._model_instance
    #
    # Example (MusicGen):
    #   worker_state._model_instance.set_generation_params(duration=body.duration_sec)
    #   wav = worker_state._model_instance.generate_with_lyrics(
    #       descriptions=[body.prompt],
    #       lyrics=[body.lyrics],
    #   )
    #   torchaudio.save(str(output_path), wav[0].cpu(), sample_rate)
    # ──────────────────────────────────────────────────────────────────────
    _write_silent_wav(output_path, duration_sec=body.duration_sec)
```

---

## Fine-Tuning & Task-Specific Adaptation

The architecture is built to support progressive model improvement. Here are the key hooks:

### Fine-tuning a base model on your lyrics corpus

1. **Prepare data** — the `lyrics` field in every request is stored verbatim in job metadata. Export it as a fine-tuning corpus:
   ```bash
   # Query all completed jobs and extract lyrics
   # (see services/job_store.py — swap InMemoryJobStore for a DB-backed one first)
   ```

2. **Fine-tune** — use your model's fine-tuning pipeline (e.g. AudioCraft's `train.py`, or LoRA for transformer-based music models). The style preset and prompt fields give you structured supervision signals.

3. **Swap in the fine-tuned weights** — in `local_generator.py` or `gpu-worker/app/state.py → load_model()`, change the pretrained model name/path to your fine-tuned checkpoint:
   ```python
   # artifacts/gpu-worker/app/state.py → load_model()
   # TODO: replace "musicgen-medium" with your fine-tuned checkpoint path
   self._model_instance = MusicGen.get_pretrained("/models/my-lyrics-finetuned-v1")
   ```

4. **Multi-model routing** — add a new `GeneratorProvider` enum value in `config.py` and a new provider class in `providers/`. Register it in `factory.py`. The `mode` field in each request can then target it explicitly.

### Adding a cloud API provider (e.g. Suno, Udio, ElevenLabs)

Create `artifacts/song-gen/app/providers/cloud_api_generator.py`:
```python
class CloudApiMusicGenerator(BaseMusicGenerator):
    async def generate_song(self, request, output_dir, job_id) -> Path:
        # POST to your cloud API, poll for result, download audio
        ...
```
Then register it in `factory.py`. No other code changes needed.

---

## Configuration Reference

### Main service (song-gen)

| Variable | Default | Description |
|---|---|---|
| `GENERATOR_PROVIDER` | `local` | Default backend: `local` or `remote_gpu` |
| `OUTPUT_DIR` | `generated` | Directory for saved WAV files |
| `REMOTE_WORKER_URL` | `http://localhost:9000` | URL of the gpu-worker |
| `REMOTE_WORKER_TOKEN` | _(blank)_ | Bearer token; leave blank to disable auth |
| `REMOTE_WORKER_TIMEOUT_SEC` | `300` | Request timeout in seconds |
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | Python log level |

### GPU worker (gpu-worker)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `9000` | Port the worker listens on |
| `WORKER_TOKEN` | _(blank)_ | Bearer token; blank = auth disabled |
| `DEFAULT_MODEL_NAME` | `musicgen-medium` | Model loaded when none is specified |
| `OUTPUT_DIR` | `output` | Directory for generated WAVs |
| `SIMULATED_VRAM_TOTAL_GB` | `24.0` | Reported VRAM total in /health |

---

## Design Notes

- **No real ML model is embedded.** Both providers write a silent WAV stub. All job orchestration, status tracking, metadata, and API flows work fully. Integrating a real model requires changing one function in one file.
- **Job store is in-memory** (`InMemoryJobStore`). For production, replace with a Redis or Postgres-backed store — the interface is a simple async `get/put` that is straightforward to swap.
- **Lyrics are first-class.** The `lyrics` field is validated, stored in job metadata, returned in results, and displayed in the UI. Every integration guide includes lyrics-conditioned generation as the primary path.
- **The worker is stateful by design.** It holds model weights in (simulated) VRAM between calls, mirrors how a real GPU worker would operate — you wouldn't reload a 7B-param model for every request.
- **Replit as orchestration layer.** The main service and GPU worker communicate over HTTP. In production, replace `REMOTE_WORKER_URL` with your GPU machine's public address. The Replit service handles UI, job queue, status polling, and result serving. The GPU machine handles only raw inference.
