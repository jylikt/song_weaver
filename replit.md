# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Structure

```text
artifacts-monorepo/
├── artifacts/              # Deployable applications
│   └── api-server/         # Express API server
├── lib/                    # Shared libraries
│   ├── api-spec/           # OpenAPI spec + Orval codegen config
│   ├── api-client-react/   # Generated React Query hooks
│   ├── api-zod/            # Generated Zod schemas from OpenAPI
│   └── db/                 # Drizzle ORM schema + DB connection
├── scripts/                # Utility scripts (single workspace package)
│   └── src/                # Individual .ts scripts, run via `pnpm --filter @workspace/scripts run <script>`
├── pnpm-workspace.yaml     # pnpm workspace (artifacts/*, lib/*, lib/integrations/*, scripts)
├── tsconfig.base.json      # Shared TS options (composite, bundler resolution, es2022)
├── tsconfig.json           # Root TS project references
└── package.json            # Root package with hoisted devDeps
```

## TypeScript & Composite Projects

Every package extends `tsconfig.base.json` which sets `composite: true`. The root `tsconfig.json` lists all packages as project references. This means:

- **Always typecheck from the root** — run `pnpm run typecheck` (which runs `tsc --build --emitDeclarationOnly`). This builds the full dependency graph so that cross-package imports resolve correctly. Running `tsc` inside a single package will fail if its dependencies haven't been built yet.
- **`emitDeclarationOnly`** — we only emit `.d.ts` files during typecheck; actual JS bundling is handled by esbuild/tsx/vite...etc, not `tsc`.
- **Project references** — when package A depends on package B, A's `tsconfig.json` must list B in its `references` array. `tsc --build` uses this to determine build order and skip up-to-date packages.

## Root Scripts

- `pnpm run build` — runs `typecheck` first, then recursively runs `build` in all packages that define it
- `pnpm run typecheck` — runs `tsc --build --emitDeclarationOnly` using project references

## Python Song Generation Service

A FastAPI-based AI song generation service at `artifacts/song-gen/`.

### Architecture

```
artifacts/song-gen/
├── main.py                        # Entry point (uvicorn runner)
├── requirements.txt
├── .env.example
├── generated/                     # Output audio files saved here
└── app/
    ├── main.py                    # FastAPI app factory, CORS, route mounting
    ├── config.py                  # Settings via pydantic-settings + env vars
    ├── models/
    │   └── generation.py          # Pydantic request/response/job models
    ├── providers/
    │   ├── base.py                # BaseMusicGenerator ABC + GenerationError
    │   ├── local_generator.py     # LocalMusicGenerator (stub, add model here)
    │   ├── remote_gpu_generator.py # RemoteGpuMusicGenerator (HTTP worker)
    │   └── factory.py             # Provider factory (singleton via lru_cache)
    ├── services/
    │   ├── job_store.py           # In-memory async job store
    │   └── generation_service.py  # Job orchestration (create/run/track)
    └── routes/
        ├── health.py              # GET /api/v1/healthz
        └── generation.py          # POST /generate, GET /jobs/{id}, etc.
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/generate` | Submit job → `202` with `job_id`, `status: queued`, `mode` |
| `GET` | `/api/v1/jobs/{id}` | Poll status → `queued / running / completed / failed` |
| `GET` | `/api/v1/jobs/{id}/result` | Result + `ResultMetadata` (provider, lyrics_used, seed, etc.) |
| `GET` | `/api/v1/jobs/{id}/download` | Stream generated WAV file |
| `GET` | `/api/v1/health` | Health check with uptime, version, active provider |
| `GET` | `/api/v1/providers` | List backends with availability status |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### Request fields (`POST /api/v1/generate`)
- `prompt` — natural language description
- `lyrics` — full lyrics with `[Verse 1]`, `[Chorus]` markers
- `style_preset` — `pop / rock / hip_hop / rnb / electronic / jazz / classical / country / folk / metal / custom`
- `duration_sec` — 5–300 seconds
- `seed` — optional integer for reproducible results
- `mode` — `local` or `remote_gpu` (per-request backend selection)

### Job states: `queued → running → completed | failed`

### Key env vars

| Variable | Default | Description |
|---|---|---|
| `GENERATOR_PROVIDER` | `local` | Default backend: `local` or `remote_gpu` |
| `OUTPUT_DIR` | `generated` | Directory for saved WAV files |
| `REMOTE_WORKER_URL` | `http://localhost:9000` | URL of the gpu-worker service |
| `REMOTE_WORKER_TOKEN` | _(blank)_ | Bearer token for gpu-worker auth |
| `REMOTE_WORKER_TIMEOUT_SEC` | `300` | Request timeout in seconds |

See `.env.example` for all options.

### Workflow

- Workflow name: `Song Generation Service`
- Command: `cd artifacts/song-gen && python main.py`
- Port: 8000

---

## GPU Worker Service

A separate FastAPI inference worker at `artifacts/gpu-worker/`.  
The main app's `RemoteGpuMusicGenerator` routes `mode=remote_gpu` jobs here.

### Architecture

```
artifacts/gpu-worker/
├── main.py                  # Entry point (uvicorn runner)
├── requirements.txt
├── .env.example
├── output/                  # Generated WAV files served as static files
└── app/
    ├── main.py              # FastAPI app factory, static file mount
    ├── config.py            # Settings via pydantic-settings
    ├── state.py             # WorkerState singleton (model + GPU telemetry)
    ├── models.py            # All Pydantic request/response schemas
    ├── auth.py              # Optional Bearer-token auth dependency
    └── routes/
        ├── health.py        # GET /health
        ├── model.py         # POST /load-model, POST /unload-model
        └── generate.py      # POST /generate (stub → real model here)
```

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health, model state, GPU telemetry |
| `POST` | `/load-model` | Yes | Load model into GPU memory |
| `POST` | `/unload-model` | Yes | Free model from GPU memory |
| `POST` | `/generate` | Yes | Run inference; returns `audio_url` |
| `GET` | `/output/{file}` | No | Download generated WAV file |
| `GET` | `/docs` | No | Swagger UI |

### Model lifecycle

- `/load-model` is idempotent — calling it twice with the same model name is safe.
- If a different model is loaded, it is unloaded first (model swap).
- `/generate` auto-loads the default model if none is loaded.
- Worker state resets on restart (model must be reloaded).

### Audio codec (xcodec_mini_infer)

`app/yue_adapter.py` → `load_codec()` loads the **m-a-p/xcodec_mini_infer** codec from `YUE_CODEC_PATH`.
This is the codec YuE was trained with — 8 RVQ codebooks, 24 kHz output.

Loading strategies tried in order:
1. **Local directory + direct class import** (primary for local snapshots): Scans `model.py`, `modeling_xcodec.py`, `modeling_xcodec2.py` for known class names (`XCodecModel`, `XCodec2Model`, `CodecModel`, `SoundCodec`). Calls `from_pretrained()` directly on the found class, bypassing `AutoConfig`. Falls back to manual JSON config + safetensors/bin weight loading if needed.
2. **AutoModel with `trust_remote_code=True`** (primary for HF repo ids): Uses `AutoModel.from_pretrained(codec_path, trust_remote_code=True, device_map={"": "cpu"})`. Required for `m-a-p/xcodec_mini_infer` since it ships custom model code.
3. **Legacy xcodec2/xcodec pip package**: Tries `xcodec2.modeling_xcodec2.XCodec2Model` then `xcodec.modeling_xcodec.XCodecModel`.

**Codec quantizer auto-detection:**  
After loading, `detect_codec_n_quantizers()` introspects the codec's quantizer structure and stores the actual n_q in `WorkerState._codec_n_codebooks`. This overrides `YUE_CODEC_N_CODEBOOKS` so the token budget and decode reshape are always correct.

**Operator requirements:**
- Set `YUE_CODEC_PATH=m-a-p/xcodec_mini_infer` (public HF repo, no token needed).
- Or download the snapshot locally and set `YUE_CODEC_PATH=/root/xcodec_mini_infer`.
- Compatible versions: PyTorch ≥ 2.3, Transformers ≥ 4.44, Accelerate ≥ 0.33.

### Key env vars

| Variable | Default | Description |
|---|---|---|
| `PORT` | `9000` | Port the worker listens on |
| `WORKER_TOKEN` | _(blank)_ | Bearer token; blank = auth disabled |
| `DEFAULT_MODEL_NAME` | `yue-base` | Model name used by /load-model |
| `OUTPUT_DIR` | `output` | Directory for generated WAV files |
| `YUE_MODEL_PATH` | _(blank)_ | Local path to YuE-s1-7B checkpoint |
| `YUE_CODEC_PATH` | `m-a-p/xcodec_mini_infer` | Local path or HF repo id for the audio codec |
| `YUE_DEVICE` | `cuda` | Compute device (`cuda` or `cpu`) |
| `YUE_DTYPE` | `fp16` | Precision (`fp16`, `bf16`, `fp32`) |

### Workflow

- Workflow name: `GPU Worker Service`
- Command: `cd artifacts/gpu-worker && python main.py`
- Port: 9000

## Packages

### `artifacts/api-server` (`@workspace/api-server`)

Express 5 API server. Routes live in `src/routes/` and use `@workspace/api-zod` for request and response validation and `@workspace/db` for persistence.

- Entry: `src/index.ts` — reads `PORT`, starts Express
- App setup: `src/app.ts` — mounts CORS, JSON/urlencoded parsing, routes at `/api`
- Routes: `src/routes/index.ts` mounts sub-routers; `src/routes/health.ts` exposes `GET /health` (full path: `/api/health`)
- Depends on: `@workspace/db`, `@workspace/api-zod`
- `pnpm --filter @workspace/api-server run dev` — run the dev server
- `pnpm --filter @workspace/api-server run build` — production esbuild bundle (`dist/index.cjs`)
- Build bundles an allowlist of deps (express, cors, pg, drizzle-orm, zod, etc.) and externalizes the rest

### `lib/db` (`@workspace/db`)

Database layer using Drizzle ORM with PostgreSQL. Exports a Drizzle client instance and schema models.

- `src/index.ts` — creates a `Pool` + Drizzle instance, exports schema
- `src/schema/index.ts` — barrel re-export of all models
- `src/schema/<modelname>.ts` — table definitions with `drizzle-zod` insert schemas (no models definitions exist right now)
- `drizzle.config.ts` — Drizzle Kit config (requires `DATABASE_URL`, automatically provided by Replit)
- Exports: `.` (pool, db, schema), `./schema` (schema only)

Production migrations are handled by Replit when publishing. In development, we just use `pnpm --filter @workspace/db run push`, and we fallback to `pnpm --filter @workspace/db run push-force`.

### `lib/api-spec` (`@workspace/api-spec`)

Owns the OpenAPI 3.1 spec (`openapi.yaml`) and the Orval config (`orval.config.ts`). Running codegen produces output into two sibling packages:

1. `lib/api-client-react/src/generated/` — React Query hooks + fetch client
2. `lib/api-zod/src/generated/` — Zod schemas

Run codegen: `pnpm --filter @workspace/api-spec run codegen`

### `lib/api-zod` (`@workspace/api-zod`)

Generated Zod schemas from the OpenAPI spec (e.g. `HealthCheckResponse`). Used by `api-server` for response validation.

### `lib/api-client-react` (`@workspace/api-client-react`)

Generated React Query hooks and fetch client from the OpenAPI spec (e.g. `useHealthCheck`, `healthCheck`).

### `scripts` (`@workspace/scripts`)

Utility scripts package. Each script is a `.ts` file in `src/` with a corresponding npm script in `package.json`. Run scripts via `pnpm --filter @workspace/scripts run <script>`. Scripts can import any workspace package (e.g., `@workspace/db`) by adding it as a dependency in `scripts/package.json`.
