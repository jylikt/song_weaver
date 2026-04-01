# Запуск GPU Worker (`artifacts/gpu-worker`)

Как поднять сервис `artifacts/gpu-worker`, настроить режим заглушки или реальный инференс YuE и проверить, что всё работает.

---

## 1) Что это за сервис

`gpu-worker` — отдельный FastAPI-сервис инференса. Его вызывает `song-gen` в режиме `remote_gpu`.

Эндпоинты:

- `GET /health`
- `POST /load-model`
- `POST /generate`
- `POST /unload-model`
- `GET /output/{filename}`

Поведение:

- Если `YUE_MODEL_PATH` **пустой** — воркер в **режиме заглушки** (тихий WAV без реальной модели).
- Если `YUE_MODEL_PATH` **задан** и зависимости/чекпоинт корректны — выполняется **реальный инференс**.

---

## 2) Требования

### Минимум

- Python **3.11+**
- `pip`

### Для реальной генерации на сервере с GPU

- Видеокарта NVIDIA и драйвер (должна работать команда `nvidia-smi`)
- Совместимая с вашей сборкой PyTorch версия CUDA

---

## 3) Перейти в каталог сервиса

Из корня репозитория:

```powershell
cd artifacts\gpu-worker
```

---

## 4) Виртуальное окружение и зависимости

### PowerShell (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS (удалённый сервер)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Создать файл `.env`

Скопировать шаблон:

### PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Дальше отредактируйте `.env` под свой сценарий.

---

## 6) Настройка `.env`

### A) Режим заглушки (без модели — для проверки API и пайплайна)

```env
PORT=9000
LOG_LEVEL=INFO
APP_ENV=development

WORKER_TOKEN=

YUE_MODEL_PATH=
YUE_DEVICE=cuda
YUE_DTYPE=fp16
YUE_MAX_DURATION_SEC=300
YUE_SAMPLE_RATE=24000
YUE_NUM_STEPS=50
YUE_CFG_SCALE=3.0
GENERATION_TIMEOUT_SEC=240

OUTPUT_DIR=output
DEFAULT_MODEL_NAME=yue-base
```

Пустой `YUE_MODEL_PATH` означает: воркер будет отдавать **тихий WAV** для теста полного цикла.

### B) Реальный YuE на удалённом GPU

Пример с локальным путём к чекпоинту:

```env
PORT=9000
LOG_LEVEL=INFO
APP_ENV=production

WORKER_TOKEN=свой_секретный_токен

YUE_MODEL_PATH=/models/yue-s2-lyric
YUE_DEVICE=cuda
YUE_DTYPE=fp16
YUE_MAX_DURATION_SEC=300
YUE_SAMPLE_RATE=24000
YUE_NUM_STEPS=50
YUE_CFG_SCALE=3.0
GENERATION_TIMEOUT_SEC=240

OUTPUT_DIR=output
DEFAULT_MODEL_NAME=yue-base
```

Или идентификатор репозитория на Hugging Face:

```env
YUE_MODEL_PATH=m-a-p/YuE-s2-lyric-audiovae-24k-v1.1
```

---

## 7) Запуск воркера

Из каталога `artifacts/gpu-worker`:

```powershell
python main.py
```

Альтернатива:

```powershell
uvicorn main:app --host 0.0.0.0 --port 9000 --reload
```

---

## 8) Проверка работоспособности

В браузере:

- `http://localhost:9000/health`
- `http://localhost:9000/docs`

Быстрая проверка через curl:

```bash
curl -s http://localhost:9000/health
```

Обратите внимание на поля:

- `model_loaded`
- `stub_mode`
- `model_name`
- `gpu.available`

---

## 9) Ручная проверка API

### Загрузка модели

Если задан `WORKER_TOKEN`:

```bash
curl -s -X POST http://localhost:9000/load-model \
  -H "Authorization: Bearer свой_секретный_токен" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Генерация

```bash
curl -s -X POST http://localhost:9000/generate \
  -H "Authorization: Bearer свой_секретный_токен" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id":"manual-test-1",
    "prompt":"Uplifting electronic pop anthem",
    "lyrics":"[Verse 1]\nWe keep moving forward\n\n[Chorus]\nTonight we rise",
    "style_preset":"electronic",
    "duration_sec":10,
    "seed":42
  }'
```

В ответе будет поле `audio_url`.

### Скачивание файла

```bash
curl -L -o out.wav http://localhost:9000/output/manual-test-1.wav
```

---

## 10) Типичные проблемы

- Ошибка загрузки модели «Failed to load model …» — неверный или недоступный `YUE_MODEL_PATH`.
- В логах откат с CUDA на CPU — указано `YUE_DEVICE=cuda`, но CUDA недоступна.
- `401` на `/generate` — задан `WORKER_TOKEN`, но в запросе нет токена или он неверный.
- Таймаут со стороны `song-gen` — увеличьте `REMOTE_WORKER_TIMEOUT_SEC` в `song-gen` и/или уменьшите длительность генерации.

---

## 11) Связка с `song-gen`

Когда воркер запущен и доступен по сети:

- Пример URL: `https://gpu.yourserver.com`
- В `song-gen` в `.env` укажите:
  - `REMOTE_WORKER_URL=https://gpu.yourserver.com`
  - `REMOTE_WORKER_TOKEN` — тот же секрет, что и `WORKER_TOKEN` на воркере (если авторизация включена)
