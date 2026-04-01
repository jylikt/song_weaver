# Запуск Song Gen (`artifacts/song-gen`)

Как поднять основной сервис SongGen, настроить локальный или удалённый режим генерации и проверить веб-интерфейс и API.

---

## 1) Что это за сервис

`song-gen` — основное FastAPI-приложение:

- **Веб-интерфейс** — корень `/`
- **REST API** — префикс `/api/v1/*`
- **Жизненный цикл задач** — `queued → running → completed | failed`
- **Скачивание результата** — `GET /api/v1/jobs/{id}/download`

Режимы генерации:

- `local` — провайдер внутри процесса `song-gen` (по умолчанию заглушка, если не подключена своя модель).
- `remote_gpu` — запросы уходят на `gpu-worker` по HTTP.

---

## 2) Требования

- Python **3.11+**
- `pip`
- Для режима `remote_gpu`: доступный по сети URL запущенного `gpu-worker`.

---

## 3) Перейти в каталог сервиса

Из корня репозитория:

```powershell
cd artifacts\song-gen
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

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5) Создать файл `.env`

### PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Отредактируйте `.env` по одному из сценариев ниже.

---

## 6) Настройка `.env`

### A) Быстрый локальный тест без удалённого воркера

```env
APP_ENV=development
LOG_LEVEL=INFO

GENERATOR_PROVIDER=local
OUTPUT_DIR=generated

REMOTE_WORKER_URL=http://localhost:9000
REMOTE_WORKER_TOKEN=
REMOTE_WORKER_TIMEOUT_SEC=300
```

Режим `mode=local` в запросе будет работать сразу; локальный провайдер остаётся заглушкой, пока вы не подключите реальную модель в коде.

### B) Локальный UI/API + генерация на удалённом GPU (`gpu-worker`)

```env
APP_ENV=development
LOG_LEVEL=INFO

GENERATOR_PROVIDER=remote_gpu
OUTPUT_DIR=generated

REMOTE_WORKER_URL=https://gpu.yourserver.com
REMOTE_WORKER_TOKEN=свой_секретный_токен
REMOTE_WORKER_TIMEOUT_SEC=300
```

Замечания:

- `REMOTE_WORKER_URL` должен быть **доступен с машины, где запущен `song-gen`**.
- `REMOTE_WORKER_TOKEN` должен **совпадать** с `WORKER_TOKEN` на `gpu-worker`, если авторизация включена.
- Даже если по умолчанию стоит `local`, в запросе можно явно передать `"mode": "remote_gpu"`.

---

## 7) Запуск сервиса

Из каталога `artifacts/song-gen`:

```powershell
python main.py
```

Альтернатива:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 8) Проверка после запуска

В браузере:

- Интерфейс: `http://localhost:8000/`
- Swagger: `http://localhost:8000/docs`
- Здоровье: `http://localhost:8000/api/v1/health`
- Провайдеры: `http://localhost:8000/api/v1/providers`

Проверьте в ответе `/api/v1/providers`:

- `active_provider`
- доступность `remote_gpu` (по умолчанию считается «настроенным», если URL задан; реальную доступность воркера лучше проверить запросом к генерации).

---

## 9) Проверка через веб-интерфейс

1. Откройте `http://localhost:8000/`
2. Заполните:
   - промпт (Prompt)
   - текст с куплетами и припевом (Lyrics), с маркерами `[Verse]`, `[Chorus]` и т.д.
   - стиль (Style)
   - длительность (Duration)
   - режим: **Local** или **GPU** (`remote_gpu`)
3. Нажмите **Generate Song**
4. Убедитесь, что обновляется панель статуса, прогресс и итоговый блок результата
5. Воспроизведите аудио во встроенном плеере
6. При необходимости скачайте **Download WAV**

---

## 10) Проверка через API (вручную)

### Отправить задачу

```bash
curl -s -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"Uplifting pop anthem about freedom",
    "lyrics":"[Verse 1]\nSky is wide\n\n[Chorus]\nWe are alive tonight",
    "style_preset":"pop",
    "duration_sec":10,
    "seed":42,
    "mode":"remote_gpu"
  }'
```

Сохраните `job_id` из ответа.

### Опрос статуса

```bash
curl -s http://localhost:8000/api/v1/jobs/<job_id>
```

### Метаданные результата

```bash
curl -s http://localhost:8000/api/v1/jobs/<job_id>/result
```

### Скачать WAV

```bash
curl -L -o song.wav http://localhost:8000/api/v1/jobs/<job_id>/download
```

---

## 11) Локальный фронт + удалённая генерация

1. Запустите `song-gen` локально (`http://localhost:8000`).
2. Запустите `gpu-worker` на удалённом сервере с GPU.
3. В локальном `artifacts/song-gen/.env` укажите:
   - `REMOTE_WORKER_URL` — URL удалённого воркера (по HTTPS или HTTP, как у вас принято)
   - `REMOTE_WORKER_TOKEN` — если на воркере задан `WORKER_TOKEN`
   - при желании `GENERATOR_PROVIDER=remote_gpu`, чтобы по умолчанию шёл удалённый режим
4. В интерфейсе выберите режим **GPU** или в API передайте `"mode":"remote_gpu"`.

---

## 12) Типичные проблемы

- Ошибка «Cannot reach worker» / недоступен воркер — неверный `REMOTE_WORKER_URL`, DNS, файрвол или VPN.
- Ответ `401` от воркера — не совпадают `REMOTE_WORKER_TOKEN` и `WORKER_TOKEN`.
- Задача падает по таймауту — увеличьте `REMOTE_WORKER_TIMEOUT_SEC` в `song-gen`.
- В `/api/v1/providers` remote_gpu «доступен», но генерация не проходит — эндпоинт провайдеров не гарантирует, что воркер реально отвечает; проверьте `GET` к воркеру `/health` и тестовый `POST /generate`.
