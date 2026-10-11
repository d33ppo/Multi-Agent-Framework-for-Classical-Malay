# Deployment Guide

## Overview

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

The prototype is intended to run locally first. Docker can be added later if needed for a cleaner demo setup.

## Local Deployment

### Prerequisites

- Python 3.10 or later.
- Node.js LTS.
- Supabase project.
- OCR dependencies for local preprocessing.
- Google Colab for heavier OCR training or GPU-based experiments.

## Environment Variables

Create a local `.env` file for development. Do not commit real secret values.

```bash
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_ANON_KEY=
SUPABASE_STORAGE_BUCKET=manuscripts

OPENROUTER_API_KEY=
OPENROUTER_MODEL=
OPENROUTER_ALLOWED_MODELS=google/gemini-2.5-flash
OPENROUTER_VISION_MODEL=
LLM_PROVIDER=openrouter
ALLOW_EXTERNAL_MANUSCRIPT_API=false
APP_MODE=

CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:8000
MAX_TEXT_CHARS=50000
TESSERACT_CMD=

JAWI_NEURAL_MODEL=mesolitica/jawi-nanot5-small-malaysian-cased

OCR_MODEL_PATH=
OCR_CONFIDENCE_THRESHOLD=0.60
MAX_UPLOAD_MB=25
```

The list above includes variables planned for Supabase, upload limits and the OCR model. The variables the code reads today are listed in `.env.example` at the repository root; copy it to `.env` to start.

### Backend Safety Settings

These variables are read by the code today. A variable that is set but empty counts as unset.

| Variable | Default | Effect |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | empty | Required for live LLM calls. Without it the API returns `503`. It never returns mock text in its place. |
| `ALLOW_EXTERNAL_MANUSCRIPT_API` | `false` | Must be `true` before any manuscript data is sent to OpenRouter. This covers the translation, summary and knowledge agents (manuscript text) and `ocr_llm.py` (page images). With `false`, the API returns `503` and `ocr_llm.py` refuses to run, even when the key is set. |
| `APP_MODE` | empty | Set to `demo_mock` to let the API serve mock output when live calls are not available, and to accept `"model": "mock"`. Mock runs report `mode: mock`. |
| `OPENROUTER_ALLOWED_MODELS` | `google/gemini-2.5-flash` | Comma-separated model IDs an API client may request. Any other model is rejected with `400`. The frontend model list is built from this value. |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash` | Model used by the CLI, and by the API when the request names no model. For the API it must also be on the allowlist. |
| `OPENROUTER_VISION_MODEL` | `google/gemini-2.5-flash` | Vision model used only by the `ocr_llm.py` OCR baseline. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:8000` | Comma-separated browser origins allowed to call the API. |
| `MAX_TEXT_CHARS` | `50000` | Longest `jawi_text` accepted by `POST /api/pipeline/run`. Longer text is rejected with `413`. |
| `TESSERACT_CMD` | empty | Full path to the Tesseract executable. When empty, `tesseract` is looked up on PATH. |

To run the live pipeline, set both `OPENROUTER_API_KEY` and `ALLOW_EXTERNAL_MANUSCRIPT_API=true`. Set the flag to `true` only if manuscript data is allowed to leave the machine.

The command-line runner (`pipeline_runner.py`) is a development tool. When live calls are not available it prints a notice and runs the agents in mock mode. The API does not do this unless `APP_MODE=demo_mock` is set.

## Romanisation Method B Dictionary

Method B needs a Jawi to Rumi lookup table built from `mesolitica/jawi-ms-en` (HuggingFace). No license is declared on that dataset card; the gathering repository (`malaysia-ai/malaysian-dataset`) is Apache-2.0 and the text is derived from Wikipedia. The files are stored under `data/`, which is gitignored.

```bash
mkdir -p data/method_b
curl -L -o data/method_b/wikipedia-1word.json https://huggingface.co/datasets/mesolitica/jawi-ms-en/resolve/main/wikipedia-1word.json
curl -L -o data/method_b/rumi-set.json https://huggingface.co/datasets/mesolitica/jawi-ms-en/resolve/main/rumi-set.json
curl -L -o data/method_b/jawi-set.json https://huggingface.co/datasets/mesolitica/jawi-ms-en/resolve/main/jawi-set.json
python jawi_dictionary.py build
```

Use `python jawi_dictionary.py build --words-only` to skip the ~59MB phrase files.

## Romanisation Method A Model

Method A runs `mesolitica/jawi-nanot5-small-malaysian-cased` (~358MB) locally with `transformers`. The weights download from HuggingFace on first use and are cached. Manuscript text is not sent to an external service for this step.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Expected local URL:

```text
http://localhost:5173
```

The exact port may change depending on the React tooling used.

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Expected local URL:

```text
http://localhost:8000
```

## Supabase Setup

1. Create a Supabase project.
2. Create a storage bucket named `manuscripts`.
3. Create the tables listed in [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md).
4. Add local environment variables for Supabase URL and keys.
5. Verify that the backend can create a `manuscript_files` record.

## OCR Dependency Setup

Install the required OCR/preprocessing dependencies in the backend environment:

```bash
pip install torch opencv-python pymupdf pillow
```

For Tesseract baseline testing:

1. Install Tesseract on the machine.
2. Install Malay/Jawi-related trained data where available.
3. If `tesseract` is not on PATH, set `TESSERACT_CMD` in `.env` to the full path of the executable, for example `C:\Program Files\Tesseract-OCR\tesseract.exe`.

The VLM OCR baseline (`ocr_llm.py`) uploads the page image to OpenRouter. It is off by default and runs only when `ALLOW_EXTERNAL_MANUSCRIPT_API=true`.

For fine-tuning OCR:

1. Use Google Colab when GPU is needed.
2. Keep model checkpoints versioned separately from raw manuscript files.
3. Export the selected model checkpoint path into `OCR_MODEL_PATH`.

## Run Commands

```bash
# Start backend
cd backend
uvicorn main:app --reload

# Start frontend
cd frontend
npm run dev

# Run backend tests
cd backend
pytest
```

## Optional Docker Deployment

Docker is optional for this FYP prototype. Add it only after the local version works.

Suggested services:

- `backend`: FastAPI app.
- `frontend`: React app.
- Supabase remains hosted through Supabase.

Example future command:

```bash
docker compose up --build
```

## Demo Checklist

Before the demo:

- `.env` exists locally and contains required keys.
- Supabase tables exist.
- Supabase storage bucket exists.
- Backend starts successfully.
- Frontend starts successfully.
- OCR model path is configured or baseline mode is clearly labelled.
- 1-3 clear demo samples are prepared.
- Gold-standard references exist for demo samples where metrics will be shown.
- External LLM/VLM use is enabled only if allowed (`ALLOW_EXTERNAL_MANUSCRIPT_API=true`). For an offline demo, set `APP_MODE=demo_mock`.
- Low-confidence and failure examples are prepared if showing error handling.

Demo flow:

1. Upload selected Jawi image/PDF.
2. Run OCR.
3. Show OCR confidence, accuracy, and CER.
4. Run full agent pipeline.
5. Show romanisation, translation, summary, and extracted knowledge.
6. Submit validation review.
7. Show stored result or status in Supabase-backed UI.

## Rollback Plan

| Problem | Rollback |
| --- | --- |
| New OCR model performs worse | Use previous checkpoint and label result clearly. |
| LLM provider fails | Failed calls are retried automatically. If the provider stays down, set `APP_MODE=demo_mock` to demonstrate the flow with mock output. |
| Supabase schema change breaks API | Revert schema migration and align `DATABASE_SCHEMA.md`. |
| Frontend demo breaks | Use backend API responses or prepared screenshots only as last resort. |
| External API privacy concern | Disable external manuscript API by setting `ALLOW_EXTERNAL_MANUSCRIPT_API=false`. |

## Troubleshooting

| Issue | Possible Cause | Fix |
| --- | --- | --- |
| Upload fails | Unsupported file type or file too large. | Check file extension and `MAX_UPLOAD_MB`. |
| OCR returns empty text | Image unreadable or preprocessing failed. | Try clearer scan or manual reference. |
| OCR confidence is low | Scan quality or model limitation. | Mark for review and avoid treating downstream output as final. |
| API returns `503` on `/api/pipeline/run` | `OPENROUTER_API_KEY` is missing, or `ALLOW_EXTERNAL_MANUSCRIPT_API` is not `true`. | Set both in `.env` and restart the backend, or set `APP_MODE=demo_mock` for a mock demo. |
| API returns `400` "model is not allowed" | The requested model is not in `OPENROUTER_ALLOWED_MODELS`. | Add the model ID to `OPENROUTER_ALLOWED_MODELS` and restart the backend. |
| API returns `413` | `jawi_text` is longer than `MAX_TEXT_CHARS`. | Send a shorter excerpt or raise `MAX_TEXT_CHARS`. |
| Browser blocks the request (CORS) | The frontend origin is not in `CORS_ALLOWED_ORIGINS`. | Add the origin and restart the backend. |
| "Tesseract executable not found" | Tesseract is not on PATH and `TESSERACT_CMD` is empty. | Set `TESSERACT_CMD` in `.env`. |
| LLM/API call fails during a run | Rate limit or provider outage. | The stage is retried up to 2 times. If it still fails, the stage is marked incomplete; run the pipeline again later. |
| Knowledge extraction JSON invalid | LLM output did not follow schema. | One repair call is made automatically. If it still fails, the stage is marked incomplete for human review. |
| Supabase write fails | Invalid key, table missing, network issue. | Check Supabase URL, keys, schema, and logs. |
