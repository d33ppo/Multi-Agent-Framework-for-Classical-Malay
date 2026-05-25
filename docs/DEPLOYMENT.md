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
LLM_PROVIDER=openrouter
ALLOW_EXTERNAL_MANUSCRIPT_API=false

OCR_MODEL_PATH=
OCR_CONFIDENCE_THRESHOLD=0.60
MAX_UPLOAD_MB=25
```

`ALLOW_EXTERNAL_MANUSCRIPT_API=false` should be the default. Set it to `true` only if manuscript data is allowed to be sent to configured external OCR/LLM/VLM providers.

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
3. Configure the Tesseract path if required by the Python wrapper.

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
- External LLM/VLM use is enabled only if allowed.
- Low-confidence and failure examples are prepared if showing error handling.

Demo flow:

1. Upload selected Jawi image/PDF.
2. Run OCR.
3. Show OCR confidence, accuracy, and CER.
4. Run full agent pipeline.
5. Show corrected OCR, romanisation, translation, summary, and extracted knowledge.
6. Submit validation review.
7. Show stored result or status in Supabase-backed UI.

## Rollback Plan

| Problem | Rollback |
| --- | --- |
| New OCR model performs worse | Use previous checkpoint and label result clearly. |
| LLM provider fails | Switch to configured fallback provider or mock/demo mode. |
| Supabase schema change breaks API | Revert schema migration and align `DATABASE_SCHEMA.md`. |
| Frontend demo breaks | Use backend API responses or prepared screenshots only as last resort. |
| External API privacy concern | Disable external manuscript API by setting `ALLOW_EXTERNAL_MANUSCRIPT_API=false`. |

## Troubleshooting

| Issue | Possible Cause | Fix |
| --- | --- | --- |
| Upload fails | Unsupported file type or file too large. | Check file extension and `MAX_UPLOAD_MB`. |
| OCR returns empty text | Image unreadable or preprocessing failed. | Try clearer scan or manual reference. |
| OCR confidence is low | Scan quality or model limitation. | Mark for review and avoid treating downstream output as final. |
| LLM/API call fails | Missing key, rate limit, provider outage. | Check `.env`, retry, or use fallback. |
| Knowledge extraction JSON invalid | LLM output did not follow schema. | Retry JSON repair or mark for human review. |
| Supabase write fails | Invalid key, table missing, network issue. | Check Supabase URL, keys, schema, and logs. |
