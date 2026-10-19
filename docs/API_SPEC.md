# API Specification

## Overview

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

The backend API is planned as a Python/FastAPI service. Routes use JSON responses and store persistent data in Supabase.

## Common Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "File type is not supported.",
    "details": {}
  }
}
```

The two routes implemented today (`/api/config` and `/api/pipeline/run`) use the FastAPI default error body instead:

```json
{
  "detail": "Model 'openai/gpt-4o' is not allowed. Allowed models: google/gemini-2.5-flash"
}
```

## Status Codes

| Code | Meaning |
| --- | --- |
| 200 | Request completed. |
| 201 | Resource created. |
| 202 | Processing accepted. |
| 400 | Invalid request. |
| 404 | Resource not found. |
| 413 | Request text is longer than the configured limit. |
| 422 | Valid request format but file/content cannot be processed. |
| 500 | Internal server error. |
| 503 | External OCR/LLM/API provider unavailable or not configured. |

## Endpoints

### Upload Document

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/documents` |
| Purpose | Upload scanned Jawi image/PDF and create `manuscript_files` record. |
| Success status | `201 CREATED` |

**Request example**

```http
POST /documents
Content-Type: multipart/form-data

file=qalam_sample.pdf
title=Majalah Qalam sample excerpt
source_collection=Majalah Qalam
year=1955
```

**Response example**

```json
{
  "id": "22222222-2222-2222-2222-222222222222",
  "title": "Majalah Qalam sample excerpt",
  "file_type": "pdf",
  "status": "pending"
}
```

**Validation rules:** File type must be PDF or image. Year should be within project scope where known. File must not exceed configured upload size.

**Errors:** `400`, `422`, `500`.

### Get Uploaded Document

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/documents/{document_id}` |
| Purpose | Retrieve uploaded document metadata from `manuscript_files`. |
| Success status | `200 OK` |

**Response example**

```json
{
  "id": "22222222-2222-2222-2222-222222222222",
  "title": "Majalah Qalam sample excerpt",
  "source_collection": "Majalah Qalam",
  "year": 1955,
  "status": "pending"
}
```

**Errors:** `404`, `500`.

### Run OCR

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/documents/{document_id}/ocr` |
| Purpose | Run OCR for an uploaded document and create `ocr_results`. |
| Success status | `202 ACCEPTED` or `200 OK` if completed synchronously. |

**Request example**

```json
{
  "provider": "fine_tuned",
  "benchmark": true
}
```

**Response example**

```json
{
  "ocr_result_id": "33333333-3333-3333-3333-333333333333",
  "provider": "fine_tuned",
  "status": "completed",
  "confidence_score": 0.67,
  "character_accuracy": 0.62,
  "cer": 0.38
}
```

**Validation rules:** Provider must be `fine_tuned`, `tesseract`, or `vlm`. Document must exist and be readable.

**Errors:** `400`, `404`, `422`, `500`, `503`.

### Get OCR Result

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/ocr-results/{ocr_result_id}` |
| Purpose | Retrieve OCR text and metrics from `ocr_results`. |
| Success status | `200 OK` |

**Response example**

```json
{
  "id": "33333333-3333-3333-3333-333333333333",
  "original_ocr_text": "نمونه جاوي...",
  "confidence_score": 0.67,
  "character_accuracy": 0.62,
  "cer": 0.38,
  "status": "completed"
}
```

**Errors:** `404`, `500`.

### Run Full Agent Pipeline

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/ocr-results/{ocr_result_id}/pipeline` |
| Purpose | Run romanisation, translation, summarisation, and knowledge extraction. |
| Success status | `202 ACCEPTED` |

**Request example**

```json
{
  "allow_low_confidence": false,
  "llm_provider": "openrouter"
}
```

**Response example**

```json
{
  "pipeline_id": "pipe-001",
  "ocr_result_id": "33333333-3333-3333-3333-333333333333",
  "status": "running",
  "agent_order": [
    "Romanisation Agent (Method A + Method B)",
    "Inter-Method Agreement Gate",
    "Translation Agent",
    "Summarisation Agent + Knowledge Extraction Agent (concurrent)"
  ]
}
```

**Validation rules:** OCR result must exist. Low-confidence OCR requires `allow_low_confidence=true` or human review.

**Errors:** `400`, `404`, `422`, `503`.

### Run Pipeline on Jawi Text (implemented prototype, streaming)

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/api/pipeline/run` |
| Purpose | Run the agent pipeline on pasted Jawi text and stream progress as Server-Sent Events. |
| Success status | `200 OK` (`text/event-stream`) |

**Request example**

```json
{
  "jawi_text": "ايبو ڤد ماس دهولو",
  "model": "google/gemini-2.5-flash"
}
```

**Validation rules**

- `jawi_text` is required, must not be blank, and may hold at most `MAX_TEXT_CHARS` characters (default 50000).
- `model` is optional. When it is omitted or empty the server uses `OPENROUTER_MODEL` if that model is on the allowlist, otherwise the first allowlist entry.
- `model` must be listed in `OPENROUTER_ALLOWED_MODELS` (default: `google/gemini-2.5-flash` only). The client cannot choose any other model.
- `"model": "mock"` (mock LLM and mock Method A) is accepted only when the server runs with `APP_MODE=demo_mock`.
- A live run needs `OPENROUTER_API_KEY` and `ALLOW_EXTERNAL_MANUSCRIPT_API=true` on the server, because the translation, summary and knowledge agents send manuscript text to OpenRouter.

All of these are checked before the stream starts, so a rejected request gets a normal HTTP error status and never a `200` stream.

**Event stream** (each line is `data: {json}`), in order:

| Event `type` | Key fields |
| --- | --- |
| `orchestrator_decision` | `label`, `skip_summary`, `word_count`, `duration` |
| `agent_start` | `agent` (`romanisation`, `translation`, `summary`, `knowledge`), `label` |
| `agent_done` | `agent`, `result`, `duration`, `incomplete`, `error`; romanisation also has `method_b_text`; translation, summary and knowledge also have `attempts` |
| `agreement_gate_done` | `status` (`passed`, `flagged`, `skipped`), `agreement`, `threshold`, `method_b_coverage`, `reason`, `needs_human_review` |
| `agent_skipped` | `agent`, `reason` (summary skipped for short text) |
| `pipeline_done` | `total_duration`, `model`, `mode`, `incomplete_stages`, `needs_human_review` |
| `error` | `message` (unexpected failure outside the stage fault handler) |

`summary` and `knowledge` `agent_done` events arrive in completion order because the two stages run concurrently.

`attempts` is the number of times the stage was called. The three LLM stages are retried up to 2 times after an exception or a timeout, so the value is 1 to 3. A stage that still fails has `incomplete: true` and an `error` message. Its `result` is then the fallback: the romanised text for translation, an empty string for summary, and `{}` for knowledge.

`mode` in `pipeline_done` is `live` or `mock`. It is `mock` only when the server runs with `APP_MODE=demo_mock`.

**Knowledge result.** The `result` of the `knowledge` `agent_done` event is validated on the server and always has this shape:

```json
{
  "people": ["Sukarno"],
  "places": ["Kuala Lumpur"],
  "dates": ["1955"],
  "organizations": [],
  "themes": ["lawatan"],
  "key_claims": [],
  "evidence": [
    {"entity": "Sukarno", "quote": "Sukarno melawat Kuala Lumpur"}
  ]
}
```

The six named fields are lists of strings. `evidence` may be empty. Each `quote` is checked on the server and is kept only if it appears word for word in the translation that was given to the agent. If the model output is still invalid after one repair call, the stage is incomplete and `result` is `{}`.

**Errors**

| Status | When |
| --- | --- |
| `400` | `jawi_text` is blank, the model is not on the allowlist, or `"mock"` is requested without `APP_MODE=demo_mock`. |
| `413` | `jawi_text` is longer than `MAX_TEXT_CHARS`. |
| `422` | The body is not valid JSON or `jawi_text` is missing (FastAPI validation). |
| `503` | Live LLM calls are not available: `OPENROUTER_API_KEY` is missing or `ALLOW_EXTERNAL_MANUSCRIPT_API` is not `true`. The server does not answer with mock text in this case. With `APP_MODE=demo_mock` the run proceeds with mock output and `mode: mock` instead. |

A failure inside a stage after the stream has started does not change the HTTP status. It is reported in that stage's `agent_done` event and in `incomplete_stages`.

### Get Frontend Configuration (implemented prototype)

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/api/config` |
| Purpose | Tell the frontend which models it may offer and how long the input may be. Contains no secrets. |
| Success status | `200 OK` |

**Response example**

```json
{
  "allowed_models": ["google/gemini-2.5-flash"],
  "default_model": "google/gemini-2.5-flash",
  "demo_mock": false,
  "max_text_chars": 50000
}
```

The frontend builds its model list from `allowed_models` and adds a mock option only when `demo_mock` is true.

### Run Individual Agent

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/ocr-results/{ocr_result_id}/agents/{agent_name}` |
| Purpose | Run one agent for testing or debugging. |
| Success status | `202 ACCEPTED` or `200 OK` if completed synchronously. |

**Request example**

```json
{
  "input_text": "Source OCR Jawi text...",
  "llm_provider": "openrouter"
}
```

**Response example**

```json
{
  "agent_run_id": "44444444-4444-4444-4444-444444444444",
  "agent_name": "Romanisation Agent",
  "status": "completed",
  "confidence_score": 0.7
}
```

**Validation rules:** Agent name must be one of `romanisation`, `translation`, `summarisation`, `knowledge-extraction`.

**Errors:** `400`, `404`, `422`, `503`.

### Get Romanisation Result

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/ocr-results/{ocr_result_id}/romanisation` |
| Purpose | Retrieve latest `romanisation_outputs` for an OCR result. |
| Success status | `200 OK` |

**Response example**

```json
{
  "romanised_text": "Ini contoh teks rumi...",
  "uncertain_tokens": ["perkataan?"],
  "notes": "Requires review."
}
```

**Errors:** `404`, `500`.

### Get Translation Result

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/ocr-results/{ocr_result_id}/translation` |
| Purpose | Retrieve latest `translation_outputs` for an OCR result. |
| Success status | `200 OK` |

**Response example**

```json
{
  "translated_text": "Ini ialah contoh terjemahan moden...",
  "target_language": "Modern Malay",
  "uncertainty_notes": "One phrase is uncertain."
}
```

**Errors:** `404`, `500`.

### Get Summary Result

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/ocr-results/{ocr_result_id}/summary` |
| Purpose | Retrieve latest `summaries` result. |
| Success status | `200 OK` |

**Response example**

```json
{
  "summary_text": "Petikan ini membincangkan isu masyarakat.",
  "key_points": ["isu masyarakat", "nasihat"]
}
```

**Errors:** `404`, `500`.

### Get Extracted Knowledge

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/ocr-results/{ocr_result_id}/knowledge` |
| Purpose | Retrieve latest `extracted_knowledge` result. |
| Success status | `200 OK` |

**Response example**

```json
{
  "entities": [{"text": "Majalah Qalam", "type": "publication"}],
  "dates": [{"text": "1955", "normalized": "1955"}],
  "places": [],
  "topics": ["pendidikan", "masyarakat"],
  "relationships": [],
  "confidence_score": 0.65
}
```

**Errors:** `404`, `500`.

### Submit Validation Review

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/documents/{document_id}/reviews` |
| Purpose | Submit human review into `validation_reviews`. |
| Success status | `201 CREATED` |

**Request example**

```json
{
  "review_target": "translation",
  "rating": 4,
  "correction_text": "Corrected translation text...",
  "comments": "Mostly accurate.",
  "status": "approved"
}
```

**Response example**

```json
{
  "review_id": "99999999-9999-9999-9999-999999999999",
  "status": "approved"
}
```

**Validation rules:** `review_target` must be a known output type. Rating must be 1-5 if provided.

**Errors:** `400`, `404`, `500`.

### Compare With Gold-Standard Reference

| Item | Detail |
| --- | --- |
| Method | `POST` |
| Route | `/documents/{document_id}/compare-gold-standard` |
| Purpose | Compare OCR or agent output with `gold_standard_references`. |
| Success status | `200 OK` |

**Request example**

```json
{
  "reference_type": "ocr",
  "target_id": "33333333-3333-3333-3333-333333333333"
}
```

**Response example**

```json
{
  "reference_type": "ocr",
  "metrics": {
    "character_accuracy": 0.62,
    "cer": 0.38
  },
  "status": "completed"
}
```

**Validation rules:** Gold-standard reference must exist for requested type.

**Errors:** `400`, `404`, `422`, `500`.

### Get Pipeline Status

| Item | Detail |
| --- | --- |
| Method | `GET` |
| Route | `/pipeline/{pipeline_id}/status` |
| Purpose | Get pipeline status using `agent_runs` and output tables. |
| Success status | `200 OK` |

**Response example**

```json
{
  "pipeline_id": "pipe-001",
  "status": "needs_review",
  "steps": [
    {"agent_name": "Romanisation Agent", "status": "completed"},
    {"agent_name": "Translation Agent", "status": "needs_review"},
    {"agent_name": "Summarisation Agent", "status": "pending"},
    {"agent_name": "Knowledge Extraction Agent", "status": "pending"}
  ]
}
```

**Errors:** `404`, `500`.
