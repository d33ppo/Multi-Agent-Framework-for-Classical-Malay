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

## Status Codes

| Code | Meaning |
| --- | --- |
| 200 | Request completed. |
| 201 | Resource created. |
| 202 | Processing accepted. |
| 400 | Invalid request. |
| 404 | Resource not found. |
| 422 | Valid request format but file/content cannot be processed. |
| 500 | Internal server error. |
| 503 | External OCR/LLM/API provider unavailable. |

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
| Purpose | Run OCR correction, romanisation, translation, summarisation, and knowledge extraction. |
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
    "OCR Correction Agent",
    "Romanisation Agent",
    "Translation Agent",
    "Summarisation Agent",
    "Knowledge Extraction Agent"
  ]
}
```

**Validation rules:** OCR result must exist. Low-confidence OCR requires `allow_low_confidence=true` or human review.

**Errors:** `400`, `404`, `422`, `503`.

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
  "input_text": "Corrected or source text...",
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

**Validation rules:** Agent name must be one of `ocr-correction`, `romanisation`, `translation`, `summarisation`, `knowledge-extraction`.

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
    {"agent_name": "OCR Correction Agent", "status": "completed"},
    {"agent_name": "Romanisation Agent", "status": "completed"},
    {"agent_name": "Translation Agent", "status": "needs_review"},
    {"agent_name": "Summarisation Agent", "status": "pending"},
    {"agent_name": "Knowledge Extraction Agent", "status": "pending"}
  ]
}
```

**Errors:** `404`, `500`.
