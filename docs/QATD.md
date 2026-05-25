# Quality Assurance Test Document

## Test Plan

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

The QA plan verifies the FYP prototype from upload to OCR, agent processing, storage, human validation, and final structured output. Testing focuses on correctness, reliability, and clear failure handling instead of production-scale load testing.

## Test Scope

### In Scope

- Document upload validation.
- OCR extraction and benchmarking.
- OCR confidence and CER calculation.
- Agent output format and sequence.
- Romanisation, translation, summarisation, and knowledge extraction quality checks.
- Supabase storage for files, OCR results, agent runs, outputs, reviews, and logs.
- API endpoint validation.
- Failure and retry behaviour.
- Basic usability for FYP demo.

### Out of Scope

- Full production load testing.
- Handwritten manuscript OCR.
- Evaluation across all Jawi archives.
- Translation to languages other than modern Malay.

## Test Environment

| Area | Environment |
| --- | --- |
| Frontend | React local development server. |
| Backend | FastAPI local server. |
| Database | Supabase project for prototype testing. |
| OCR training | Google Colab with PyTorch where GPU is required. |
| OCR preprocessing | Python, OpenCV, PyMuPDF. |
| Agents | LangChain with mocked LLM for automated tests and configured LLM API for selected manual tests. |
| Test runner | Pytest or equivalent Python test runner. |

## Test Data Strategy

- Use selected scanned Majalah Qalam image/PDF excerpts.
- Prepare manual gold-standard references for OCR text, romanisation, translation, summary, and knowledge extraction.
- Keep small samples for quick local testing.
- Avoid uploading manuscript data to external services unless explicitly configured.

## Test Sample Size Proposal

| Stage | Sample Size | Purpose |
| --- | --- | --- |
| Early pilot | 10-15 samples | Verify upload, OCR, agent flow, and review workflow. |
| Final prototype evaluation | 30-50 manuscript excerpts | Evaluate OCR and agent metrics for FYP reporting. |
| Final confirmed count | To be confirmed | Supervisor or library collaborator approval required. |

## OCR Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| OCR-01 | Clear scanned Jawi page | Upload clear sample and run fine-tuned OCR. | OCR text is produced with confidence and stored in `ocr_results`. |  |
| OCR-02 | Blurry scanned Jawi page | Upload blurry sample and run OCR. | OCR either produces low-confidence result or marks `needs_review`. |  |
| OCR-03 | Low OCR confidence | Use sample expected to produce weak OCR. | System preserves OCR text and warns that downstream output may be unreliable. |  |
| OCR-04 | Missing text region | Upload page with cropped/missing text. | System logs issue and marks result for review. |  |
| OCR-05 | Failed OCR extraction | Use unreadable image. | System returns clear failure and does not create false completed output. |  |
| OCR-06 | Benchmark with gold standard | Run OCR where manual transcription exists. | Accuracy and CER are calculated. |  |
| OCR-07 | Compare OCR providers | Run fine-tuned OCR, Tesseract, and VLM OCR where configured. | Benchmark table is generated with comparable metrics. |  |

## Agent Pipeline Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| AG-01 | Run full pipeline | Run pipeline on valid OCR result. | All agents complete and outputs are stored. |  |
| AG-02 | Stop on missing OCR | Run pipeline without OCR result. | API returns validation error. |  |
| AG-03 | Low-confidence OCR pipeline | Run pipeline on low-confidence OCR. | System blocks or flags run unless user allows continuation. |  |
| AG-04 | Invalid agent JSON output | Mock invalid JSON from Knowledge Extraction Agent. | System retries JSON repair and then marks failed/needs_review if still invalid. |  |
| AG-05 | Failed LLM/API call | Mock unavailable LLM provider. | System retries or uses configured fallback; otherwise status becomes `failed_external_api`. |  |
| AG-06 | Agent contradiction | Mock translation contradicting source meaning. | System flags contradiction for human review. |  |

## Romanisation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| ROM-01 | Romanise valid corrected Jawi | Run Romanisation Agent. | Romanised text is produced and stored in `romanisation_outputs`. |  |
| ROM-02 | Uncertain Jawi token | Include ambiguous OCR token. | Output includes uncertain token or note. |  |
| ROM-03 | Compare to gold standard | Compare romanisation with reference. | BLEU or human review result is recorded. |  |

## Translation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| TR-01 | Translate valid romanised text | Run Translation Agent. | Modern Malay translation is produced. |  |
| TR-02 | Hallucinated translation | Mock translation with unsupported added meaning. | Human review or consistency check flags output. |  |
| TR-03 | Compare to gold standard | Compare translation with reference. | BLEU or human review result is recorded. |  |

## Summarisation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| SUM-01 | Summarise valid translation | Run Summarisation Agent. | Summary and key points are stored. |  |
| SUM-02 | Summary misses main point | Compare summary with reference or reviewer judgement. | Output is marked for review if incomplete. |  |
| SUM-03 | ROUGE evaluation | Compare summary with gold-standard summary. | ROUGE result or reviewer note is recorded. |  |

## Knowledge Extraction Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| KE-01 | Extract entities and topics | Run Knowledge Extraction Agent. | JSON includes entities, dates, places, topics, relationships, and confidence. |  |
| KE-02 | Inconsistent extracted entity | Mock entity not supported by translation/source. | System flags entity for review. |  |
| KE-03 | Invalid JSON extraction | Return malformed JSON. | Retry and then mark failed/needs_review if unresolved. |  |
| KE-04 | Compare to annotation | Compare extracted knowledge with gold-standard annotation. | Structured output accuracy is calculated or reviewed. |  |

## Failure/Error Test Cases

| ID | Failure | Expected Handling | Pass/Fail |
| --- | --- | --- | --- |
| ERR-01 | Unsupported file type | Return `400` and do not store invalid file. |  |
| ERR-02 | Unreadable manuscript image | Return `422`, log error, and request better sample/manual review. |  |
| ERR-03 | External API unavailable | Retry or use configured fallback; otherwise mark `failed_external_api`. |  |
| ERR-04 | Supabase write failure | Retry and log failure. |  |
| ERR-05 | Missing gold standard | Run pipeline but mark metrics unavailable. |  |
| ERR-06 | Human reviewer correction | Store correction in `validation_reviews` and preserve original output. |  |

## Usability Test Cases

| ID | Test Case | Expected Result | Pass/Fail |
| --- | --- | --- | --- |
| UX-01 | Upload document from frontend | User can upload without technical steps. |  |
| UX-02 | View pipeline status | User can see pending/running/completed/needs_review states. |  |
| UX-03 | Compare original and processed text | User can view OCR, corrected OCR, romanisation, translation, summary, and knowledge output. |  |
| UX-04 | Submit review | User can approve or correct output. |  |
| UX-05 | Demo flow | Supervisor/evaluator can follow the system from upload to final output. |  |

## Acceptance Criteria

| Area | Acceptance Criteria |
| --- | --- |
| OCR | Fine-tuned OCR targets at least 60% character accuracy on confirmed evaluation samples. |
| OCR benchmark | Fine-tuned OCR, Tesseract, and VLM OCR are compared where configured and references exist. |
| Agents | All agent stages produce stored outputs with status and confidence/uncertainty where possible. |
| Gold standard | Manual references are used for OCR and agent evaluation. |
| Failure handling | Low confidence, invalid JSON, failed API calls, and unreadable scans are handled without silent data loss. |
| Review | Human corrections are stored separately from original output. |
| Demo | Full local prototype demo can be completed on selected samples. |

## Pass/Fail Table

| Test ID | Result | Evidence / Notes | Reviewer |
| --- | --- | --- | --- |
| OCR-01 |  |  |  |
| OCR-02 |  |  |  |
| OCR-03 |  |  |  |
| OCR-04 |  |  |  |
| OCR-05 |  |  |  |
| AG-01 |  |  |  |
| AG-04 |  |  |  |
| AG-05 |  |  |  |
| TR-02 |  |  |  |
| KE-02 |  |  |  |
| ERR-06 |  |  |  |
