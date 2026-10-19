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
| AG-04 | Invalid agent JSON output | Mock invalid JSON from Knowledge Extraction Agent. | System makes one JSON repair call. If the output is still invalid, the stage is marked incomplete and no further LLM call is made. |  |
| AG-05 | Failed LLM/API call | Mock an LLM stage that raises. | Stage is retried up to 2 times with backoff. It completes if a later attempt succeeds; otherwise the last valid output is passed through and the stage is marked incomplete. `attempts` is recorded. |  |
| AG-07 | Retry scope | Mock empty output from an LLM stage, and a failing romanisation method. | Neither is retried: each is called once. |  |
| AG-06 | Agent contradiction | Mock translation contradicting source meaning. | System flags contradiction for human review. |  |

## Romanisation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| ROM-01 | Romanise valid OCR Jawi | Run Romanisation Agent. | Method A and Method B text are both produced; Method A is passed to Translation. |  |
| ROM-02 | Ambiguous Jawi word | Include a Jawi spelling with several common Rumi readings. | Most frequent candidate is used and the word is logged as `ambiguous` with candidates. |  |
| ROM-03 | Unknown Jawi word | Include a word missing from the dictionary. | Word stays visible as `⟦jawi⟧`, is logged as `unknown`, and nothing crashes. |  |
| ROM-04 | Old (pre-1986) spelling | Include ف-for-p or final ة spellings. | Dictionary resolves the modern variant and records the rule used. |  |
| ROM-05 | Methods agree | Mock A and B with matching output. | Agreement Gate passes (at least 85%). |  |
| ROM-06 | Methods disagree | Mock A and B with different output. | Gate flags the item for human review; pipeline continues. |  |
| ROM-07 | Method A fails | Mock neural model error. | Method B text is passed through; gate is skipped; stage marked incomplete. |  |
| ROM-08 | Compare to gold standard | Run `stage_metrics.py romanisation` against Qalam romanised PDFs. | BLEU and chrF are recorded. |  |

## Translation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| TR-01 | Translate valid romanised text | Run Translation Agent. | Modern Malay translation is produced. |  |
| TR-02 | Hallucinated translation | Mock translation with unsupported added meaning. | Human review or consistency check flags output. |  |
| TR-03 | Human rubric evaluation | Student and supervisor rate a sample for adequacy and fluency (1-5); run `stage_metrics.py rubric`. | Mean and standard deviation per criterion are recorded. No LLM-as-a-judge. |  |

## Summarisation Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| SUM-01 | Summarise valid translation | Run Summarisation Agent. | Summary and key points are stored. |  |
| SUM-02 | Summary stage fails | Mock summary exception or empty output. | Runtime Fault Handler marks the stage incomplete; knowledge extraction still completes. |  |
| SUM-03 | ROUGE evaluation | Run `stage_metrics.py summary` against 20-30 human reference summaries. | ROUGE-1/2/L are recorded. |  |

## Knowledge Extraction Test Cases

| ID | Test Case | Steps | Expected Result | Pass/Fail |
| --- | --- | --- | --- | --- |
| KE-01 | Extract entities and topics | Run Knowledge Extraction Agent. | JSON has exactly `people`, `places`, `dates`, `organizations`, `themes`, `key_claims` as lists of strings, plus an `evidence` list. Unknown keys are dropped. |  |
| KE-02 | Inconsistent extracted entity | Mock entity not supported by translation/source. | System flags entity for review. |  |
| KE-03 | Invalid JSON extraction | Return malformed JSON, or JSON that breaks the schema. | One repair call. If the repaired output is valid it is used; otherwise the stage is marked incomplete. |  |
| KE-04 | Compare to annotation | Run `stage_metrics.py knowledge` against manually tagged gold entities. | Per-field and micro precision/recall/F1 are recorded. |  |
| KE-05 | Concurrency | Run summary and knowledge extraction on the same translation. | Both start after translation and run concurrently. |  |
| KE-06 | Evidence quote check | Mock evidence items whose quotes are and are not in the input text. | Quotes found word for word are kept. Paraphrased, missing, or malformed evidence is dropped without failing the stage. |  |

## Failure/Error Test Cases

| ID | Failure | Expected Handling | Pass/Fail |
| --- | --- | --- | --- |
| ERR-01 | Unsupported file type | Return `400` and do not store invalid file. |  |
| ERR-02 | Unreadable manuscript image | Return `422`, log error, and request better sample/manual review. |  |
| ERR-03 | External API unavailable | Retry up to 2 times with backoff; otherwise pass the last valid output through and mark the stage incomplete. |  |
| ERR-04 | Supabase write failure | Retry and log failure. |  |
| ERR-05 | Missing gold standard | Run pipeline but mark metrics unavailable. |  |
| ERR-07 | Stage timeout or empty output | Runtime Fault Handler logs, passes through last valid output, flags stage incomplete. |  |
| ERR-06 | Human reviewer correction | Store correction in `validation_reviews` and preserve original output. |  |
| ERR-08 | LLM key missing, or external API not allowed | `POST /api/pipeline/run` returns `503` and no mock text. With `APP_MODE=demo_mock` it runs and reports `mode: mock`. |  |
| ERR-09 | Model not on the server allowlist | Return `400`. No LLM call is made. |  |
| ERR-10 | Input text over `MAX_TEXT_CHARS` | Return `413`. |  |
| ERR-11 | VLM OCR baseline without `ALLOW_EXTERNAL_MANUSCRIPT_API=true` | `ocr_llm.py` refuses to run and the image is not read or sent. |  |

## Usability Test Cases

| ID | Test Case | Expected Result | Pass/Fail |
| --- | --- | --- | --- |
| UX-01 | Upload document from frontend | User can upload without technical steps. |  |
| UX-02 | View pipeline status | User can see pending/running/completed/needs_review states. |  |
| UX-03 | Compare original and processed text | User can view OCR, romanisation, translation, summary, and knowledge output. |  |
| UX-04 | Submit review | User can approve or correct output. |  |
| UX-05 | Demo flow | Supervisor/evaluator can follow the system from upload to final output. |  |

## Acceptance Criteria

| Area | Acceptance Criteria |
| --- | --- |
| OCR | Fine-tuned OCR reaches CER of at most 10% and character accuracy of at least 90% on the held-out test split (split by article). |
| Romanisation gate | Method A/B agreement is reported per item; items below 85% are flagged for human review. |
| Evaluation method | Stage metrics use BLEU/chrF, human rubric, ROUGE, and F1 only. LLM-as-a-judge is not used. |
| OCR benchmark | Fine-tuned OCR, Tesseract, and VLM OCR are compared where configured and references exist. |
| Agents | All agent stages produce stored outputs with status and confidence/uncertainty where possible. |
| Gold standard | Manual references are used for OCR and agent evaluation. |
| Failure handling | Low confidence, invalid JSON, failed API calls, and unreadable scans are handled without silent data loss. |
| Review | Human review comments are stored in `validation_reviews` separately from agent output. |
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
