# Project Tasks

## Overview

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

This task list breaks the FYP prototype into practical phases. The goal is a simple working prototype, not a production archive system.

## Phase 1: Documentation and Setup

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T1.1 | Confirm project scope with supervisor/library collaborator. | None | Confirmed scope notes. | In-scope and out-of-scope items agreed. | PRD.md |
| T1.2 | Confirm evaluation sample plan. | T1.1 | Pilot and final sample targets. | 10-15 pilot and 30-50 final excerpts accepted or adjusted. | PRD.md, QATD.md |
| T1.3 | Set up repo folders for backend, frontend, tests, and notebooks. | None | Project folder structure. | Folders exist and are documented. | README.md |
| T1.4 | Configure environment variable template. | T1.3 | `.env.example`. | No real API keys committed. | DEPLOYMENT.md |

## Phase 2: Document Upload Module

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T2.1 | Create upload API endpoint. | T1.3 | `POST /documents`. | PDF/image upload creates `manuscript_files` record. | API_SPEC.md |
| T2.2 | Add file validation. | T2.1 | Type and size validation. | Unsupported files rejected with clear error. | QATD.md |
| T2.3 | Store uploaded file in Supabase Storage. | T2.1 | Storage path saved. | File can be retrieved through backend metadata. | DATABASE_SCHEMA.md |

## Phase 3: OCR Pipeline

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T3.1 | Build preprocessing function for image/PDF input. | T2.1 | Preprocessed page images. | Clear and unreadable samples handled. | SAD.md |
| T3.2 | Integrate fine-tuned OCR inference. | T3.1 | OCR text and confidence. | `ocr_results` record created. | DATABASE_SCHEMA.md |
| T3.3 | Preserve original OCR text separately. | T3.2 | Stored `original_ocr_text`. | Original OCR is never overwritten silently. | AGENTS.md |
| T3.4 | Add low-confidence OCR handling. | T3.2 | `needs_review` status. | Low-confidence output is flagged. | SAD.md, QATD.md |

## Phase 4: OCR Benchmarking

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T4.1 | Add gold-standard reference storage. | T3.2 | `gold_standard_references` records. | Manual references can be saved. | DATABASE_SCHEMA.md |
| T4.2 | Calculate character accuracy and CER. | T4.1 | OCR metrics. | Metrics are returned by OCR result API. | API_SPEC.md |
| T4.3 | Add Tesseract baseline comparison. | T4.2 | Baseline metrics. | Tesseract result included in benchmark. | PRD.md |
| T4.4 | Add VLM OCR comparison where configured. | T4.2 | VLM baseline metrics. | VLM result included only when API use is explicitly configured. | PRD.md, DEPLOYMENT.md |

## Phase 5: Multi-Agent Pipeline

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T5.1 | Create LangChain orchestration skeleton. | T3.2 | Agent runner. | Agent steps can run in documented order. | SAD.md |
| T5.2 | Implement Romanisation Agent. | T5.1 | Romanised text. | Output stored in `romanisation_outputs`. | DATABASE_SCHEMA.md |
| T5.3 | Implement Translation Agent. | T5.2 | Modern Malay translation. | Output stored in `translation_outputs`. | DATABASE_SCHEMA.md |
| T5.4 | Implement Summarisation Agent. | T5.3 | Summary and key points. | Output stored in `summaries`. | DATABASE_SCHEMA.md |
| T5.5 | Implement Knowledge Extraction Agent. | T5.3 | Structured JSON knowledge. | Valid JSON stored in `extracted_knowledge`. | API_SPEC.md |
| T5.6 | Add retry, fallback, and invalid JSON handling. | T5.2-T5.5 | Robust agent execution. | Failure cases pass QA tests. | SAD.md, QATD.md |

## Phase 6: Frontend Screens

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T6.1 | Build upload screen. | T2.1 | React upload UI. | User can upload sample. | README.md |
| T6.2 | Build OCR result screen. | T3.2 | OCR display. | OCR text, confidence, accuracy, and CER visible. | PRD.md |
| T6.3 | Build pipeline output screen. | T5.5 | Multi-output display. | OCR, romanisation, translation, summary, and knowledge visible. | SAD.md |
| T6.4 | Build validation review screen. | T6.3 | Review form. | Reviewer can submit approval or comments. | QATD.md |

## Phase 7: Supabase Integration

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T7.1 | Create Supabase tables. | T1.4 | Database schema. | Tables match `DATABASE_SCHEMA.md`. | DATABASE_SCHEMA.md |
| T7.2 | Connect backend to Supabase. | T7.1 | Supabase client. | Backend can create and read records. | DEPLOYMENT.md |
| T7.3 | Add system logging. | T7.2 | `system_logs` records. | Failures and retries are logged. | SAD.md |

## Phase 8: Gold-Standard Evaluation

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T8.1 | Prepare pilot gold-standard references. | T4.1 | 10-15 annotated samples. | Pilot references reviewed. | PRD.md, QATD.md |
| T8.2 | Prepare final evaluation references. | T8.1 | 30-50 excerpts if approved. | Final count confirmed with supervisor/collaborator. | QATD.md |
| T8.3 | Evaluate agent outputs. | T5.6, T8.2 | BLEU/ROUGE/human review/structured accuracy results. | Results ready for report. | PRD.md |

## Phase 9: Testing and Debugging

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T9.1 | Add OCR tests. | T3.2 | OCR test cases. | Clear, blurry, low-confidence, and failed OCR cases covered. | QATD.md |
| T9.2 | Add API tests. | T2.1, T5.6 | Endpoint tests. | Main routes pass. | API_SPEC.md |
| T9.3 | Add agent failure tests. | T5.7 | Mocked failure tests. | Invalid JSON and failed LLM/API calls covered. | QATD.md |
| T9.4 | Fix demo-blocking bugs. | T9.1-T9.3 | Stable prototype. | Demo flow runs end to end. | DEPLOYMENT.md |

## Phase 10: Demo Preparation

| Task ID | Task Description | Dependency | Expected Output | Completion Criteria | Related Document |
| --- | --- | --- | --- | --- | --- |
| T10.1 | Select demo samples. | T8.1 | Demo sample set. | Samples are clear enough for demonstration. | DEPLOYMENT.md |
| T10.2 | Prepare demo script. | T9.4 | Step-by-step demo checklist. | Upload to review flow can be shown. | README.md |
| T10.3 | Export benchmark/evaluation results. | T8.3 | Tables/figures for report. | Results are ready for supervisor/evaluator. | PRD.md |
| T10.4 | Final documentation consistency check. | All phases | Updated docs. | Docs match implemented prototype. | All docs |
