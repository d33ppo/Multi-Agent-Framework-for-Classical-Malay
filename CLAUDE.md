# Claude Code Instructions

## Project Summary

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

Final year project building a multi-agent pipeline to process Classical Malay documents from Majalah Qalam (1950–1969) written in Jawi script. Improves OCR through manually prepared ground-truth data and TrOCR fine-tuning, then passes extracted Jawi text through specialised LLM agents for romanisation, modern Malay translation, summarisation, and structured knowledge extraction.

Keep this file consistent with `docs/AGENTS.md`.

---

## Quick Technical Reference

### Stack

- **Frontend:** React
- **Backend:** Python / FastAPI
- **OCR Model:** TrOCR (fine-tuned on Jawi ground-truth data, PyTorch / Google Colab)
- **Agent Framework:** LangChain
- **LLM for agents:** OpenRouter (configurable model, default: Gemini 2.5 Flash)
- **Storage:** Supabase
- **Image processing:** OpenCV, PyMuPDF

### Pipeline

1. Scanned Jawi PDF or image input
2. Pre-processing — OpenCV, PyMuPDF
3. Fine-tuned TrOCR — extracts Jawi Unicode text
4. Romanisation agent — Method A (neural, jawi-nanot5-small) and Method B (reverse-lookup dictionary) run in parallel
5. Inter-Method Agreement Gate — ≥85% character agreement to pass; below that, flag for human review
6. Translation agent — Classical Malay to Modern Malay (OpenRouter LLM)
7. Summarisation agent and Knowledge extraction agent — run concurrently on the translation
8. Supabase storage
9. React frontend display

Runtime Fault Handler wraps every stage: on exception/timeout/empty output it logs, passes through the last valid output, and flags the stage incomplete.

### Dataset

- **Source:** Majalah Qalam scanned PDFs and existing romanised PDFs
- **Ground truth columns:** `file_name`, `rumi_sentence`, `jawi_autotransliterated`, `jawi_corrected`
- **Human-corrected ground truth:** issues 1950_11 and 1950_12 only. Issues 1950_07, 1950_09 and 1950_10 are uncorrected and are excluded unless corrected text arrives
- **Split:** by article into train, validation and test (not by issue)
- **Line count (dataset v1.0):** 4,180 lines from 77 annotated pages: 3,173 train, 366 validation, 641 test. Locked test articles are in `ocr_training/splits/test_articles.json`
- **Gold standard for romanisation evaluation:** existing Majalah Qalam romanised PDFs

### Evaluation Plan

| Stage | Gold Standard | Metric |
|---|---|---|
| OCR | `jawi_corrected` annotations | CER ≤10%, character accuracy ≥90% (RO2 target) |
| Romanisation | Majalah Qalam romanised PDFs | BLEU / chrF |
| Translation | Small-sample human rubric (student + supervisor) | Adequacy + fluency, 1–5 |
| Summarisation | ~20–30 human-written reference summaries | ROUGE |
| Knowledge extraction | Manually tagged gold entities | F1 |

Do not use LLM-as-a-judge anywhere in evaluation (rejected by panel and supervisor). Offline metrics live in `stage_metrics.py`.

### Three distinct validation mechanisms (never call these "the Validation Agent")

1. **Inter-Method Agreement Gate** — runtime, after romanisation (`agreement_gate.py`)
2. **Stage Evaluation Metrics** — offline, thesis reporting only (`stage_metrics.py`)
3. **Runtime Fault Handler** — runtime fault tolerance around every stage (`fault_handler.py`)

Total corpus line count is TBD — compute from annotation data (average lines/page × ~7,980 pages). Do not use the 240k/270k placeholders.

### Key Jawi Constraints

- Classical Malay Jawi has unique glyphs absent from standard Arabic: ca ﭺ, ga ڬ, nga ڠ, pa ڤ, va ۏ, nya ڽ
- Tesseract fails on these glyphs — hence the custom TrOCR fine-tuning approach
- Romanisation uses Method A (pretrained `mesolitica/jawi-nanot5-small-malaysian-cased`, used as-is, text-to-text) and Method B (reverse-lookup dictionary in `jawi_dictionary.py`, built with `python jawi_dictionary.py build`). No LLM romanisation and no fine-tuning of a romanisation model
- Majalah Qalam uses pre-1986 Jawi spelling (ف for p, ة for final t, omitted vowel letters); the dictionary data is modern Jawi, so Method B retries old-spelling variants and flags low-confidence matches as ambiguous
- Never overwrite original OCR output

---

## Current Progress

The Stack list above is the target stack. This section states what exists in the repo today.

- **Dataset preparation:** 2 issues (1950_11, 1950_12) have human-corrected ground truth. 1950_07, 1950_09 and 1950_10 are uncorrected and excluded
- **TrOCR:** training code is in `ocr_training/` (dataset builder, YOLO line detector, Colab notebooks; weights stay on Drive). TrOCR v1 test line CER is 3.78% (greedy) against Tesseract 64.31% on 641 held-out lines. Inference is wired as the `trocr` provider (`ocr_trocr.py`), the default once `OCR_MODEL_PATH` is set. It reuses the training preprocessing (YOLO boxes, XY-cut reading order, eval crop) and models stay local under `models/`. Tesseract and fixture providers remain for the baseline and tests. Page-level evaluation on the locked test articles is `scripts/eval_page_cer.py`. The low-confidence threshold is uncalibrated until task 3.6
- **Agent pipeline:** implemented with Method A/B romanisation, agreement gate, fault handler, concurrent summary + KE (`pipeline_runner.py`)
- **LangChain:** used for the three LLM stages. `llm_text_runtime.py` calls OpenRouter through `langchain-openai` `ChatOpenAI`, and `pipeline_runner.build_llm_chain` expresses Translation, then Summarisation and Knowledge Extraction, as one runnable chain with `RunnableParallel`. Romanisation and the agreement gate are plain Python. LangGraph is not used
- **Frontend:** a single static `frontend/index.html`, not a React app yet
- **Backend:** FastAPI app in `backend/main.py` with a health route, a config route, the streaming pipeline route, a document upload route, and run status and event routes. Uploads are processed by a background job runner (`job_runner.py`)
- **Supabase:** not implemented in code yet. The CLI runner writes results to local files under `output/`. Uploads and run records are stored locally under `storage/` behind a `Storage` interface (`document_storage.py`)

---

## Required Reading Order Before Coding

1. `docs/README.md`
2. `docs/PRD.md`
3. `docs/SAD.md`
4. `docs/API_SPEC.md`
5. `docs/DATABASE_SCHEMA.md`
6. `docs/QATD.md`

---

## Development Workflow

1. Read the required documents.
2. Inspect the affected code before editing.
3. Prepare a short implementation plan.
4. Make the smallest practical change.
5. Run relevant tests.
6. Summarise affected files and verification result.

---

## Planning-Before-Coding Rule

Before coding, identify:

- The feature or bug being changed
- Files likely to be affected
- API, database, and documentation impact
- Tests that should be added or updated

---

## Affected-Files Summary Rule

At the end of every change, provide a short summary listing:

- Files changed
- Main behaviour changed
- Tests run
- Any documentation updated

---

## Testing-Before-Finish Rule

- Run relevant unit or integration tests before finishing
- Mock external LLM and API calls unless live testing is explicitly required
- Include tests for OCR output, agent output, API endpoints, and failure cases
- If tests cannot be run, state why and what should be run later

---

## Documentation Update Rule

- Do not change architecture without updating `docs/SAD.md`
- Do not change database structure without updating `docs/DATABASE_SCHEMA.md`
- Do not change API routes without updating `docs/API_SPEC.md`
- Update `docs/QATD.md` when test approach or acceptance criteria changes
- Update `docs/DEPLOYMENT.md` and `docs/README.md` when setup commands change

---

## Architecture, Security, and Testing Constraints

- Keep code simple and suitable for FYP demonstration
- Use the documented stack only — do not introduce new infrastructure
- Do not hardcode API keys
- Do not commit `.env` files or secret values
- Do not upload manuscript data to external services unless explicitly configured
- Preserve original OCR text and never overwrite it silently
- AI outputs must include confidence or uncertainty indicators where possible
- Store agent run status and error messages in Supabase for every pipeline run
- Add tests for OCR output, agent output, API endpoints, and failure cases

---

## Forbidden Changes

- Do not add large production archive features unless explicitly requested
- Do not replace the FYP prototype with unnecessary complex infrastructure
- Do not change the agent workflow without updating `docs/AGENTS.md` and `docs/SAD.md`
- Do not remove human review from low-confidence or contradictory outputs
- Do not fine-tune a separate romanisation model — use the pretrained Method A model as-is plus the Method B dictionary