# AI Coding Agent Instructions

## Project Summary

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

This project is a Final Year Project prototype for selected Majalah Qalam Classical Malay Jawi documents. The system improves OCR using manually prepared ground-truth data, benchmarks OCR quality, and runs a multi-agent pipeline for romanisation, modern Malay translation, summarisation, and structured knowledge extraction.

## Required Reading Order Before Coding

1. `docs/README.md`
2. `docs/PRD.md`
3. `docs/SAD.md`
4. `docs/API_SPEC.md`
5. `docs/DATABASE_SCHEMA.md`
6. `docs/QATD.md`

## Coding Standards

- Keep code simple and suitable for FYP demonstration.
- Prefer readable Python and TypeScript/JavaScript over complex abstractions.
- Use clear function names and small modules.
- Validate external inputs at API boundaries.
- Store structured outputs as JSON where appropriate.
- Add concise comments only when the logic is not obvious.
- Preserve existing user data and do not silently overwrite outputs.

## Folder Structure

```text
backend/       # FastAPI services, OCR pipeline, agent orchestration
frontend/      # React user interface
notebooks/     # OCR training and experiments
data/          # Local development samples and generated outputs
tests/         # Unit and integration tests
docs/          # Documentation
```

## Allowed Tech Stack

- Frontend: React.
- Backend: Python and FastAPI.
- OCR training and experiments: Python, Google Colab, PyTorch, OpenCV, PyMuPDF.
- Agent framework: LangChain.
- LLM access: OpenRouter or explicitly configured LLM APIs.
- Romanisation Method A: HuggingFace `transformers` running `mesolitica/jawi-nanot5-small-malaysian-cased` locally.
- Offline evaluation metrics: `sacrebleu` (BLEU/chrF), `rouge-score` (ROUGE).
- Database and storage: Supabase.
- Deployment: local prototype first, optional Docker later.

## Current Implementation Status

The stack above is the target. This is what exists in the repo today. Keep this section consistent with "Current Progress" and "Dataset" in `CLAUDE.md`.

- Dataset: only issues 1950_11 and 1950_12 have human-corrected ground truth. Issues 1950_07, 1950_09 and 1950_10 are uncorrected and are excluded unless corrected text arrives.
- Dataset split: by article into train, validation and test, not by issue. Line count is TBD from the annotation export.
- TrOCR: no training or inference code in this repo yet.
- Agent pipeline: implemented in `pipeline_runner.py` with Method A/B romanisation, agreement gate, fault handler, and concurrent summarisation and knowledge extraction.
- LangChain: not used in code yet. Agents call OpenRouter directly through `llm_text_runtime.py`.
- Frontend: a single static `frontend/index.html`, not a React app yet.
- Backend: FastAPI app in `backend/main.py` with a health route and the streaming pipeline route.
- Supabase: not implemented in code yet. The CLI runner writes results to local files under `output/`.

## Forbidden Changes

- Do not change architecture without updating `docs/SAD.md`.
- Do not change database structure without updating `docs/DATABASE_SCHEMA.md`.
- Do not change API routes without updating `docs/API_SPEC.md`.
- Do not hardcode API keys.
- Do not upload manuscript data to external services unless explicitly configured.
- Do not silently overwrite original OCR text.
- Do not add production-scale archive features outside FYP scope unless requested.
- Do not replace simple prototype code with unnecessary framework complexity.

## Testing Rules

- Add tests for OCR output, agent output, API endpoints, and failure cases.
- Test invalid file upload, unreadable image, low OCR confidence, failed OCR extraction, invalid JSON, and failed LLM/API call.
- Mock external LLM/API calls in automated tests unless live testing is explicitly required.
- Run relevant tests before finishing a change.

## Security Rules

- Store API keys in `.env` or secret manager only.
- Never commit `.env` or secret values.
- Validate upload file type and size.
- Do not send manuscript data to external LLM/VLM services unless the configuration explicitly enables it.
- Log errors without exposing API keys or private manuscript content unnecessarily.

## Documentation Update Rules

- If architecture changes, update `docs/SAD.md`.
- If database tables, fields, or constraints change, update `docs/DATABASE_SCHEMA.md`.
- If API routes, request bodies, or response bodies change, update `docs/API_SPEC.md`.
- If test coverage or acceptance criteria change, update `docs/QATD.md`.
- If setup or deployment commands change, update `docs/DEPLOYMENT.md` and `docs/README.md`.

## Agent Workflow

```text
OCR text
  -> SubAgent 1 Romanisation: Method A (jawi-nanot5-small, neural) || Method B (reverse-lookup dictionary)
  -> Inter-Method Agreement Gate (>=85% character agreement, else flag for human review)
  -> SubAgent 2 Translation (OpenRouter LLM, default google/gemini-2.5-flash)
  -> SubAgent 3 Summarisation || SubAgent 4 Knowledge Extraction (concurrent, both consume the translation)
Runtime Fault Handler wraps every stage.
```

- Keep the three validation mechanisms separate and named: Inter-Method Agreement Gate (runtime), Stage Evaluation Metrics (offline), Runtime Fault Handler (runtime fault tolerance).
- Do not use LLM-as-a-judge for scoring or evaluation.
- Do not fine-tune a romanisation model; Method A is a pretrained model used as-is.
- Changing this workflow requires updating `docs/SAD.md` and this file.

## Strict Project Rules

- Preserve original OCR text and never overwrite it silently.
- AI outputs must include confidence/uncertainty when possible.
- Store agent run status and error messages.
- Keep the project aligned with selected Majalah Qalam samples and FYP prototype scope.
