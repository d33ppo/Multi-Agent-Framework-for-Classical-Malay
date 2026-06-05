# Claude Code Instructions

## Project Summary

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

Final year project building a multi-agent pipeline to process Classical Malay documents from Majalah Qalam (1950–1969) written in Jawi script. Improves OCR through manually prepared ground-truth data and TrOCR fine-tuning, then passes extracted Jawi text through specialised LLM agents for OCR correction, romanisation, modern Malay translation, summarisation, and structured knowledge extraction.

Keep this file consistent with `docs/AGENTS.md`.

---

## Quick Technical Reference

### Stack

- **Frontend:** React
- **Backend:** Python / FastAPI
- **OCR Model:** TrOCR (fine-tuned on Jawi ground-truth data, PyTorch / Google Colab)
- **Agent Framework:** LangChain
- **LLM for agents:** Claude Sonnet 4.6 via OpenRouter
- **Storage:** Supabase
- **Image processing:** OpenCV, PyMuPDF

### Pipeline

1. Scanned Jawi PDF or image input
2. Pre-processing — OpenCV, PyMuPDF
3. Fine-tuned TrOCR — extracts Jawi Unicode text
4. OCR correction agent — LangChain / LLM
5. Romanisation agent — Jawi to Classical Malay Rumi
6. Translation agent — Classical Malay to Modern Malay
7. Summarisation agent
8. Knowledge extraction agent
9. Supabase storage
10. React frontend display

### Dataset

- **Source:** Majalah Qalam scanned PDFs and existing romanised PDFs
- **Ground truth columns:** `file_name`, `rumi_sentence`, `jawi_autotransliterated`, `jawi_corrected`
- **Target:** approximately 5 annotated issues, approximately 6,000 line-level image-text pairs
- **Split by issue:** 3 issues for training (~3,600 lines), 1 for validation (~1,200 lines), 1 for testing (~1,200 lines)
- **Gold standard for romanisation evaluation:** existing Majalah Qalam romanised PDFs

### Evaluation Plan

| Stage | Gold Standard | Metric |
|---|---|---|
| OCR | `jawi_corrected` annotations | CER, character accuracy |
| Romanisation | Majalah Qalam romanised PDFs | BLEU / chrF |
| Translation | LLM-as-a-judge | Semantic score |
| Summarisation | LLM-as-a-judge | Content coverage |
| Knowledge extraction | LLM-as-a-judge | F1 entity/concept |

### Key Jawi Constraints

- Classical Malay Jawi has unique glyphs absent from standard Arabic: ca ﭺ, ga ڬ, nga ڠ, pa ڤ, va ۏ, nya ڽ
- Tesseract fails on these glyphs — hence the custom TrOCR fine-tuning approach
- No separate romanisation model is to be built — romanisation agent uses LLM on Unicode text only, which is text-to-text, not vision
- Never overwrite original OCR output — keep corrected text in a separate field

---

## Current Progress

- **Dataset preparation:** 2 issues annotated, targeting 5
- **TrOCR fine-tuning:** in progress
- **Agent pipeline:** partially implemented, see `romanisation_agent.py`
- **Frontend:** React app scaffolded
- **Backend:** FastAPI scaffolded
- **Supabase:** connected

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
- Keep corrected OCR text separate from original OCR text in all storage
- AI outputs must include confidence or uncertainty indicators where possible
- Store agent run status and error messages in Supabase for every pipeline run
- Add tests for OCR output, agent output, API endpoints, and failure cases

---

## Forbidden Changes

- Do not add large production archive features unless explicitly requested
- Do not replace the FYP prototype with unnecessary complex infrastructure
- Do not change the agent workflow without updating `docs/AGENTS.md` and `docs/SAD.md`
- Do not remove human review from low-confidence or contradictory outputs
- Do not fine-tune a separate romanisation model — romanisation is handled by the LLM agent only
- Do not merge or overwrite the original Jawi OCR output with corrected output