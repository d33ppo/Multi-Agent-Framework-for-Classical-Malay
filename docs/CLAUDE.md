# Claude Code Instructions

## Project Summary

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

This project is a Final Year Project prototype for selected Majalah Qalam Classical Malay Jawi documents. It improves OCR through manually prepared ground-truth data and fine-tuning, then uses a multi-agent pipeline for OCR correction, romanisation, modern Malay translation, summarisation, and structured knowledge extraction.

Keep this file consistent with `docs/AGENTS.md`.

## Required Reading Order Before Coding

1. `docs/README.md`
2. `docs/PRD.md`
3. `docs/SAD.md`
4. `docs/API_SPEC.md`
5. `docs/DATABASE_SCHEMA.md`
6. `docs/QATD.md`

## Development Workflow

1. Read the required documents.
2. Inspect the affected code before editing.
3. Prepare a short implementation plan.
4. Make the smallest practical change.
5. Run relevant tests.
6. Summarise affected files and verification result.

## Planning-Before-Coding Rule

Before coding, identify:

- Feature or bug being changed.
- Files likely to be affected.
- API/database/documentation impact.
- Tests that should be added or updated.

## Affected-Files Summary Rule

At the end of a change, provide a short summary listing:

- Files changed.
- Main behaviour changed.
- Tests run.
- Any documentation updated.

## Testing-Before-Finish Rule

- Run relevant unit or integration tests before finishing.
- Mock external LLM/API calls unless live testing is explicitly required.
- Include tests for OCR output, agent output, API endpoints, and failure cases.
- If tests cannot be run, state why and what should be run later.

## Documentation Update Rule

- Do not change architecture without updating `docs/SAD.md`.
- Do not change database structure without updating `docs/DATABASE_SCHEMA.md`.
- Do not change API routes without updating `docs/API_SPEC.md`.
- Update `docs/QATD.md` when test approach or acceptance criteria changes.
- Update `docs/DEPLOYMENT.md` and `docs/README.md` when setup commands change.

## Architecture, Security, and Testing Constraints

- Keep code simple and suitable for FYP demonstration.
- Use the documented stack: React, Python/FastAPI, Google Colab/PyTorch/OpenCV/PyMuPDF, LangChain, OpenRouter or configured LLM APIs, and Supabase.
- Do not hardcode API keys.
- Do not commit `.env` files or secret values.
- Do not upload manuscript data to external services unless explicitly configured.
- Preserve original OCR text and never overwrite it silently.
- Keep corrected OCR text separate from original OCR text.
- AI outputs must include confidence/uncertainty when possible.
- Store agent run status and error messages.
- Add tests for OCR output, agent output, API endpoints, and failure cases.

## Forbidden Changes

- Do not add large production archive features unless explicitly requested.
- Do not replace the FYP prototype with unnecessary complex infrastructure.
- Do not change the agent workflow without updating the documentation.
- Do not remove human review from low-confidence or contradictory outputs.
