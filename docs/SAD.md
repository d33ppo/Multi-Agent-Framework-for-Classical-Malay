# System Architecture Document

## System Overview

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

The system is a local FYP prototype that accepts selected scanned Majalah Qalam Jawi image/PDF documents, preprocesses them, runs fine-tuned OCR, evaluates OCR quality, and then coordinates specialised agents for romanisation, modern Malay translation, summarisation, and structured knowledge extraction. Outputs and review records are stored in Supabase.

## System Architecture

```mermaid
flowchart TB
    U[User / Reviewer] --> FE[React Frontend]
    FE --> API[FastAPI Backend]
    API --> PRE[Preprocessing Service]
    API --> OCR[Fine-tuned OCR Service]
    API --> BENCH[OCR Benchmark Service]
    API --> ORCH[LangChain Agent Orchestrator]
    ORCH --> A1[Romanisation Agent: Method A neural + Method B dictionary]
    A1 --> GATE[Inter-Method Agreement Gate]
    ORCH --> A2[Translation Agent]
    ORCH --> A3[Summarisation Agent]
    ORCH --> A4[Knowledge Extraction Agent]
    ORCH --> LLM[OpenRouter / Configured LLM APIs]
    API --> DB[(Supabase Database)]
    API --> ST[(Supabase Storage)]
    API --> LOG[System Logs]
    DB --> FE
```

## Module Breakdown

| Module | Responsibility |
| --- | --- |
| Document Upload Module | Accept scanned image/PDF, validate type and size, store uploaded file metadata. |
| Preprocessing Module | Convert PDF pages to images, clean image where appropriate, detect unreadable input. |
| OCR Module | Run fine-tuned Jawi OCR and store original OCR text, confidence, accuracy, and CER. |
| OCR Benchmarking Module | Compare fine-tuned OCR with Tesseract and VLM-based OCR on gold-standard samples. |
| Multi-Agent Processing Module | Coordinate romanisation, translation, summarisation, and extraction. |
| Output Module | Display structured output and preserve every stage result. |
| Review Module | Store human validation, corrections, and review status. |
| Storage Module | Store uploaded documents, outputs, references, reviews, and logs in Supabase. |

## Multi-Agent Pipeline

```mermaid
flowchart LR
    A[OCR Jawi Text + Metadata] --> B{OCR Confidence OK?}
    B -- No --> C[Flag for Human Review]
    B -- Yes --> RA
    C --> RA
    subgraph S1[SubAgent 1: Romanisation]
        RA[Method A: jawi-nanot5-small neural]
        RB[Method B: reverse-lookup dictionary]
    end
    A --> RB
    RA --> GATE{Inter-Method Agreement Gate: at least 85%}
    RB --> GATE
    GATE -- Below 85% --> R[Flag for Human Review]
    GATE --> F[SubAgent 2: Translation]
    R --> F
    F --> G[SubAgent 3: Summarisation]
    F --> H[SubAgent 4: Knowledge Extraction]
    G --> I[Structured Result]
    H --> I
    I --> J[Human Validation Review]
```

Method A and Method B run in parallel on the same OCR text. Translation must finish before Summarisation and Knowledge Extraction start, because both consume the modern Malay translation. Summarisation and Knowledge Extraction do not depend on each other and run concurrently. Every stage is wrapped by the Runtime Fault Handler.

## Agent Roles

| Agent | Role |
| --- | --- |
| Romanisation Agent (SubAgent 1) | Convert OCR Jawi text into romanised Classical Malay (Rumi) using two independent methods (A: neural, B: dictionary) and check their agreement. |
| Translation Agent | Translate Classical Malay/romanised text into modern Malay. |
| Summarisation Agent | Summarise the modern Malay translation. |
| Knowledge Extraction Agent | Extract structured entities, dates, places, topics, and relationships. |

## Agent Input/Output Table

| Agent | Input | Output | Stored In |
| --- | --- | --- | --- |
| Romanisation Agent | Original OCR Jawi text | Method A text (primary), Method B text, ambiguous/unknown word log, agreement score | `romanisation_outputs` |
| Translation Agent | Romanised text | Modern Malay translation, notes, confidence/uncertainty | `translation_outputs` |
| Summarisation Agent | Modern Malay translation | Summary, key points, confidence/uncertainty | `summaries` |
| Knowledge Extraction Agent | Translation, summary, optional source text | JSON with entities, dates, places, topics, relationships, confidence | `extracted_knowledge` |

## Model/Tool Used By Each Agent

| Component | Model/Tool | Notes |
| --- | --- | --- |
| OCR preprocessing | OpenCV, PyMuPDF | PDF/image handling and page preprocessing. |
| Fine-tuned OCR | PyTorch OCR model trained in Google Colab | Target CER at most 10% and character accuracy at least 90%. |
| Romanisation Method A | `mesolitica/jawi-nanot5-small-malaysian-cased` via `transformers`, run locally | Pretrained model used as-is (no fine-tuning). Reported chrF2++ 98.01 Jawi to Rumi on modern Jawi. |
| Romanisation Method B | Reverse-lookup dictionary built from `mesolitica/jawi-ms-en` (`wikipedia-1word.json` + `rumi-set.json`/`jawi-set.json` train split) | Rule-based. Retries pre-1986 spelling variants, splits joined prefixes, and flags ambiguous/unknown words. |
| OCR baseline | Tesseract | Used as benchmark baseline. |
| VLM OCR baseline | Configured VLM/API | Used where sample privacy and API settings permit. |
| Agent orchestration | LangChain | Coordinates steps, validation, retries, and output parsing. |
| Agent LLM | OpenRouter or configured LLM APIs | Must be configured via environment variables. |
| Storage | Supabase | Database and file storage. |

| Agent | Model/Tool | Output Control |
| --- | --- | --- |
| Romanisation Agent | Method A neural model + Method B dictionary (no LLM) | Must return both readings, the agreement score, and the ambiguous/unknown word log. |
| Translation Agent | LangChain prompt chain using configured LLM provider | Must translate into modern Malay and avoid unsupported additions. |
| Summarisation Agent | LangChain prompt chain using configured LLM provider | Must summarise from the translation, not invent new facts. |
| Knowledge Extraction Agent | LangChain structured-output chain using configured LLM provider | Must return JSON with entities, dates, places, topics, relationships, and confidence. |

## Coordination Logic

1. Backend receives uploaded document and creates a `manuscript_files` record.
2. Preprocessing prepares the document for OCR.
3. OCR module generates original OCR text and stores it in `ocr_results`.
4. If a gold-standard reference exists, benchmark metrics are calculated.
5. Agent orchestrator creates one `agent_runs` record per agent stage.
6. Agents run in this order:
   - Romanisation Agent: Method A and Method B in parallel.
   - Inter-Method Agreement Gate: compares A and B; below 85% flags the item for human review.
   - Translation Agent: consumes Method A output (or the fallback, see below).
   - Summarisation Agent and Knowledge Extraction Agent concurrently, both on the translation.
7. The Runtime Fault Handler wraps every stage: on exception, timeout, or empty output it logs the incident, passes the last valid output through, and marks the stage incomplete.
8. Flagged or incomplete items are shown to the user and require human review before being used as final evidence.

The same stage order is implemented once in `pipeline_runner.iter_agent_pipeline` and reused by the CLI runner and the FastAPI streaming endpoint.

## Validation Mechanisms

The project uses three separate mechanisms. They must not be conflated or described as a single "Validation Agent".

| Mechanism | When | Purpose | Implementation |
| --- | --- | --- | --- |
| Inter-Method Agreement Gate | Runtime, after SubAgent 1 | Character-level agreement between Method A and Method B, computed as 1 - edit distance / longer length after lowercasing and removing punctuation. At least 85% passes; lower flags the item for human review. Translation still runs so the reviewer sees full output. | `agreement_gate.py` |
| Stage Evaluation Metrics | Offline, for thesis reporting only | Romanisation: BLEU and chrF against Majalah Qalam romanised PDFs. Translation: human adequacy and fluency rubric (1-5). Summarisation: ROUGE against human reference summaries. Knowledge extraction: F1 against manually tagged gold entities. No LLM-as-a-judge. | `stage_metrics.py` |
| Runtime Fault Handler | Runtime, around every stage | Catches exceptions, timeouts, and empty output; logs; passes through the last valid output; marks the stage incomplete. Not a quality score. | `fault_handler.py` |

Interpretation limit: Method A and Method B are both derived from the same modern Jawi data, so agreement measures consistency between the two readings, not correctness against the manuscript. Majalah Qalam uses pre-1986 spelling (for example fa for p, and omitted vowel letters), which lowers both dictionary coverage and agreement.
9. Final output is displayed in the frontend and stored in Supabase.

## OCR Error Propagation Explanation

OCR errors are the main upstream risk. A wrong Jawi character can change the romanised word. A wrong romanised word can change the translation. A wrong translation can create an inaccurate summary. A wrong summary or translation can cause false extracted entities, topics, dates, or relationships.

For this reason, the system:

- Stores original OCR text separately.
- Tracks confidence and CER.
- Marks low-confidence output for review.
- Allows human validation before using results as final evidence.

## Failure Handling

| Failure Case | System Response |
| --- | --- |
| OCR confidence is low | Store OCR output, mark status as `needs_review`, continue only if user allows, and show warning in frontend. |
| Manuscript image is unreadable | Stop OCR run, store failure in `system_logs`, return `422 UNPROCESSABLE_ENTITY`, and request clearer sample or manual transcription. |
| Any stage raises, times out, or returns empty output | Runtime Fault Handler logs the incident, passes the last valid output through (Method A fails: Method B text; both fail: original OCR text; Translation fails: romanised text), and marks the stage incomplete. |
| Methods A and B disagree (below 85%) | Inter-Method Agreement Gate flags the item for human review; the pipeline continues. |
| LLM/API response is incomplete | Retry with stricter prompt and shorter context; if still incomplete, store partial output with status `failed` or `needs_review`. |
| Agent output contradicts previous agent output | Run consistency check, flag contradiction, store both outputs, and require human review before final approval. |
| JSON output is invalid | Retry JSON repair once or twice; if still invalid, store raw output and mark agent run as `failed`. |
| External API is unavailable | Use configured fallback provider if allowed; otherwise mark run as `failed_external_api` and allow retry later. |

## Retry Strategy

| Step | Retry Rule |
| --- | --- |
| OCR preprocessing | Retry once with safer preprocessing settings. |
| OCR model inference | Retry once if runtime fails, not if input is unreadable. |
| LLM agent call | Retry up to 2 times with stricter output instruction. |
| JSON parsing | Retry up to 2 times with schema repair prompt. |
| Supabase write | Retry up to 3 times for temporary network errors. |

Retries must be logged in `system_logs` and linked to `agent_runs` where relevant.

## Fallback Strategy

| Failure | Fallback |
| --- | --- |
| Fine-tuned OCR unavailable | Use Tesseract or VLM baseline only if configured, and mark output as baseline. |
| Low OCR confidence | Send result to human review before trusting downstream output. |
| LLM provider unavailable | Use fallback provider configured in environment variables, or pause the run. |
| Knowledge extraction JSON invalid | Store raw text and require reviewer correction. |
| Gold-standard reference missing | Run pipeline but mark evaluation metrics as unavailable. |

## Human Review Workflow

1. Reviewer opens a processed document.
2. Reviewer compares original scan, OCR text, romanisation, translation, summary, and extracted knowledge.
3. Reviewer submits corrections or approval.
4. Review is stored in `validation_reviews`.
5. If approved, output can be used for final prototype evaluation.

## Frontend/Backend/Database Interaction

```mermaid
sequenceDiagram
    participant User
    participant FE as React Frontend
    participant API as FastAPI Backend
    participant OCR as OCR Service
    participant Agent as LangChain Orchestrator
    participant DB as Supabase

    User->>FE: Upload Jawi image/PDF
    FE->>API: POST /documents
    API->>DB: Store file metadata
    API->>OCR: Preprocess and run OCR
    OCR-->>API: OCR text + confidence
    API->>DB: Store ocr_results
    FE->>API: POST /pipeline/run
    API->>Agent: Run agents in order
    Agent-->>API: Structured outputs
    API->>DB: Store agent outputs
    FE->>API: GET /pipeline/{id}/status
    API-->>FE: Status + output links
    User->>FE: Submit review
    FE->>API: POST /reviews
    API->>DB: Store validation review
```

## Panel Monitoring Coverage

This SAD directly addresses the requested clearer agent details:

- Agent roles and input/output are defined.
- Model/tool choices are stated.
- Coordination logic is described.
- OCR error propagation is explained.
- Failure handling, retry, fallback, and human review workflows are included.
- Agent failure cases such as invalid JSON, contradiction, and API outage are explicitly handled.
