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
7. The Runtime Fault Handler wraps every stage: on exception, timeout, or empty output it logs the incident, passes the last valid output through, and marks the stage incomplete. The three LLM stages (Translation, Summarisation, Knowledge Extraction) are retried up to 2 times before that happens. See "Runtime Fault Handler Behaviour" below.
8. Flagged or incomplete items are shown to the user and require human review before being used as final evidence.

The same stage order is implemented once in `pipeline_runner.iter_agent_pipeline` and reused by the CLI runner and the FastAPI streaming endpoint.

## Validation Mechanisms

The project uses three separate mechanisms. They must not be conflated or described as a single "Validation Agent".

| Mechanism | When | Purpose | Implementation |
| --- | --- | --- | --- |
| Inter-Method Agreement Gate | Runtime, after SubAgent 1 | Character-level agreement between Method A and Method B, computed as 1 - edit distance / longer length after lowercasing and removing punctuation. At least 85% passes; lower flags the item for human review. Translation still runs so the reviewer sees full output. | `agreement_gate.py` |
| Stage Evaluation Metrics | Offline, for thesis reporting only | Romanisation: BLEU and chrF against Majalah Qalam romanised PDFs. Translation: human adequacy and fluency rubric (1-5). Summarisation: ROUGE against human reference summaries. Knowledge extraction: F1 against manually tagged gold entities. No LLM-as-a-judge. | `stage_metrics.py` |
| Runtime Fault Handler | Runtime, around every stage | Catches exceptions, timeouts, and empty output; retries LLM stages on exception or timeout; logs; passes through the last valid output; marks the stage incomplete. Not a quality score. | `fault_handler.py` |

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
| Any stage raises, times out, or returns empty output | Runtime Fault Handler logs the incident, passes the last valid output through (Method A fails: Method B text; both fail: original OCR text; Translation fails: romanised text), and marks the stage incomplete. LLM stages are retried first when the failure is an exception or a timeout. |
| Methods A and B disagree (below 85%) | Inter-Method Agreement Gate flags the item for human review; the pipeline continues. |
| LLM/API response is incomplete | The call raises an error. The Runtime Fault Handler retries it up to 2 times with the same prompt, then passes the last valid output through and marks the stage incomplete. |
| Agent output contradicts previous agent output | Run consistency check, flag contradiction, store both outputs, and require human review before final approval. |
| JSON output is invalid | Knowledge Extraction validates its JSON against the schema. Invalid output gets one repair call. If it is still invalid, the stage is marked incomplete and its result is empty. Storing the raw invalid output is planned with Supabase. |
| External API is unavailable or not configured | Before a run: the API returns `503` when the key is missing or external calls are not allowed. It does not answer with mock text. During a run: the failed call is retried up to 2 times with backoff, then the stage is marked incomplete. A second provider and the `failed_external_api` status are planned, not implemented. |

## Runtime Fault Handler Behaviour

Implemented in `fault_handler.run_stage`. Every stage goes through it.

1. The stage function is called under a per-attempt timeout (180 seconds by default).
2. The result is classified as `completed`, `failed_exception`, `failed_timeout`, or `failed_empty_output`.
3. If the stage has retries (`max_retries`) and the failure is an exception or a timeout, the handler waits and calls the stage again. Empty output is never retried, because the stage ran and had nothing to return.
4. The wait uses exponential backoff with jitter: a random value between half and the full ceiling, where the ceiling is 1 second for the first retry and doubles each time, capped at 10 seconds.
5. Some errors are never retried because another attempt cannot fix them: a missing LLM configuration, and knowledge output that is still invalid after its repair call.
6. If the stage still fails, the last valid output is passed through, the stage is marked incomplete, and the error is recorded.

Every outcome records `attempts`, the number of times the stage function was called. It appears in `stage_status`, in the CLI result files, and in the `agent_done` events of the LLM stages.

| Stage | `max_retries` | Reason |
| --- | --- | --- |
| Romanisation Method A, Method B | 0 | They run locally. A failure such as a missing model or lookup file is not transient. |
| Translation, Summarisation, Knowledge Extraction | 2 | They call an external API, where rate limits and short outages are common. |

A timed-out call cannot be cancelled, so a retry after a timeout can overlap with the abandoned call. Retries are written to the Python log today. They will also be stored in `system_logs` and counted in `agent_runs.retry_count` once Supabase is connected.

## Knowledge Extraction Output Validation

Implemented in `knowledge_agent.py`. The model output is not trusted as is.

- The result must be a JSON object with exactly these keys: `people`, `places`, `dates`, `organizations`, `themes`, `key_claims`. Each must be a list of strings.
- Unknown keys are dropped. A single string is accepted as a one-item list. Blank items are removed.
- A missing key, or a value that is not a list of strings, makes the output invalid.
- Invalid output gets one repair call that sends the bad output and the problem back to the model and asks for valid JSON only. If the repaired output is also invalid, the agent raises an error and the Runtime Fault Handler marks the stage incomplete. This error is not retried, so the stage makes at most two LLM calls for one validation failure.
- The model may add an `evidence` list of `{entity, quote}` items. A quote is kept only if it appears word for word in the translation given to the agent (runs of whitespace are treated as one space, case must match). Other evidence items are dropped. This gives the reviewer a checkable source for an entity, and it is not a quality score.

The planned `extracted_knowledge` table uses different field names (`entities`, `topics`, `relationships`). The mapping will be decided when Supabase is connected.

## External Service Boundary

- Manuscript data leaves the machine only when `ALLOW_EXTERNAL_MANUSCRIPT_API=true`. The flag covers the three LLM agents (manuscript text to OpenRouter) and the VLM OCR baseline in `ocr_llm.py` (page images to OpenRouter). It is checked when the runtime is created and again before each call.
- Romanisation runs locally and sends nothing out. Method A downloads its model weights from HuggingFace on first use.
- The API never substitutes mock text for a live result. Without a key, or without the flag, `POST /api/pipeline/run` returns `503`. Mock output is served only when `APP_MODE=demo_mock` is set, and such runs report `mode: mock`.
- The client cannot choose an arbitrary model. `OPENROUTER_ALLOWED_MODELS` is the server-side allowlist, and `GET /api/config` exposes it so the frontend offers only those models.
- CORS origins come from `CORS_ALLOWED_ORIGINS`, and `jawi_text` is limited by `MAX_TEXT_CHARS`.

## Retry Strategy

| Step | Retry Rule |
| --- | --- |
| OCR preprocessing | Retry once with safer preprocessing settings. (Planned.) |
| OCR model inference | Retry once if runtime fails, not if input is unreadable. (Planned.) |
| LLM agent call | Implemented. Retry up to 2 times on an exception or a timeout, with exponential backoff and jitter. Same prompt each time. Not retried on empty output. |
| JSON parsing (Knowledge Extraction) | Implemented. One repair call that sends the invalid output back and asks for valid JSON only. No further retry. |
| Romanisation Method A and Method B | No retry. Both run locally. |
| Supabase write | Retry up to 3 times for temporary network errors. (Planned.) |

Retries must be logged in `system_logs` and linked to `agent_runs` where relevant. Until Supabase is connected they are written to the Python log.

## Fallback Strategy

| Failure | Fallback |
| --- | --- |
| Fine-tuned OCR unavailable | Use Tesseract or VLM baseline only if configured, and mark output as baseline. |
| Low OCR confidence | Send result to human review before trusting downstream output. |
| LLM provider unavailable | The stage is retried, then its last valid output is passed through and the stage is marked incomplete. A second provider is not implemented. |
| Knowledge extraction JSON invalid | One repair call, then an empty result with the stage marked incomplete so a reviewer can correct it. |
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
