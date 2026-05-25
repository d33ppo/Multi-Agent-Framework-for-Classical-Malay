# Product Requirements Document

## Project Overview

**Project title:** Multi-Agent Framework for Classical Malay Understanding and Knowledge Extraction

This project builds a Final Year Project prototype for processing selected scanned Jawi documents from Majalah Qalam, especially scanned image/PDF samples from 1950-1969. The system improves Jawi OCR using manually prepared ground-truth data and fine-tuning, then sends the extracted Jawi text through a multi-agent pipeline for OCR correction, romanisation, modern Malay translation, summarisation, and structured knowledge extraction.

## Problem Statement

Generic OCR tools do not perform well on Classical Malay Jawi documents. The main causes are:

- Low or inconsistent scan quality.
- Historical typography variation.
- Arabic-based script complexity.
- Malay-specific Jawi letters.
- Layout noise and missing text regions.

If OCR is inaccurate, downstream agents may romanise wrongly, translate incorrectly, summarise misleading content, or extract false entities and topics. Therefore, OCR improvement and OCR quality measurement must happen before the multi-agent NLP pipeline is treated as reliable.

## Objectives

| ID | Objective |
| --- | --- |
| RO1 | Develop a multi-agent framework for Classical Malay Jawi documents that produces structured outputs. |
| RO2 | Develop and fine-tune a Jawi OCR model targeting at least 60% character accuracy, benchmarked against Tesseract and VLM-based OCR using accuracy and CER. |
| RO3 | Design specialised agents for romanisation, translation, summarisation, and knowledge extraction, evaluated using BLEU, ROUGE, human review, and structured output accuracy where suitable. |

## Target Users

- Researchers studying Classical Malay texts.
- Librarians and archivists involved in digital preservation.
- Students and academics requiring searchable Jawi materials.
- Supervisor and FYP evaluator.

## Scope

### In Scope

- Selected scanned image/PDF documents of Majalah Qalam written in Classical Malay Jawi.
- Custom OCR model using manually prepared Jawi ground-truth data.
- OCR benchmarking against existing systems.
- Multi-agent pipeline for OCR correction, romanisation, translation, summarisation, and knowledge extraction.
- FYP prototype with frontend, backend, agent pipeline, and storage.

### Out of Scope

- Handwritten manuscript OCR.
- Large-scale deployment across all Jawi archives.
- Translation beyond Classical Malay/Jawi to modern Malay.
- Full production-level digital archive system.

## Core Features

| Feature | Description |
| --- | --- |
| Document upload | Upload selected scanned Jawi image/PDF documents. |
| Document preprocessing | Validate file type, extract pages, and prepare images for OCR. |
| Fine-tuned OCR | Extract Jawi text using a custom trained OCR model. |
| OCR benchmarking | Compare custom OCR with Tesseract and VLM-based OCR. |
| OCR Correction Agent | Correct likely OCR errors while preserving the original OCR text. |
| Romanisation Agent | Convert corrected Jawi text into romanised Malay. |
| Translation Agent | Translate romanised/Classical Malay text into modern Malay. |
| Summarisation Agent | Produce concise summary of the document excerpt. |
| Knowledge Extraction Agent | Extract entities, dates, topics, places, and relationships. |
| Human validation | Allow reviewer correction and quality judgement. |
| Structured storage | Store files, OCR results, agent runs, outputs, reviews, and logs in Supabase. |

## Functional Requirements

| ID | Requirement |
| --- | --- |
| FR1 | The system shall allow users to upload selected scanned Jawi image/PDF documents. |
| FR2 | The system shall validate file type, file size, and readable page/image content. |
| FR3 | The system shall run preprocessing before OCR. |
| FR4 | The system shall run the fine-tuned OCR model and store the original OCR text. |
| FR5 | The system shall benchmark OCR output against gold-standard references using accuracy and CER. |
| FR6 | The system shall compare fine-tuned OCR with Tesseract and VLM-based OCR where samples are available. |
| FR7 | The system shall run OCR correction, romanisation, translation, summarisation, and knowledge extraction agents. |
| FR8 | The system shall store each agent input, output, status, confidence/uncertainty, and error message if any. |
| FR9 | The system shall allow a reviewer to submit validation comments and corrected outputs. |
| FR10 | The system shall display pipeline status and final structured output. |

## Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR1 | The prototype should be simple enough for FYP demonstration and maintenance. |
| NFR2 | Original OCR text must be preserved and never overwritten silently. |
| NFR3 | API keys must be stored in environment variables, not hardcoded. |
| NFR4 | Manuscript data must not be uploaded to external services unless explicitly configured. |
| NFR5 | Agent outputs should include confidence or uncertainty where possible. |
| NFR6 | The system should fail gracefully with clear error messages. |
| NFR7 | The backend should log OCR and agent failures for debugging. |
| NFR8 | Local prototype performance should be acceptable for 10-15 pilot samples and 30-50 final evaluation excerpts. |

## Success Metrics

| Area | Metric | Target / Proposal |
| --- | --- | --- |
| OCR | Character accuracy | At least 60% target for the fine-tuned OCR model. |
| OCR | Character Error Rate | Lower CER than baseline Tesseract on selected evaluation samples. |
| OCR comparison | Benchmark table | Fine-tuned OCR, Tesseract, and VLM-based OCR compared on the same samples where possible. |
| Romanisation | BLEU / human review | Compare against gold-standard romanisation or reviewer judgement. |
| Translation | BLEU / human review | Compare against modern Malay reference translation or reviewer judgement. |
| Summarisation | ROUGE / human review | Compare against reference summary or reviewer judgement. |
| Knowledge extraction | Structured output accuracy | Check extracted entities, dates, places, topics, and relationships against reference annotations. |
| Reliability | Failure handling | Low confidence, invalid JSON, incomplete LLM response, and unavailable API cases handled without data loss. |

## Gold-Standard Evaluation Overview

The gold standard is the manually verified reference used to evaluate OCR and agent outputs. It should include:

| Output Type | Gold-Standard Reference |
| --- | --- |
| OCR | Manually transcribed Jawi text from selected Majalah Qalam excerpts. |
| Romanisation | Human-reviewed romanised Malay text. |
| Translation | Human-reviewed modern Malay translation. |
| Summarisation | Human-written or human-approved summary. |
| Knowledge extraction | Human-annotated entities, dates, topics, places, and relationships. |

The proposed evaluation sample size is:

- **Early pilot:** 10-15 samples.
- **Final prototype evaluation:** 30-50 manuscript excerpts.
- **Final number:** To be confirmed with supervisor or library collaborator.

## Literature Review Direction

The literature review should not focus only on OCR. It should also cover:

- NLP agent architectures, including sequential pipelines, tool-using agents, retry logic, and validation loops.
- Malay language processing, including romanisation, translation, summarisation, and low-resource NLP.
- Classical Malay/Jawi text processing and the effect of OCR noise on downstream NLP.
- Evaluation methods for OCR and NLP agents, including CER, BLEU, ROUGE, human review, and structured extraction accuracy.

## Impact of OCR Accuracy Above 60%

The 60% character accuracy target is a realistic FYP milestone for improving poor baseline OCR on difficult Jawi scans. Accuracy above 60% does not mean the output is perfect, but it can reduce the number of severe errors passed into downstream agents. Better OCR should improve:

- Romanisation consistency.
- Translation faithfulness.
- Summary relevance.
- Entity and topic extraction accuracy.

If OCR accuracy remains low, agents may produce fluent but incorrect output. For this reason, OCR confidence, CER, human validation, and fallback handling are part of the system requirements.

## Assumptions

- The monitoring PDF named `Multi-Agent Framework 22001821.pdf` was referenced but not found in the current workspace during documentation generation; the provided request text is treated as the monitoring reference.
- The project will use selected Majalah Qalam samples, not a full archive.
- Manual ground-truth preparation is feasible for a limited FYP-scale sample.
- Supabase will be used for prototype storage.
- External LLM/API use will be explicitly configured through environment variables.
- Final evaluation sample count will be confirmed with the supervisor or library collaborator.

## Constraints

- OCR model training may require Google Colab/GPU instead of local laptop execution.
- Historical Jawi scan quality may limit OCR accuracy.
- Gold-standard annotation takes manual effort.
- External LLM APIs may have cost, rate limit, privacy, and availability constraints.
- The prototype must remain simple enough for FYP completion.

## Panel Monitoring Comments Addressed

| Panel Comment | How This PRD Addresses It |
| --- | --- |
| Literature review should focus more on NLP agent architectures and Malay language processing. | Scope and success metrics explicitly include specialised NLP agents and Malay/Jawi processing. |
| Define the gold-standard reference used to evaluate the agents. | The gold-standard evaluation overview defines OCR, romanisation, translation, summary, and knowledge extraction references. |
| State a realistic number of test samples. | The PRD proposes 10-15 pilot samples and 30-50 final evaluation excerpts, with final number to be confirmed. |
| Explain the impact of OCR accuracy above 60% and downstream effects. | The OCR accuracy section explains how OCR quality affects all downstream agents. |
| Provide clearer agent details including models, coordination, error handling, fallback, retry, and failures. | This PRD defines the requirement; SAD.md and AGENTS.md provide the detailed implementation rules. |
