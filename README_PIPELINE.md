# Post-OCR Multi-Agent Pipeline

Backend-first pipeline for the current FYP phase where the OCR model is treated
as an input dependency and the downstream text-understanding flow is developed
first.

## Current scope

This pipeline starts from OCR text and runs four subagents:

1. `Romanisation`
2. `Classical Malay -> modern Malay translation`
3. `Summarisation`
4. `Knowledge extraction`

The pipeline runs directly from OCR text without a correction step.

## Modules

- `ocr_input_adapter.py`
- `romanisation_agent.py`
- `translation_agent.py`
- `summary_agent.py`
- `knowledge_agent.py`
- `pipeline_runner.py`
- `result_exporter.py`
- `llm_text_runtime.py`
- `pipeline_types.py`

## Supported inputs

- Plain OCR text fixtures (`.txt`)
- Saved OCR output text files from this repo
- Canonical JSON OCR input records
- Local image files via Tesseract

Notes:
- Heavy Jawi-specialized OCR models are treated as off-device resources.
- Direct scanned PDF OCR is not enabled in the lightweight local pipeline.

## Fixture-driven development inputs

Sample OCR fixtures are stored in:

- `fixtures/pipeline_inputs/good_qwen_sample.txt`
- `fixtures/pipeline_inputs/good_kraken_sample.txt`
- `fixtures/pipeline_inputs/tesseract_sample.txt`
- `fixtures/pipeline_inputs/manually_cleaned_reference.txt`

## Running the pipeline

Automatic provider selection:

```bash
python pipeline_runner.py fixtures/pipeline_inputs/good_qwen_sample.txt
```

Offline mock mode:

```bash
python pipeline_runner.py fixtures/pipeline_inputs/good_qwen_sample.txt --provider mock
```

Local Tesseract image input:

```bash
python pipeline_runner.py test-data/jawi-manuscript-4.png --ocr-provider tesseract --provider mock
```

## Outputs

Each run writes a timestamped folder in `output/pipeline/` containing:

- `01_ocr_jawi.txt`
- `02_romanized.txt`
- `03_modern_malay_translation.txt`
- `04_summary.txt`
- `05_knowledge_extraction.json`
- `pipeline_result.json`
- `pipeline_report.md`

## Testing

Run the lightweight offline tests:

```bash
python -m unittest tests/test_pipeline.py
```
