# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FoodHealthQA: Vietnamese food-label Q&A for additives and allergens. Streamlit app (`app.py`) plus a research harness (`scripts_research/`). Docs, README, and UI strings are in Vietnamese. Image → OCR → normalization/entity linking → retrieval → deterministic safety rules → LLM-phrased answer with citations, or principled abstention.

## Commands

Windows; use the project venv (`venv\`, Python 3.10/3.11 only; `venv_ocr3` is a separate env for PaddleOCR 3.x experiments).

```powershell
.\venv\Scripts\python.exe -m pytest tests                      # all tests (pytest.ini: testpaths=tests)
.\venv\Scripts\python.exe -m pytest tests/test_abstention.py::TestName::test_x   # single test
.\venv\Scripts\python.exe -m unittest discover tests           # README's alternative runner
streamlit run app.py                                           # UI at localhost:8501
```

`tests/test_ocr_regression.py` runs OCR regression against 376 real images and `data/ocr_regression_baseline.json`. Update the baseline (`scripts_research/ocr_regression.py --update-baseline`) only for an approved change.

No linter or build step is configured.

Research pipeline (order matters; see README "Kịch Bản Nghiên Cứu" for the full list):
```powershell
python scripts_research\calibrate_confidence.py --stage all     # OCR cache + rule layer + abstention threshold; no LLM, don't run concurrently with generate_predictions
python scripts_research\generate_predictions.py --llm-provider gemini|ollama [--resume|--status]   # writes data/runs/<provider>/
python scripts_research\run_evaluation.py --run-dir data/runs/ollama
python scripts_research\compare_providers.py
python scripts_research\ocr_regression.py --worst 10
```

## Architecture

- **Three configurations** (ablation). A = LLM only, closed-book. B = reference label text plus the full retrieval/rule stack. C = real image → OCR → same stack as B. B and C share code, so B-vs-C isolates OCR error. The UI exposes only C; A and B run from `scripts_research/` only. Details: `docs/QUY_TRINH_3_CAU_HINH_HE_THONG.md`.
- **`modules/pipeline.py`** (~3000 lines) is the hub: KB/persona/NHANES/FNDDS loaders, entity linking (`link_all_mentions`, FoodOntoRAG 4-agent), the deterministic rule engine (`_evaluate_safety_core` / `evaluate_safety_and_guardrail`), LLM calls (Gemini with multi-key fallback, or Ollama Qwen2.5, selected by `LLM_PROVIDER`), and answer generation. **Rules decide the safety label; the LLM only phrases it** and its citations are cleaned/validated afterward (`clean_llm_citations_and_label`).
- Supporting modules in `modules/`:
  - Perception: `ocr_extractor` (PaddleOCR DBNet + VietOCR, preprocessing, ingredient extraction), `ocr_lexicon` (domain spelling correction).
  - Linking: `additive_linker` (INS/E-number), `foodon`.
  - Retrieval and QA: `dynamic_qa` (the question decides what to look up and records the trace), `passage_retriever`/`hybrid_retriever`/`ollama_embedder` (BM25+TF-IDF, optional bge-m3 dense).
  - Safety and verification: `abstention`, `constraint_model`, `evidence`, `nli_verifier`, `infant_rules`.
  - Evaluation: `evaluation`, `benchmark`, `ablation`, `compare_runner`.
- **Abstention**: `CONFIDENCE_ABSTAIN_THRESHOLD` (currently 0.65) is calibrated by `calibrate_confidence.py`. A question naming an additive code that is absent from both the KB and the label is refused rather than guessed.
- **Knowledge base**: `knowledge_base/*.json` (additive registry, standards, allergen groups, OCR lexicon). `knowledge_base/sources/` holds raw PDFs, derived data, and `catalog/source_citation_catalog.json` (the citation keys such as `[VN_TT24_2019]`).
- **Data** (`data/`):
  - The only evaluation set is `foodhealthqa_eval_set.jsonl` (750 Q&A, 376 images); every script defaults to it and errors if it is missing.
  - `qa_benchmark.jsonl` (2,311) is the source for building the 750 set; never score or demo against it.
  - The benchmark is silver/rule-verified, with no expert validation (`expert_validated=false`). Don't claim gold-standard performance.
  - `data/runs/`, `data/cache/`, and predictions are regenerable outputs.
  - `data/additives_standard.csv` (10 rows) is not the KB; `knowledge_base/additives_standard.json` is.
- Benchmark/leakage rule: the QA benchmark must never be used as retrieval KB or in the UI. Split and resample by `split_group` (one image = one group); the OCR lexicon is built only from the development images.

## Configuration

Env/`.env`: `LLM_PROVIDER` (`gemini` default | `ollama`, model via `OLLAMA_MODEL`), `GEMINI_API_KEYS` or `gemini_keys.txt` (template `gemini_keys.example.txt`), `PASSAGE_EMBEDDING=bge-m3` (needs `ollama pull bge-m3`; otherwise BM25+TF-IDF).

## Conventions

- Reported metrics must be measured from data or be an explicitly documented rule. Otherwise mark them "not applicable"; don't invent constants.
- Changing the eval set, `persona_conditions`, or OCR logic invalidates the previously computed metrics, so rerun them.
- This repo's Bash tool can collapse `\\` into control characters when writing files. Patch files with Write/Edit and grep for `\x08\x0b` afterward.
