# CLAUDE.md — Agentic RAG Legal Challenge Solution Guide

You are working inside this repository to build a competition-grade RAG pipeline.
Read and follow these files first:
- README.md (starter kit overview)
- API.md (submission JSON + telemetry rules)
- EVALUATION.md (scoring, grounding-first)
- openapi.yaml (API schema)

## Primary objective
Maximize total score by prioritizing:
1) Grounding accuracy (correct doc_id + page_numbers, precise citations)
2) Deterministic correctness (exact answers for extractable questions)
3) Assistant quality (clear, non-hallucinated explanations)
4) Telemetry completeness (required fields, correct formats)
5) Speed (minimize TTFT; avoid unnecessary LLM calls)

## Hard rules (must comply)
- Do NOT commit or include real API keys in the code archive. Use `.env.example` only.
- Each phase has its own corpus (warmup/final). Index each corpus independently.
- Telemetry:
  - `doc_id` must be the PDF filename (SHA-like string), NOT a human label.
  - `page_numbers` are physical PDF pages, 1-based indexing.
  - Include ONLY pages actually used to form the final answer. Extra pages reduce grounding precision.
  - If no streaming is used, set `ttft_ms == total_time_ms`.

## Repository layout (do not change without updating this file)
- `src/pipeline/` ingestion, indexing, retrieval, answering, telemetry
- `src/runners/` runnable entrypoints for warmup/final
- `scripts/` CLI utilities: download corpus, build submission, submit
- `configs/rag.yaml` main config

## Implementation plan (execute in this order)
1) Data download:
   - Implement `scripts/download_corpus.py` using `arlc.EvaluationClient.from_env()`
   - Download questions JSON + documents ZIP into `DOCS_DIR/<phase>/`
2) Ingestion:
   - Parse PDFs into per-page text.
   - Handle scanned PDFs: add OCR fallback (only when extracted text is empty/very small).
   - Persist page-level units with:
     - doc_id (filename)
     - page_number (1-based)
     - text
3) Indexing:
   - Build a retrieval index over page-level units (BM25 + embeddings if available).
   - Persist indexes per phase to disk (cache folder).
4) Retrieval:
   - High recall: retrieve top-K pages (K large).
   - Rerank to top-k pages for context (k small).
   - Deduplicate and merge contiguous pages where useful.
5) Answering:
   - Route by answer_type:
     - For extractable facts: prefer deterministic extraction from retrieved pages (regex / string match).
     - Otherwise: single LLM call to synthesize grounded answer.
   - Always output answer in the expected type: string/number/bool/array-of-strings/null.
6) Telemetry:
   - Measure timings for the final answer generation stage.
   - Count tokens using provider-reported usage if possible; otherwise approximate.
   - Populate `retrieved_chunk_pages` with doc_id + page_numbers actually used.
7) Submission build:
   - Produce `submission.json` that matches schema (no `version` field).
   - Create `code_archive.zip` <= 25MB with reproducible code (exclude corpora, caches, secrets).
8) Local validation:
   - Add JSON schema checks for submission format.
   - Add sanity checks: unknown doc_id, invalid page numbers, missing telemetry fields.

## Commands (should work)
- Install: `pip install -r requirements.txt`
- Download warmup: `python scripts/download_corpus.py --phase warmup`
- Run warmup: `python src/runners/run_warmup.py`
- Submit: `python scripts/submit.py submission.json code_archive.zip`

## Output contracts
- `submission.json` structure must match API.md and openapi.yaml schema.
- Use `answers[*].telemetry` exactly as required.
- Always keep `architecture_summary` <= 500 chars.

## Coding constraints
- Keep code simple and reproducible.
- Avoid unnecessary dependencies unless clearly beneficial.
- Prefer deterministic behavior (fixed seeds, stable retrieval).

##Additional context:
- reference/discord_qa.md contains organizer Q&A. Use it to interpret rules and edge cases.
- Do not include reference/ in code_archive.zip.
- Do not include claude/ in code_archive.zip
– Commit to github repository significant changes you make
