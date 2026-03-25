# Legal RAG Pipeline — ARLC 2026

My solution for [ARLC 2026](https://www.agentic-challenge.ai/) (Agentic RAG Legal Challenge) — a competition on building RAG pipelines over DIFC legal documents.

**Results:** Warmup 0.791 (v14 best), Final 0.457

**Write-up:** [Habr article (Russian)](https://habr.com/ru/users/tagir_analyzes/) — full technical breakdown with code, metrics for all 17 iterations, and lessons learned.

## Architecture

```
PDF Corpus → Ingestion (pdfplumber + OCR) → BM25 + Embedding Index
                                                    ↓
Question → Doc Routing → Hybrid Retrieval (RRF) → Cross-encoder Rerank
                                                    ↓
                    Deterministic Fast Path ←→ LLM (Haiku/Sonnet)
                                                    ↓
                    Post-LLM Verification → Evidence-Based Grounding
```

Key design decisions:
- **Page-level retrieval** (not chunks) — grounding is evaluated per physical PDF page
- **Hybrid BM25 + embeddings** with RRF fusion
- **Deterministic fast paths** for extractable questions (regex, ~1ms, no API cost)
- **Haiku** for extractive, **Sonnet** for free_text
- **Post-LLM verification** catches hallucinations via text matching
- **Evidence-based grounding** — article index → CITE → verify

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in EVAL_API_KEY and ANTHROPIC_API_KEY

python scripts/download_corpus.py --phase warmup
python scripts/build_submission.py --phase warmup
python scripts/submit.py submission.json code_archive.zip
```

## Score progression

| Version | Det | G | Total | Key change |
|---------|-----|-------|-------|------------|
| v1 | 0.857 | 0.050 | 0.034 | First submit (`.pdf` bug) |
| v4 | 0.857 | 0.550 | 0.438 | Fixed doc_id format |
| v8 | 0.929 | 0.788 | 0.704 | Multi-doc + F-beta math |
| v14 | 0.971 | 0.862 | **0.791** | Precision-focused grounding |
| F-v1 | 0.696 | 0.647 | 0.457 | Final phase (300 docs) |

## Stack

- Python 3.9+
- Claude Haiku / Sonnet (Anthropic API)
- rank_bm25 (sparse retrieval)
- sentence-transformers: all-MiniLM-L6-v2 (embeddings), ms-marco-MiniLM (reranking)
- PyMuPDF + Tesseract OCR
- Built with Claude Code

## Cost

87.95 USD total API cost for the entire competition (17 warmup runs + 2 final submissions).
