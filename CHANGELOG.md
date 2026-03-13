# Changelog — Agentic RAG Legal Challenge

Detailed log of all pipeline iterations with metrics, for post-competition publication.

---

## v1 — 0.034 (2026-03-11 13:50)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.857 | 0.613 | 0.050 | 0.900 | 3828ms | 0.960 | **0.034** |

- Initial submission. Grounding nearly zero — wrong corpus or doc_id format.

## v2 — 0.035 (2026-03-11 14:09)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.857 | 0.613 | 0.050 | 0.900 | 2477ms | 0.990 | **0.035** |

- Minor TTFT improvement. Grounding still broken.

## v3 — 0.036 (2026-03-11 14:17)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.857 | 0.640 | 0.050 | 0.900 | 1997ms | 1.002 | **0.036** |

- Slightly better assistant score. Grounding still broken (`.pdf` extension bug in doc_ids).

## v4 — 0.438 (2026-03-11 14:27)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.857 | 0.673 | 0.550 | 0.986 | 1936ms | 1.005 | **0.438** |

- **Fixed doc_id format** (removed `.pdf` extension) — grounding jumped 0.05 -> 0.55
- Fixed telemetry format — T: 0.90 -> 0.986
- First meaningful submission

## v5 — 0.422 (2026-03-11 19:09)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.886 | 0.647 | 0.521 | 0.990 | 1892ms | 1.005 | **0.422** |

- Det improved (0.857 -> 0.886) but grounding regressed (0.550 -> 0.521)
- Score slightly lower than v4 overall

## v6 — 0.462 (2026-03-11 21:43)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.800 | 0.607 | 0.610 | 0.981 | 960ms | 1.040 | **0.462** |

- BM25+embeddings hybrid (all-MiniLM-L6-v2) with RRF fusion
- Document routing index (case numbers + law names -> doc_ids)
- CoT for comparison booleans
- CITE-based grounding
- Context distillation
- Temperature=0
- **Grounding best so far** (0.610) but Det regressed (0.800)
- TTFT halved (1892 -> 960ms), F bonus improved

## v7 — 0.559 (2026-03-12 05:19)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.929 | 0.593 | 0.659 | 0.991 | 1020ms | 1.035 | **0.559** |

- Adversarial question detection (jury, plea, Miranda, parole)
- Page-specific retrieval (last/second/title page, Date of Issue)
- Cached Anthropic client for fast TTFT
- chars_per_page tuning by answer type
- **Det jumped to 0.929** (best ever), G improved to 0.659

## v8 — 0.704 (2026-03-12 09:36)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.929 | 0.713 | 0.788 | 0.996 | 937ms | 1.038 | **0.704** |

### Changes (sessions 8-9 combined):
1. ±1 page expansion for ALL answer types (β=2.5 favors recall)
2. ±2 expansion for free_text
3. Multi-doc coverage for ALL comparison types
4. Article number BM25 boost + type-aware query expansion
5. Section/Part keyword support in article detection
6. Smart free_text truncation (sentence-boundary)
7. Block reference stripping from LLM answers
8. Comparison boolean doc coverage (at least 1 page per doc)
9. Targeted page injection (Schedule 3, Conclusion sections)
10. Multi-article support (detect multiple Article+Law refs)
11. Post-citation verification
12. Free_text prompt improvements
13. LLM timeout (8s)
14. Reduce free_text over-citation
15. Single-entity doc filtering for grounding precision
16. Restrict comparison boolean citations to 1 page per doc
17. Article-reference filter for boolean citations

### Key improvement: Grounding 0.659 -> 0.788 (+0.129)

## v9 — 0.626 (2026-03-12 19:42)
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.957 | 0.680 | 0.690 | 0.996 | 836ms | 1.041 | **0.626** |

- Det improved (0.929→0.957) but G regressed (0.788→0.690)
- Adjacent page retention in article filters added too many irrelevant pages
- Net negative: -0.078 total

## v10 — 0.716 (2026-03-13 16:37) ← current best
| Det | Asst | G | T | TTFT | F | **Total** |
|-----|------|---|---|------|---|-----------|
| 0.971 | 0.687 | 0.781 | 0.996 | — | 1.040 | **0.716** |

### Changes (sessions 13-14):
1. **Evidence-span grounding overhaul** — CITE + article + evidence page merge
2. **Evidence UNION** — keep BOTH CITE and evidence pages for typed answers
3. **Recall floor** — min 2 pages for free_text
4. **3 Det fixes**: SCT 295/2025, AED 405M claim, 30 days Art 10(3)
5. **Post-LLM text verification** — `_verify_in_text()` overrides hallucinated answers
6. **`_number_search_variants` fix** — bare digit form always included
7. **retrieve.py O(N) optimization** — 17 corpus scans → O(1) dict lookups
8. **API retry** with exponential backoff for 429/5xx
9. **Reranker hard cap** 30 pages
10. **Sonnet timeout** 12→20s, Haiku 6→8s

### Key improvement: Det 0.929 → 0.971 (+0.042), new best total

---

## Score progression

```
v1  ████                                          0.034
v2  ████                                          0.035
v3  ████                                          0.036
v4  ██████████████████████                        0.438
v5  █████████████████████                         0.422
v6  ███████████████████████                       0.462
v7  ████████████████████████████                  0.559
v8  ███████████████████████████████████           0.704
v9  ███████████████████████████████               0.626
v10 ████████████████████████████████████          0.716
```

## Metric progression
| Version | Det | Asst | G | T | TTFT | F | Total |
|---------|-----|------|-------|-------|------|-------|-------|
| v1 | 0.857 | 0.613 | 0.050 | 0.900 | 3828 | 0.960 | 0.034 |
| v2 | 0.857 | 0.613 | 0.050 | 0.900 | 2477 | 0.990 | 0.035 |
| v3 | 0.857 | 0.640 | 0.050 | 0.900 | 1997 | 1.002 | 0.036 |
| v4 | 0.857 | 0.673 | 0.550 | 0.986 | 1936 | 1.005 | 0.438 |
| v5 | 0.886 | 0.647 | 0.521 | 0.990 | 1892 | 1.005 | 0.422 |
| v6 | 0.800 | 0.607 | 0.610 | 0.981 | 960  | 1.040 | 0.462 |
| v7 | 0.929 | 0.593 | 0.659 | 0.991 | 1020 | 1.035 | 0.559 |
| v8 | 0.929 | 0.713 | 0.788 | 0.996 | 937  | 1.038 | 0.704 |
| v9 | 0.957 | 0.680 | 0.690 | 0.996 | 836  | 1.041 | 0.626 |
| v10 | 0.971 | 0.687 | 0.781 | 0.996 | — | 1.040 | 0.716 |

## Key architectural decisions
- **BM25+embeddings hybrid** with RRF fusion (not pure dense)
- **Haiku** for extractive, **Sonnet** for free_text
- **CoT reasoning** only for comparison booleans
- **Evidence-span grounding**: CITE + article + text-verified pages
- **Page-level retrieval** (not chunk-level) to match grounding format
- **Document routing index** for multi-doc comparison questions
- **Post-LLM verification** catches hallucinated typed answers

## Tools used
- Claude Code (AI-assisted development)
- Anthropic Haiku + Sonnet (generation)
- all-MiniLM-L6-v2 (embeddings)
- cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking)
- rank_bm25 (sparse retrieval)
