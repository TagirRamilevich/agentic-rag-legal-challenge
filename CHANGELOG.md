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

## v8 — 0.704 (2026-03-12 09:36) ← current best
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

### Tried and reverted:
- **Sonnet for comparison booleans**: 11/19 returned null (too cautious). Reverted to Haiku.
- **Prompt caching via system messages**: No measurable TTFT benefit.

### Key improvement: Grounding 0.659 -> 0.788 (+0.129)
- ±1 expansion math: missing gold page (G: 1.0->0.537) costs far more than extra page (G: 1.0->0.879)
- Multi-doc coverage ensures both case docs cited in comparisons

## v9 (unreleased) — Session 10 precision optimizations

### Changes:
1. **Single-entity doc filtering** — For non-comparison questions, restrict citations to the primary document using 3 strategies: case reference matching, law name matching (page 1 title), DIFC Law No. matching
2. **Free_text ±1 expansion removed** — Trust Sonnet's SOURCES citation directly. Avg pages: 2.4 → 1.1 (54% reduction). For 1-page gold: G jumps 0.78 → 1.0
3. **1-page-per-doc for comparison bool + name** — Keep only lowest page per doc to improve precision
4. **Article-reference filter extended** — Now applies to number/date/name (was bool-only). Removes ±1 expanded pages that don't contain the referenced article
5. **Fixed false comparison detection** — Removed "time" from comparison regex; "same time" ≠ comparison
6. **Free_text context size** — max_pages 5, chars 2000 (was 1500)
7. **Free_text page cap** — Max 3 cited pages (was uncapped)

### Metrics (local, pre-submission):
- Pages avg: 2.2 → 1.7 overall (22% reduction)
- Free_text pages: 2.4 → 1.1 (54% reduction)
- 0 nulls, 4 adversarial skips
- TTFT: ~920ms
- Estimated G improvement: 0.788 → ~0.82-0.85

### Session 11 additions:
8. **Skip boolean ±1 expansion when LLM citation available** — Trust CITE indices directly. For 1-page gold: G=1.0 vs G=0.879 with expansion (+0.121 per question)
9. **Evidence-replace for extractive types** — For non-comparison number/date/name: REPLACE LLM citation with evidence-verified pages from primary doc (more precise than UNION)
10. **Conservative article-aware inclusion** — Only add article pages when not already cited. Prevents adding redundant pages.
11. **Tighter caps** — names 3→2, free_text 4→3 for non-comparison
12. **Word-number fallback** — If LLM returns "six" instead of 6, convert via dictionary
13. **Date format fallback** — Parse "15 March 2024" and "March 15, 2024" if LLM doesn't return YYYY-MM-DD
14. **Clean up redundant comparison check** in retrieve.py

### Session 12 additions:
15. **Fix negative number extraction** — `parse_number` treated `(6)` in "six (6) months" as accounting negative → always positive for legal counts/durations. Fixed 9 deterministic answers.
16. **Strip Article/Section sub-references before number parsing** — Prevents extracting sub-article numbers like "(2)" from "Article 14(2)(b)".
17. **Law number extraction pattern** — For "what is the law number" questions, extract "Law No. X" directly instead of picking up the year. Fixed 3 answers.
18. **Structured comparison-date-name extraction** — For "which case has earlier date" questions, extract dates from both case docs and compare. Fixed 10 answers.
19. **Context size reduction** — bool: 5→4 pages 1500→1200 chars (-34%), free_text: 5→4 pages 2000→1500 chars (-39%), number: 4→3 pages (-25%). Expected TTFT: 937→700-750ms.
20. **Better distill_page article scoring** — "Article N" heading gets +5, just N gets +2. Sub-clause match bonus.
21. **Adversarial detection for ALL answer types** — Not just free_text. Returns null + empty pages for deterministic types. Prevents G=0.0 when gold expects null.
22. **LLM-null deterministic fallback** — When LLM returns null but context has extractable info, try deterministic extraction. Reduces false nulls.
23. **Improved free_text prompt** — Allow 150-250 chars (was "UNDER 200"), address all aspects for better completeness score.

### Deterministic answers fixed in session 12: 22 total
- 9 negative numbers → positive (Q20, Q28, Q37, Q47, Q48, Q55, Q56, Q80, Q89)
- 3 law numbers fixed (Q3: 2018→2, Q53: 2005→4, Q96: 2004→3)
- 10 comparison-name questions fixed (Q9, Q11, Q16, Q42, Q49, Q65, Q68, Q73, Q93, Q94)

### Expected score: ~0.78-0.85 (up from 0.704)

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
v9  ██████████████████████████████████████        0.7?? (pending)
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

## Leaderboard (warmup, 2026-03-12 ~18:00)
| Rank | Team | Total | Det | Asst | G | T | TTFT | F |
|------|------|-------|-----|------|-------|-------|------|-------|
| 1 | KenKo | 0.858 | 0.971 | 0.720 | 0.922 | 0.996 | 904 | 1.043 |
| 2 | Kovalyoff | 0.824 | 0.971 | 0.687 | 0.890 | 0.996 | 84 | 1.050 |
| 3 | Dmitry Ulybin | 0.760 | 0.943 | 0.607 | 0.893 | 0.997 | 1945 | 1.013 |
| 4 | StepSolutions | 0.735 | 0.929 | 0.773 | 0.840 | 0.999 | 2508 | 0.993 |
| 5 | IAS Partners | 0.716 | 0.943 | 0.740 | 0.804 | 0.994 | 1714 | 1.016 |
| **6** | **Tagir Analyzes** | **0.704** | **0.929** | **0.713** | **0.788** | **0.996** | **937** | **1.038** |
| 7 | Tzur Labs | 0.696 | 0.943 | 0.707 | 0.766 | 0.996 | 375 | 1.046 |
| 8 | MaxOps AI | 0.612 | 0.893 | 0.567 | 0.745 | 0.994 | 820 | 1.039 |
| 9 | RAGdolls | 0.593 | 0.957 | 0.653 | 0.655 | 0.996 | 16 | 1.050 |

### Gap analysis (us vs top-3):
- **Det**: 0.929 vs KenKo 0.971 — gap 0.042 (need ~1-2 more correct deterministic answers)
- **G**: 0.788 vs KenKo 0.922 — gap 0.134 (biggest opportunity)
- **Asst**: 0.713 vs StepSolutions 0.773 — gap 0.060

### Special nomination candidates:
- **Speed Champion**: RAGdolls (16ms) or Tzur Labs (375ms) — we're at 937ms, not competitive
- **Retrieval Master**: KenKo (0.922) dominates — we're at 0.788
- **Efficiency Expert**: unclear metric, but our Haiku usage is very efficient
- **Best Publication**: open field — our main target

## Key architectural decisions
- **BM25+embeddings hybrid** with RRF fusion (not pure dense)
- **Haiku** for generation (fast, cheap, good enough)
- **CoT reasoning** only for comparison booleans
- **Deterministic fallback** when API unavailable
- **Page-level retrieval** (not chunk-level) to match grounding format
- **Document routing index** for multi-doc comparison questions

## Tools used
- Claude Code (AI-assisted development)
- Anthropic Haiku (generation)
- all-MiniLM-L6-v2 (embeddings)
- rank_bm25 (sparse retrieval)
