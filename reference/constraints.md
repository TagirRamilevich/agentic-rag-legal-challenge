# Competition constraints (from Discord Q&A)

## Team / submissions
- Team size: up to 3 people.
- Final: up to 2 submissions; leaderboard counts the best final score.

## Documents / questions
- Some PDFs contain screenshots of text → OCR may be needed.
- Corpus is fixed; you cannot enrich/extend it.
- You may preprocess/index documents before evaluation; preprocessing time is NOT scored.
- You cannot use questions during indexing stage (index documents only).

## Answer formats
- Deterministic answer types must be correct JSON types: int/number, string, list[string], bool, or null.
- Boolean answers: JSON true/false (not "Yes"/"No").
- Dates: ISO 8601 YYYY-MM-DD.
- Numbers: numeric only, no currency; evaluation uses ±1% tolerance for numbers.
- Unanswerable:
  - deterministic types → answer = null, references empty
  - free_text → short explicit “not available” message, references empty
- Free_text length: max 280 characters, 1–3 sentences.

## Grounding / sources
- References are page-level (even if you chunk smaller internally).
- Page numbers are physical PDF pages, 1-based (first page = 1), not printed labels.
- If answer spans pages, include all those pages.
- Include ONLY pages actually used to generate the final answer.
- Grounding uses weighted F-score with β=2.5 (recall much more important than precision).
- Gold references are intended to be unambiguous; don’t list “all possible sources”.

## Telemetry / TTFT
- TTFT is time to first token of the FINAL answer returned to the user.
- If not streaming, TTFT = total response time.
- Parallel model calls are allowed, but TTFT must include time spent on query rewrite/search/etc.
- Prize contenders undergo code audit; suspicious telemetry can be flagged.
- TTFT multiplier range: 0.85 (avg TTFT > 5s) to 1.05 (avg TTFT < 1s).

## Submission artifacts
- Submit 2 files: submission.json + code_archive.zip (private; used for reproducibility).
- Do NOT include API keys in the archive.
- Using Claude Code / Claude API in pipeline is allowed if reproducible and included.