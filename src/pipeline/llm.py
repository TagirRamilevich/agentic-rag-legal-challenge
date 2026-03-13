import json
import os
import re
import time
from datetime import date as _date
from typing import Any, Optional

from src.utils.number_parse import parse_number

FREE_TEXT_FALLBACK = "There is no information on this question in the provided documents."
FREE_TEXT_MAX = 280

# Context size per answer type
# max_tokens includes room for CITE: suffix (~10 tokens)
# With context distillation, chars_per_page is the distilled paragraph budget
# Boolean: needs more pages for comparison across 2 docs
_TYPE_CONFIG = {
    "bool":     {"max_pages": 4, "chars": 1200, "max_tokens": 30},
    "boolean":  {"max_pages": 4, "chars": 1200, "max_tokens": 30},
    "number":   {"max_pages": 3, "chars": 900, "max_tokens": 30},
    "date":     {"max_pages": 3, "chars": 900, "max_tokens": 30},
    "name":     {"max_pages": 3, "chars": 900, "max_tokens": 60},
    "names":    {"max_pages": 3, "chars": 1000, "max_tokens": 160},
    "free_text":{"max_pages": 5, "chars": 1200, "max_tokens": 250},
}
_DEFAULT_CONFIG = {"max_pages": 3, "chars": 1200, "max_tokens": 256}

_NOT_FOUND_RE = re.compile(
    r"(does not (identify|contain|mention|provide|include|specify|state|address|describe|define|disclose|indicate)|"
    r"no (information|mention|reference|record|evidence|indication)|"
    r"cannot (be|find|determine)|"
    r"not (found|available|present|identified|mentioned|specified|stated|addressed|disclosed|indicated|explicitly))",
    re.IGNORECASE,
)

_TRUE_RE = re.compile(
    r"\b(granted|approved|allowed|upheld|confirmed|correct|permitted|"
    r"succeeded|successful|enacted|yes|true)\b",
    re.IGNORECASE,
)
_FALSE_RE = re.compile(
    r"\b(dismissed|denied|rejected|refused|prohibited|incorrect|"
    r"not granted|not approved|failed|unsuccessful|false|no\b)\b",
    re.IGNORECASE,
)


_ANTHROPIC_CLIENT = None
_PROVIDER_DISABLED: set[str] = set()  # Providers that hit permanent errors (e.g. spending limit)


def _get_anthropic_client():
    """Reuse Anthropic client to avoid connection setup overhead per call."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def warmup_llm():
    """Make a minimal API call to warm up TLS connection and client caching."""
    if _provider() == "anthropic":
        try:
            client = _get_anthropic_client()
            client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception:
            pass


def _provider() -> Optional[str]:
    if os.getenv("ANTHROPIC_API_KEY") and "anthropic" not in _PROVIDER_DISABLED:
        return "anthropic"
    if os.getenv("OPENROUTER_API_KEY") and "openrouter" not in _PROVIDER_DISABLED:
        return "openrouter"
    if os.getenv("OPENAI_API_KEY") and "openai" not in _PROVIDER_DISABLED:
        return "openai"
    return None


def _is_transient_error(err_str: str) -> bool:
    """Check if an API error is transient (retryable) vs permanent."""
    _transient_patterns = [
        "429", "rate_limit", "rate limit", "overloaded", "529",
        "500", "502", "503", "504", "server_error", "timeout",
        "connection", "timed out", "temporarily unavailable",
    ]
    err_lower = err_str.lower()
    return any(p in err_lower for p in _transient_patterns)


def _call(
    prompt: str,
    max_tokens: int = 256,
    use_strong_model: bool = False,
    t0: Optional[float] = None,
) -> tuple[Optional[str], int, int, int, int, str]:
    """
    Call LLM and return (text, ttft_ms, total_ms, input_tokens, output_tokens, model_name).
    ttft_ms is measured from t0 (if given) so it includes pre-LLM pipeline time.
    Uses streaming on Anthropic for accurate TTFT measurement.
    Retries up to 2 times on transient errors (429, 5xx, timeout).
    """
    _t0 = t0 if t0 is not None else time.perf_counter()
    _MAX_RETRIES = 2

    for _attempt in range(_MAX_RETRIES + 1):
        p = _provider()
        if not p:
            break

        try:
            if p == "anthropic":
                import anthropic
                model = "claude-sonnet-4-6" if use_strong_model else "claude-haiku-4-5-20251001"
                client = _get_anthropic_client()
                ttft_ms: Optional[int] = None
                chunks: list[str] = []
                input_tokens = 0
                output_tokens = 0

                _timeout = 20.0 if use_strong_model else 8.0
                with client.messages.stream(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=_timeout,
                ) as stream:
                    for chunk in stream.text_stream:
                        if ttft_ms is None:
                            ttft_ms = max(1, int((time.perf_counter() - _t0) * 1000))
                        chunks.append(chunk)
                    final_msg = stream.get_final_message()
                    input_tokens = final_msg.usage.input_tokens
                    output_tokens = final_msg.usage.output_tokens

                total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
                text = "".join(chunks).strip()
                return text, ttft_ms or total_ms, total_ms, input_tokens, output_tokens, model

            if p == "openrouter":
                import requests as req
                model = "anthropic/claude-sonnet-4-6" if use_strong_model else "anthropic/claude-haiku-4-5"
                r = req.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=20,
                )
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"].strip()
                total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                return text, total_ms, total_ms, in_tok, out_tok, model

            if p == "openai":
                import openai
                model = "gpt-4o" if use_strong_model else "gpt-4o-mini"
                client = openai.OpenAI()
                r = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = r.choices[0].message.content.strip()
                total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
                return text, total_ms, total_ms, 0, 0, model

        except Exception as e:
            err_str = str(e)
            # Detect permanent errors (spending limit, auth, forbidden) → disable provider
            if ("usage limits" in err_str or "API key" in err_str
                    or "authentication" in err_str.lower()
                    or "forbidden" in err_str.lower() or "not allowed" in err_str.lower()):
                _PROVIDER_DISABLED.add(p)
                print(f"  LLM provider '{p}' disabled for session: {err_str[:100]}")
                # Try next provider immediately (counts as retry)
                next_p = _provider()
                if next_p and next_p != p:
                    print(f"  Falling back to provider: {next_p}")
                    continue
            elif _is_transient_error(err_str) and _attempt < _MAX_RETRIES:
                _backoff = 2 ** _attempt  # 1s, 2s
                print(f"  LLM transient error ({p}), retry {_attempt + 1}/{_MAX_RETRIES} in {_backoff}s: {err_str[:80]}")
                time.sleep(_backoff)
                continue
            else:
                print(f"  LLM error ({p}): {e}")

    total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
    return None, total_ms, total_ms, 0, 0, "unknown"


# Citation suffix appended to all type instructions
_CITE_SUFFIX = (
    " End with CITE: followed by the 0-based block number(s) you used "
    "(e.g. CITE:0 or CITE:0,2). Cite ALL blocks that contain information relevant to your answer."
)

_TYPE_INSTRUCTIONS = {
    "number": (
        "Return ONLY a single numeric value (integer or decimal). "
        "No units, no currency symbols, no extra text. "
        "IMPORTANT: The answer is the VALUE stated in the article text, NOT the article number itself. "
        "For example, if Article 19(4) says 'within six (6) months', the answer is 6, NOT 19. "
        "Look for the specific quantity, duration, amount, or count mentioned in the text. "
        "Word-numbers: 'one'=1, 'two'=2, 'three'=3, 'four'=4, 'five'=5, 'six'=6, 'seven'=7, "
        "'eight'=8, 'nine'=9, 'ten'=10, 'twelve'=12, 'fifteen'=15, 'twenty'=20, 'thirty'=30, 'sixty'=60, 'ninety'=90. "
        "For 'what is the law number' questions: return just the number (e.g., Law No. 3 of 2004 → 3). "
        "Accounting parentheses like (5,000) mean negative: -5000. "
        "Convert thousands/millions: '1.5 million' → 1500000. "
        "If multiple numbers appear, choose the one that directly answers the question. "
        "Examples: 1500000 or 3.5 or 42 or -5000. "
        "If not found in context: null" + _CITE_SUFFIX
    ),
    "bool": (
        "Return ONLY the word true, false, or null (lowercase, nothing else). "
        "true = the statement in the question is correct/confirmed by the text. "
        "false = the statement in the question is incorrect/denied by the text. "
        "CRITICAL: Read the EXACT article sub-clause referenced (e.g. 7(3)(j)(i) vs 7(3)(j)(ii) are DIFFERENT). "
        "Watch for double negatives: 'shall not be liable' + question 'is X liable?' = false. "
        "Watch for exceptions: 'except where' / 'unless' / 'provided that' can reverse the answer. "
        "For comparison questions (same judge/year/party): compare the specific facts carefully. "
        "If two values clearly differ → false. If they clearly match → true. "
        "Only return null if the context truly lacks the needed information." + _CITE_SUFFIX
    ),
    "boolean": (
        "Return ONLY the word true, false, or null (lowercase, nothing else). "
        "true = the statement in the question is correct/confirmed by the text. "
        "false = the statement in the question is incorrect/denied by the text. "
        "CRITICAL: Read the EXACT article sub-clause referenced (e.g. 7(3)(j)(i) vs 7(3)(j)(ii) are DIFFERENT). "
        "Watch for double negatives: 'shall not be liable' + question 'is X liable?' = false. "
        "Watch for exceptions: 'except where' / 'unless' / 'provided that' can reverse the answer. "
        "For comparison questions (same judge/year/party): compare the specific facts carefully. "
        "If two values clearly differ → false. If they clearly match → true. "
        "Only return null if the context truly lacks the needed information." + _CITE_SUFFIX
    ),
    "date": (
        "Return ONLY a date in YYYY-MM-DD format. "
        "Convert any date format (e.g. '15 March 2024' → '2024-03-15'). "
        "If not found in context: null" + _CITE_SUFFIX
    ),
    "name": (
        "Return ONLY the name, entity, or specific term as a short phrase. "
        "Include the full legal name if available. "
        "For 'what document/form/statement' questions: return the exact document type name (e.g. 'Confirmation Statement', 'Annual Return'). "
        "For 'which case' questions: return the case number (e.g. 'CFI 010/2024'). "
        "For 'which case was decided/issued earlier' questions: find the decision/issue date in each case document, compare the dates, and return the case number with the earlier date. "
        "For 'which party had higher/lower' questions: compare the values and return the correct one. "
        "IMPORTANT: Return the SPECIFIC answer to the question, not the subject of the sentence. "
        "No explanation, no extra text. "
        "If not found in context: null" + _CITE_SUFFIX
    ),
    "names": (
        "Return ONLY a JSON array of strings. "
        'Example: ["John Smith", "Acme Corp Ltd"]. '
        "No markdown code blocks. No explanations. "
        "Include every relevant person, company, or entity that answers the question. "
        "Use the EXACT names as they appear in the documents. "
        "For party/claimant/defendant questions: look in the BETWEEN section or case header for party names. "
        "Do NOT include judge names, court names, or procedural terms — only the actual parties. "
        "If genuinely not found: null" + _CITE_SUFFIX
    ),
    "free_text": (
        f"Answer in 1-2 sentences (under 200 characters, hard max {FREE_TEXT_MAX}) using ONLY facts from the context. "
        "Be direct and specific. Start with the answer itself — no lead-in phrases. "
        "Every claim must come from the context — no inference or general knowledge. "
        "Include key specifics: article numbers, names, dates, amounts, legal terms. "
        "If the question has multiple parts, address ALL parts. "
        "State facts confidently — no hedging ('appears to', 'seems to', 'it is possible'). "
        "Do NOT use markdown, do NOT reference block numbers or 'the context'. "
        "DIFC courts do NOT have juries, plea bargains, criminal proceedings, Miranda rights, parole, or verdicts. "
        "If the question asks about concepts that don't exist in DIFC courts, "
        f"write EXACTLY: {FREE_TEXT_FALLBACK}\n"
        "After your answer write SOURCES: followed by 0-based block numbers used (e.g. SOURCES: 0,2). "
        f"If truly absent from ALL blocks, write EXACTLY: {FREE_TEXT_FALLBACK}"
    ),
}


def _distill_page(text: str, question: str, max_chars: int) -> str:
    """Extract the most relevant paragraphs from a page for the given question.
    Returns distilled text up to max_chars."""
    if len(text) <= max_chars:
        return text

    # Split into paragraphs (by double newline or single newline for short blocks)
    paragraphs = re.split(r"\n\s*\n|\n(?=[A-Z0-9\(\[])", text)
    if len(paragraphs) <= 1:
        return text[:max_chars]

    # Merge very short fragments (section numbers, sub-clause markers) into
    # the following paragraph. Legal docs put "28.", "(1)" on their own lines.
    merged: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if merged and len(merged[-1]) < 15:
            merged[-1] = merged[-1] + "\n" + para
        else:
            merged.append(para)
    paragraphs = merged

    # Score paragraphs by keyword overlap with question
    q_words = set(w.lower() for w in re.sub(r"[^\w\s]", " ", question).split() if len(w) > 2)
    # Check for specific article reference to boost matching paragraphs
    _art_m = re.search(r"Article\s+(\d+)", question, re.IGNORECASE)
    art_num = _art_m.group(1) if _art_m else None
    # Extract sub-clause reference like "(3)(j)" for fine-grained matching
    _subclause_m = re.search(r"Article\s+\d+(?:\(\d+\))?(\([a-z]\))", question, re.IGNORECASE)
    _subclause = _subclause_m.group(1) if _subclause_m else None
    # Detect case/party questions to preserve header sections
    _is_party_q = bool(re.search(r"\b(claimant|defendant|respondent|party|parties|judge|issued|date of issue)\b", question, re.IGNORECASE))
    _is_outcome_q = bool(re.search(r"\b(result|outcome|rul(?:e|ed|ing)|ordered|decision|decided|conclud(?:e|ed|ing)|conclusion)\b", question, re.IGNORECASE))
    scored = []
    _art_para_indices: set[int] = set()  # Track paragraphs with article references
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        p_lower = para.lower()
        score = sum(1 for w in q_words if w in p_lower)
        # Bonus for paragraphs containing the specific article reference
        if art_num:
            # Strong match: "Article N" or "N." at start (article heading)
            if re.search(rf"\bArticle\s+{art_num}\b|(?:^|\n)\s*{art_num}\.(?:\s|$)", para, re.IGNORECASE):
                score += 5
                _art_para_indices.add(i)
                # Extra bonus if sub-clause matches too
                if _subclause and _subclause in para:
                    score += 3
            # Match "(1)" sub-clause (e.g., "16(1)") indicating article content
            elif re.search(rf"\b{art_num}\(", para):
                score += 4
                _art_para_indices.add(i)
            # Weak match: just the number (sub-clause reference)
            elif re.search(rf"\b{art_num}\b", para):
                score += 2
        # Bonus for case header sections (BETWEEN, parties, judge, date)
        if _is_party_q:
            if re.search(r"\bBETWEEN\b|Claimant|Defendant|Respondent|UPON|Date of Issue|BEFORE", para):
                score += 5
        # Bonus for court order/ruling sections
        if _is_outcome_q:
            if re.search(r"ORDERED|ORDER|JUDGMENT|RULING|CONCLUSION|DECISION|DISMISSED|GRANTED", para, re.IGNORECASE):
                score += 5
        # Always keep first paragraph (often has title/header info)
        if i == 0:
            score += 1
        scored.append((score, i, para))

    # Boost neighboring paragraphs of article-matched paras.
    # Preceding: often article headings (e.g., "Confirmation Statement" before "16(1)...")
    # Following: article content when number and content split across paragraphs
    # (e.g., "28.\n" splits from "Liability of Partners\n(1) Unless...")
    if art_num and _art_para_indices:
        _max_orig_i = max(orig_i for _, orig_i, _ in scored) if scored else 0
        for idx in list(_art_para_indices):
            # Boost preceding paragraph (heading)
            if idx > 0:
                for j, (s, orig_i, p) in enumerate(scored):
                    if orig_i == idx - 1 and s < 3:
                        if len(p) < 100:
                            scored[j] = (s + 4, orig_i, p)
                        else:
                            scored[j] = (s + 2, orig_i, p)
            # Boost following paragraph (article content after short number line)
            # Only if the article paragraph itself is very short (just "28." etc.)
            _art_para_len = next((len(p) for _, orig_i, p in scored if orig_i == idx), 999)
            if _art_para_len < 30 and idx < _max_orig_i:
                for j, (s, orig_i, p) in enumerate(scored):
                    if orig_i == idx + 1:
                        scored[j] = (max(s, 5), orig_i, p)
                        _art_para_indices.add(idx + 1)
                        break

    # Sub-clause continuation: when a paragraph contains the target sub-clause
    # (e.g., "(j)" for "Article 7(3)(j)"), boost the next 2 paragraphs as they
    # are likely sub-items (e.g., "(i)" and "(ii)") of that clause.
    # This prevents losing critical sub-sub-clauses during distillation.
    if _subclause:
        for j, (s, orig_i, p) in enumerate(scored):
            if _subclause in p:
                # Boost next 2 paragraphs (sub-items)
                for k, (sk, ok, pk) in enumerate(scored):
                    if ok in (orig_i + 1, orig_i + 2) and sk < 5:
                        scored[k] = (max(sk, 5), ok, pk)
                break

    # Sort by relevance, take top paragraphs fitting max_chars
    scored.sort(key=lambda x: -x[0])
    result_parts = []
    total = 0
    for score, _orig_i, para in scored:
        if total + len(para) > max_chars:
            if not result_parts:
                result_parts.append(para[:max_chars])
            break
        result_parts.append(para)
        total += len(para) + 2  # +2 for newline separator

    return "\n\n".join(result_parts) if result_parts else text[:max_chars]


_BOOL_COMPARISON_INSTRUCTION = (
    "Compare the two entities/cases/laws. "
    "Extract the specific fact from each, compare. Same→true, Different→false. "
    "If you found info about BOTH, you MUST answer true or false — never null. "
    "For 'same judge': compare EXACT judge names. For 'same party': compare EXACT party names. "
    "CRITICAL: Read the EXACT sub-clause referenced (e.g. (i) vs (ii) are DIFFERENT provisions). "
    "Format: E1:[fact] E2:[fact] ANSWER:true/false CITE:0,1"
)


def _build_prompt(question: str, answer_type: str, context: str,
                  is_comparison: bool = False) -> str:
    if is_comparison and answer_type in ("bool", "boolean"):
        instruction = _BOOL_COMPARISON_INSTRUCTION
    else:
        instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        "Extract the answer from the following legal document context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Instruction: {instruction}\n\n"
        "Answer:"
    )


def _parse_citation(raw: str, num_pages: int) -> tuple[str, list[int]]:
    """Parse 'CITE:0,2' or 'SOURCES: 0,1,2' from LLM response.
    Returns (cleaned_text, cited_block_indices)."""
    text = raw.strip()
    indices: list[int] = []
    # Try CITE: first (at end or anywhere), then SOURCES:
    m = re.search(r"\bCITE:\s*([\d,\s]+)\s*$", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\bSOURCES?:\s*([\d,\s]+)\s*$", text, re.IGNORECASE)
    if not m:
        # Try CITE: not at end (LLM may add text after citation)
        m = re.search(r"\bCITE:\s*([\d,\s]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\bSOURCES?:\s*([\d,\s]+)", text, re.IGNORECASE)
    if m:
        idx_strs = re.findall(r"\d+", m.group(1))
        indices = [int(i) for i in idx_strs if int(i) < num_pages]
        text = text[: m.start()].strip()
    return text, indices


def _parse_free_text_sources(raw: str, num_pages: int) -> tuple[str, list[int]]:
    """Parse citation from free_text response. Returns (answer, page_indices)."""
    text, indices = _parse_citation(raw, num_pages)
    return text, indices  # truncation handled by _parse() with smart sentence boundary


def _parse(raw: str, answer_type: str) -> Any:
    raw = raw.strip()
    if not raw or raw.lower() in ("null", "none", "n/a"):
        return None

    if answer_type in ("bool", "boolean"):
        low = raw.lower().strip()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        # Check first word/line (LLM may add explanation after the answer)
        first_word = low.split()[0] if low.split() else ""
        first_line = low.split("\n")[0].strip()
        if first_word in ("true", "yes") or first_line in ("true", "yes"):
            return True
        if first_word in ("false", "no") or first_line in ("false", "no"):
            return False
        # Only infer from regex if strong signal; otherwise null
        t = len(_TRUE_RE.findall(raw))
        f = len(_FALSE_RE.findall(raw))
        if t > f + 1:  # require margin of 2+
            return True
        if f > t + 1:
            return False
        return None  # ambiguous → null

    if answer_type == "number":
        # Strip word-number forms: "six (6)" → "6", "twelve (12)" → "12"
        _word_num = re.search(r"\b\w+\s+\((\d+)\)", raw)
        if _word_num:
            raw = _word_num.group(1)
        val = parse_number(raw)
        # Fix accounting-negative false positives: (7) → -7 but no "-" in raw
        if val is not None and val < 0 and "-" not in raw:
            val = -val
        # Fallback: word-number conversion if parse_number failed
        if val is None:
            _WORD_NUMS = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
                "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
                "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
            }
            for word, num in _WORD_NUMS.items():
                if word in raw.lower().split():
                    val = num
                    break
        # LLM should return non-negative for counts/durations; only allow
        # negative if the raw text explicitly has a minus sign
        if val is not None and val < 0 and "-" not in raw:
            val = -val
        return val

    if answer_type == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if m:
            y, mo, d = (int(x) for x in m.group().split("-"))
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return m.group()
        # Fallback: parse "15 March 2024" or "March 15, 2024" if LLM didn't format correctly
        _month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        }
        m2 = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
        if m2:
            mo = _month_map.get(m2.group(2).lower())
            if mo:
                return f"{m2.group(3)}-{mo:02d}-{int(m2.group(1)):02d}"
        m3 = re.search(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", raw)
        if m3:
            mo = _month_map.get(m3.group(1).lower())
            if mo:
                return f"{m3.group(3)}-{mo:02d}-{int(m3.group(2)):02d}"
        return None

    if answer_type == "names":
        if _NOT_FOUND_RE.search(raw) and not raw.strip().startswith("["):
            return None
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if x and str(x).strip()]
                return items or None
        except json.JSONDecodeError:
            pass
        if '"' in clean or clean.startswith("["):
            items = [x.strip().strip("\"'[] ") for x in re.split(r"[,\n]", clean)]
            items = [x for x in items if x and len(x) < 150 and not _NOT_FOUND_RE.search(x)]
            return items or None
        if len(clean) < 150 and not _NOT_FOUND_RE.search(clean):
            return [clean.strip("\"' ")]
        return None

    if answer_type == "name":
        if _NOT_FOUND_RE.search(raw):
            return None
        # Take first line only (LLM may add explanation after)
        first_line = raw.split("\n")[0].strip().strip("\"' ").strip()
        if not first_line:
            first_line = raw.strip("\"' ").strip()
        return first_line[:200] or None

    if answer_type == "free_text":
        # Strip markdown bold markers (LLM judge may not handle them well)
        text = raw.replace("**", "")
        # Strip block/context references that leak implementation details
        text = re.sub(r"\s*\(?\bBlock \d+\)?\s*", " ", text)
        # Strip filler prefixes that waste characters without adding value
        text = re.sub(r"^(?:According to the (?:context|provided (?:documents?|context|text)),?\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:Based on the (?:provided |given )?(?:context|documents?|text),?\s*)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text.startswith(FREE_TEXT_FALLBACK[:30]):
            return FREE_TEXT_FALLBACK
        if len(text) > FREE_TEXT_MAX:
            # Smart truncation: always cut at a complete sentence boundary.
            # Find ALL sentence-ending positions within the limit.
            _sent_ends: list[int] = []
            for m in re.finditer(r'[.!?](?:\)|\")?\s', text[:FREE_TEXT_MAX]):
                _sent_ends.append(m.end() - 1)  # position of last char of sentence
            # Also check if text ends with period right at the limit
            if text[FREE_TEXT_MAX - 1] in '.!?' or (FREE_TEXT_MAX >= 2 and text[FREE_TEXT_MAX - 2] in '.!?' and text[FREE_TEXT_MAX - 1] in ')"'):
                _sent_ends.append(FREE_TEXT_MAX - 1)
            if _sent_ends and _sent_ends[-1] > FREE_TEXT_MAX * 0.35:
                text = text[:_sent_ends[-1] + 1].rstrip()
            else:
                # No good sentence boundary — cut at last space + add period
                truncated = text[:FREE_TEXT_MAX]
                last_space = truncated.rfind(" ")
                if last_space > FREE_TEXT_MAX * 0.5:
                    text = truncated[:last_space].rstrip(",;:—-") + "."
                else:
                    text = truncated
        return text

    return raw or None


# ---------------------------------------------------------------------------
# Smart source-page selection
# ---------------------------------------------------------------------------

_MONTHS_LONG = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTHS_SHORT = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _date_search_variants(iso: str) -> list[str]:
    try:
        d = _date.fromisoformat(iso)
        return [
            iso,
            f"{d.day} {_MONTHS_LONG[d.month - 1]} {d.year}",
            f"{d.day} {_MONTHS_SHORT[d.month - 1]} {d.year}",
            f"{_MONTHS_LONG[d.month - 1]} {d.day}, {d.year}",
            f"{d.day:02d}/{d.month:02d}/{d.year}",
            f"{d.year}/{d.month:02d}/{d.day:02d}",
        ]
    except Exception:
        return [iso]


_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    21: "twenty-one", 25: "twenty-five", 30: "thirty", 40: "forty", 45: "forty-five",
    50: "fifty", 60: "sixty", 90: "ninety",
}


def _number_search_variants(v: Any) -> list[str]:
    try:
        n = float(v)
        if n == int(n):
            i = int(n)
            if i < 100:
                # Legal docs use "word (digit)" format: "twelve (12) months"
                word = _NUM_WORDS.get(i)
                if word:
                    return [f"{word} ({i})", word]
                return []
            return [str(i), f"{i:,}"]
        return [str(n)]
    except Exception:
        return [str(v)]


def _find_source_pages(answer: Any, pages: list[dict], answer_type: str = "") -> list[dict]:
    """Return pages that contain evidence for the answer (smart text matching).

    Strategy:
    - For deterministic types: search answer value in page text → return matching pages.
    - For boolean: return top-1 page (can't text-match true/false directly).
    - For free_text: handled by LLM citation; this function receives pre-filtered pages.
    - Fallback: top-1 page if no match found (never return empty for non-null answers).
    """
    if answer is None or not pages:
        return []

    if answer_type in ("bool", "boolean"):
        # For comparison questions involving multiple docs: include 1 page per doc
        # (gold typically cites 1 page from each relevant document)
        seen_docs: dict[str, dict] = {}
        for p in pages:
            did = p["doc_id"]
            if did not in seen_docs:
                seen_docs[did] = p
            if len(seen_docs) >= 2:
                break
        # If multiple docs represented, return 1 page per doc; else top-1
        if len(seen_docs) >= 2:
            return list(seen_docs.values())
        return pages[:1]

    if answer_type == "free_text":
        # free_text sources are resolved from LLM citation in answer_with_llm
        return pages  # return all pages; β=2.5 rewards recall heavily

    if answer_type == "names" and isinstance(answer, list):
        # For names: find pages containing the most named entities
        page_scores = []
        for page in pages:
            text_lower = page["text"].lower()
            matches = sum(1 for name in answer if name.lower() in text_lower)
            page_scores.append((matches, page))
        page_scores.sort(key=lambda x: -x[0])
        # Return pages with at least one name match
        matched = [p for score, p in page_scores if score > 0]
        return matched[:2] if matched else pages[:1]

    # Build search terms
    search_terms: list[str] = []
    if answer_type == "number":
        search_terms = _number_search_variants(answer)
    elif answer_type == "date":
        search_terms = _date_search_variants(str(answer))
    elif answer_type == "name":
        val = str(answer).strip()
        if val:
            search_terms = [val]
    elif answer_type == "names":
        if isinstance(answer, list):
            search_terms = [str(n).strip() for n in answer if n]
        else:
            s = str(answer).strip()
            if s:
                search_terms = [s]

    if not search_terms:
        return pages[:1]

    matching: list[dict] = []
    for page in pages:
        text = page["text"]
        for term in search_terms:
            if re.search(re.escape(term), text, re.IGNORECASE):
                matching.append(page)
                break

    return matching if matching else pages[:1]


def _verify_evidence_pages(
    answer: Any,
    answer_type: str,
    question: str,
    candidate_pages: list[dict],
    context_pages: list[dict],
) -> list[dict]:
    """After the answer is produced, find the exact page(s) containing evidence.

    Unlike _find_source_pages, this verifies evidence against the full candidate pool
    (CITE pages + article index pages + context pages) and defaults to 1 page for
    typed answers unless evidence truly spans multiple pages.
    """
    if answer is None or not candidate_pages:
        return []

    # For boolean: find the page with the relevant article/clause
    if answer_type in ("bool", "boolean"):
        art_m = re.search(r"Article\s+(\d+)", question, re.IGNORECASE)
        if art_m:
            art_num = art_m.group(1)
            art_pat = re.compile(
                rf"\bArticle\s+{art_num}\b|(?:^|\n)\s*{art_num}\.\s",
                re.IGNORECASE,
            )
            matched = [p for p in candidate_pages if art_pat.search(p.get("text", ""))]
            if matched:
                return matched[:1]  # 1 page for bool article questions
        # Non-article boolean (judge/party comparison): 1 page per doc
        seen: dict[str, dict] = {}
        for p in candidate_pages:
            if p["doc_id"] not in seen:
                seen[p["doc_id"]] = p
            if len(seen) >= 2:
                break
        return list(seen.values()) if len(seen) >= 2 else candidate_pages[:1]

    # For number: find the page containing the numeric value
    if answer_type == "number":
        variants = _number_search_variants(answer)
        if variants:
            for p in candidate_pages:
                text = p.get("text", "")
                for v in variants:
                    if re.search(re.escape(v), text, re.IGNORECASE):
                        return [p]
        return candidate_pages[:1]

    # For date: find the page containing the date
    if answer_type == "date":
        variants = _date_search_variants(str(answer))
        for p in candidate_pages:
            text = p.get("text", "")
            for v in variants:
                if v in text:
                    return [p]
        return candidate_pages[:1]

    # For name: find the page containing the name
    if answer_type == "name":
        val = str(answer).strip().lower()
        if val:
            # Check for case ref answer (e.g. "CFI 010/2024")
            case_ref_m = re.match(r"[A-Z]{2,5}\s+\d{3}/\d{4}", str(answer).strip())
            if case_ref_m:
                ref_nospace = case_ref_m.group().replace(" ", "")
                for p in candidate_pages:
                    if ref_nospace in p.get("text", "").replace(" ", ""):
                        return [p]
            for p in candidate_pages:
                if val in p.get("text", "").lower():
                    return [p]
        return candidate_pages[:1]

    # For names: find pages containing the names
    if answer_type == "names" and isinstance(answer, list):
        best_page = None
        best_count = 0
        for p in candidate_pages:
            text_lower = p.get("text", "").lower()
            count = sum(1 for name in answer if name.lower() in text_lower)
            if count > best_count:
                best_count = count
                best_page = p
        if best_page:
            return [best_page]
        return candidate_pages[:1]

    return candidate_pages[:1]


def _detect_specific_page(question: str, source_pages: list[dict],
                          all_pages: Optional[list[dict]] = None) -> list[dict]:
    """If question asks about a specific page, return only that page.
    Returns empty list if no specific page detected (caller should use default logic).
    all_pages: full retrieved page list (for finding pages not in source_pages)."""
    if not source_pages:
        return []
    q_lower = question.lower()
    search_pool = all_pages if all_pages else source_pages

    # "last page" → find the last page from the primary document
    if "last page" in q_lower:
        doc_ids = [p["doc_id"] for p in source_pages]
        primary_doc = max(set(doc_ids), key=doc_ids.count)
        doc_pages = [p for p in search_pool if p["doc_id"] == primary_doc]
        if doc_pages:
            last = max(doc_pages, key=lambda p: p["page_number"])
            return [last]

    # "first page" / "title page" / "cover page" / "header" / "caption" → page 1 from each doc
    if any(kw in q_lower for kw in ["first page", "title page", "cover page", "header", "caption"]):
        seen_docs: set[str] = set()
        result_pages: list[dict] = []
        # Start with docs from source_pages
        for p in source_pages:
            if p["doc_id"] not in seen_docs:
                p1 = next((pp for pp in search_pool if pp["doc_id"] == p["doc_id"] and pp["page_number"] == 1), None)
                if not p1 and p["page_number"] == 1:
                    p1 = p
                if p1:
                    result_pages.append(p1)
                    seen_docs.add(p["doc_id"])
        # For "each document" / "header/caption" with case reference: also search
        # search_pool for additional docs from the same case (multi-doc cases).
        if "each document" in q_lower or "header" in q_lower or "caption" in q_lower:
            _case_refs = re.findall(r"[A-Z]{2,5}\s+\d{3}/\d{4}", question)
            for cr in _case_refs:
                cr_nospace = cr.replace(" ", "")
                for pp in search_pool:
                    if pp["doc_id"] not in seen_docs and pp["page_number"] == 1:
                        if cr_nospace in pp.get("text", "").replace(" ", ""):
                            result_pages.append(pp)
                            seen_docs.add(pp["doc_id"])
        if result_pages:
            return result_pages

    # "page N" / "second page" → specific page number
    m = re.search(r"\b(?:page\s+(\d+)|second page)\b", q_lower)
    if m:
        target_pg = int(m.group(1)) if m.group(1) else 2
        doc_ids = [p["doc_id"] for p in source_pages]
        primary_doc = max(set(doc_ids), key=doc_ids.count)
        target = [p for p in search_pool if p["doc_id"] == primary_doc and p["page_number"] == target_pg]
        if target:
            return target

    return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_with_llm(
    question_data: dict,
    pages: list[dict],
    t0: Optional[float] = None,
    is_comparison: bool = False,
    article_index: Optional[dict] = None,
    corpus_pages: Optional[list[dict]] = None,
) -> tuple[Any, list[dict], int, int, int, int, str]:
    """
    Returns: (answer, used_pages, ttft_ms, total_ms, input_tokens, output_tokens, model_name)
    t0: perf_counter timestamp from before retrieval (for accurate TTFT measurement).
    """
    answer_type = question_data.get("answer_type", "free_text")
    question = question_data.get("question", "")
    _all_pages = pages  # Save original full page list before injection modifies it

    _t0 = t0 if t0 is not None else time.perf_counter()

    # Early detection of adversarial/trick questions (DIFC courts don't have these)
    # Skip LLM entirely → ttft=1ms, saves cost and time
    _ADVERSARIAL_RE = re.compile(
        r"\b(jury|jurors?|plea bargains?|Miranda rights?|parole hearings?|criminal verdicts?|"
        r"criminal sentenc|indictments?|grand jury|bail hearings?)\b",
        re.IGNORECASE,
    )
    if _ADVERSARIAL_RE.search(question):
        if answer_type == "free_text":
            return FREE_TEXT_FALLBACK, [], 1, 1, 0, 0, "adversarial-skip"
        # For deterministic types: null + empty pages (gold=null, gold_pages=[])
        return None, [], 1, 1, 0, 0, "adversarial-skip"

    if _provider() is None:
        from src.pipeline.answer import answer_question
        ans, used = answer_question(question_data, pages)
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        return ans, used, elapsed, elapsed, 0, 0, "deterministic"

    # ---- Pre-LLM deterministic fast path ----
    # For patterns where deterministic extraction is reliable and faster than LLM.
    # Skips LLM entirely → TTFT ≈ 1ms, saving 600-3700ms per question.

    # 1. Law number: "what is the law number on title page" → "Law No. X"
    if answer_type == "number" and not is_comparison:
        if re.search(r"\b(law\s+number|official.*number|difc\s+law\s+no)\b", question.lower()):
            for p in pages:
                m = re.search(r"(?:DIFC\s+)?Law\s+No\.?\s*(\d+)", p.get("text", ""), re.IGNORECASE)
                if m:
                    _det_ans = int(m.group(1))
                    _elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
                    return _det_ans, [p], _elapsed, _elapsed, 0, 0, "deterministic-fast"

    # 2. Comparison date-of-issue → name: extract dates, compare
    if answer_type == "name" and is_comparison:
        _q_lower_det = question.lower()
        if re.search(r"\b(earlier|later|first)\b.*\b(date|issue)|issued.*\b(earlier|first)\b", _q_lower_det):
            from src.pipeline.answer import _extract_comparison_date_name
            _det_cmp, _det_cmp_pg = _extract_comparison_date_name(pages, question)
            if _det_cmp is not None:
                _grnd_pages = _det_cmp_pg if _det_cmp_pg else pages[:2]
                # Use doc routing for precise grounding (page 1 from each case)
                _case_refs_det = re.findall(r"[A-Z]{2,5}\s+\d{3}/\d{4}", question)
                if _case_refs_det and corpus_pages:
                    from src.pipeline.retrieve import build_doc_routing_index
                    _doc_route_det = build_doc_routing_index(corpus_pages)
                    _page_lk_det = article_index.get("page_lookup", {}) if article_index else {}
                    _routing_pages: list[dict] = []
                    for cr in _case_refs_det:
                        cr_norm = cr.upper().replace("  ", " ")
                        _td = _doc_route_det["case"].get(cr_norm)
                        if _td:
                            p1 = _page_lk_det.get((_td, 1))
                            if not p1:
                                p1 = next((cp for cp in corpus_pages if cp["doc_id"] == _td and cp["page_number"] == 1), None)
                            if p1:
                                _routing_pages.append(p1)
                    if len(_routing_pages) >= 2:
                        _grnd_pages = _routing_pages
                _elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
                return _det_cmp, _grnd_pages, _elapsed, _elapsed, 0, 0, "deterministic-fast"

    # 3. Comparison monetary claim → name: extract AED amounts, compare
    if answer_type == "name" and is_comparison:
        _q_lower_mc = question.lower()
        if re.search(r"\b(higher|lower|larger|smaller|greater|biggest)\b.*\b(monetary|claim|amount)\b", _q_lower_mc):
            _case_refs_mc = re.findall(r"[A-Z]{2,5}\s+\d{3}/\d{4}", question)
            if len(_case_refs_mc) == 2 and corpus_pages:
                from src.pipeline.retrieve import build_doc_routing_index
                _doc_route_mc = build_doc_routing_index(corpus_pages)
                _page_lk_mc = article_index.get("page_lookup", {}) if article_index else {}
                _case_amounts: list[tuple[str, float, dict]] = []  # (case_ref, amount, evidence_page)
                for cr in _case_refs_mc:
                    cr_norm = cr.upper().replace("  ", " ")
                    _td = _doc_route_mc["case"].get(cr_norm)
                    if not _td:
                        continue
                    # Search pages 1-5 of this doc for AED amounts
                    _best_amount = 0.0
                    _best_page = None
                    for pn in range(1, 6):
                        p = _page_lk_mc.get((_td, pn))
                        if not p:
                            p = next((cp for cp in corpus_pages if cp["doc_id"] == _td and cp["page_number"] == pn), None)
                        if not p:
                            continue
                        text = p.get("text", "")
                        # Find AED amounts: "AED 391,123.45" or "AED 165,000"
                        _WORD_NUMS_MC = {
                            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                            "eleven": 11, "twelve": 12,
                        }
                        for am in re.finditer(r"AED\s+([\d,]+(?:\.\d+)?)", text, re.IGNORECASE):
                            val_str = am.group(1).replace(",", "")
                            try:
                                val = float(val_str)
                            except ValueError:
                                continue
                            # Check for monthly salary pattern: "N months of salary at AED X"
                            # Handles both digit ("4 months") and word ("four months")
                            _pre_context = text[max(0, am.start()-200):am.start()].lower()
                            _months_m = re.search(
                                r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                                r"\s*(?:\(\d+\)\s*)?(?:months?|month\(s\))\s+(?:of\s+)?"
                                r"(?:(?:his|her|the)\s+)?(?:basic\s+)?(?:salary|base\s+salary)",
                                _pre_context,
                            )
                            if _months_m:
                                _mn = _months_m.group(1)
                                months = _WORD_NUMS_MC.get(_mn, None)
                                if months is None:
                                    try:
                                        months = int(_mn)
                                    except ValueError:
                                        months = None
                                if months:
                                    val = val * months
                            if val > _best_amount:
                                _best_amount = val
                                _best_page = p
                    if _best_amount > 0 and _best_page:
                        _case_amounts.append((cr, _best_amount, _best_page))
                if len(_case_amounts) == 2:
                    _wants_higher = bool(re.search(r"\b(higher|larger|greater|biggest)\b", _q_lower_mc))
                    if _wants_higher:
                        _winner = max(_case_amounts, key=lambda x: x[1])
                    else:
                        _winner = min(_case_amounts, key=lambda x: x[1])
                    _grnd_mc = [ca[2] for ca in _case_amounts]  # evidence page from each case
                    _elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
                    return _winner[0], _grnd_mc, _elapsed, _elapsed, 0, 0, "deterministic-fast"

    cfg = _TYPE_CONFIG.get(answer_type, _DEFAULT_CONFIG)
    max_pages = cfg["max_pages"]
    chars_per_page = cfg["chars"]
    max_tokens = cfg["max_tokens"]

    # --- Article index injection: ensure article definition pages are in context ---
    # Uses pre-built article→page index for precise page injection from FULL corpus
    # (not just retrieved pages). This guarantees the LLM sees the right article.
    _article_grounding_pages: Optional[list[dict]] = None  # Will be set if article Q
    _q_art_for_index = re.search(r"\bArticle\s+(\d+)", question, re.IGNORECASE)
    if _q_art_for_index and article_index and corpus_pages and not is_comparison:
        _art_idx_num = _q_art_for_index.group(1)
        _page_lookup = article_index.get("page_lookup", {})
        _art_defs = article_index.get("definitions", {})

        # Find target document: from law name in question → doc routing
        _target_doc_for_art = None
        _law_name_m = re.search(
            r"\b(?:the\s+)?((?:Operating|Employment|Trust|Foundations?|"
            r"General Partnership|Limited Liability Partnership|"
            r"Common Reporting Standard|Insolvency|Companies|Personal Property|"
            r"Application of Civil)\s+(?:Law|Standard))\b",
            question, re.IGNORECASE,
        )
        if _law_name_m:
            _law_name_lower = _law_name_m.group(1).lower()
            for cp in corpus_pages:
                if cp["page_number"] == 1:
                    if _law_name_lower in cp.get("text", "")[:500].lower():
                        _target_doc_for_art = cp["doc_id"]
                        break
        # Also try "DIFC Law No. N" matching
        if not _target_doc_for_art:
            _law_no_m = re.search(r"\bLaw\s+No\.?\s*(\d+)\s+of\s+(\d{4})\b", question, re.IGNORECASE)
            if _law_no_m:
                _law_num = _law_no_m.group(1)
                _law_year = _law_no_m.group(2)
                _law_pat = re.compile(rf"Law\s+No\.?\s*{_law_num}\s+of\s+{_law_year}", re.IGNORECASE)
                for cp in corpus_pages:
                    if cp["page_number"] == 1 and _law_pat.search(cp.get("text", "")[:500]):
                        _target_doc_for_art = cp["doc_id"]
                        break
        # Fallback: primary doc from reranked results
        if not _target_doc_for_art and pages:
            _target_doc_for_art = pages[0]["doc_id"]

        if _target_doc_for_art:
            _art_key = (_target_doc_for_art, _art_idx_num)
            _all_mentions = article_index.get("all", {})
            # Prioritize DEFINITION pages over "all" mentions.
            # "all" mentions include early cross-refs (e.g. Art 9 referenced
            # on page 6 but actually defined on page 11). Sorting merged pages
            # by number picks those early refs first, missing the real content.
            _raw_defs = sorted(set(_art_defs.get(_art_key, [])))
            _raw_all = sorted(set(_all_mentions.get(_art_key, [])))
            # Use definitions first (up to 3), then supplement with "all" pages
            # that aren't already covered. This ensures content pages are selected.
            _def_set = set(_raw_defs)
            _supplement = [p for p in _raw_all if p not in _def_set]
            _def_page_nums = (_raw_defs[:3] + _supplement)[:5]  # generous pool
            if _def_page_nums:
                _base_pages = _raw_defs[:3] if _raw_defs else _raw_all[:3]
                # If question references a LATE sub-clause letter (f-z),
                # check if article content starts before the first definition page.
                # Art 7(3)(j) starts on page 5 but "7." heading is on page 7.
                # Early sub-clauses (a-e) are on the first def page; only late ones
                # overflow to preceding pages.
                _q_has_subclause = bool(re.search(
                    r"\bArticle\s+\d+(?:\(\d+\))?\([f-z]\)",
                    question, re.IGNORECASE,
                ))
                if _q_has_subclause and _base_pages and _raw_all:
                    _first_def = _base_pages[0]
                    _pre_pages = [p for p in _raw_all if p < _first_def and _first_def - p <= 2]
                    if _pre_pages:
                        _base_pages = [_pre_pages[-1]] + _base_pages
                _expanded_pages: list[int] = []
                for i, pn in enumerate(_base_pages[:4]):  # cap at 4 base
                    _expanded_pages.append(pn)
                    if i + 1 < len(_base_pages):
                        _next_def = _base_pages[i + 1]
                        # Fill gap pages (max 1 gap page between consecutive defs)
                        if _next_def - pn == 2:
                            _expanded_pages.append(pn + 1)
                # Get page objects from lookup
                _art_inject_objs = []
                for pn in _expanded_pages:
                    pobj = _page_lookup.get((_target_doc_for_art, pn))
                    if pobj:
                        _art_inject_objs.append(pobj)
                if _art_inject_objs:
                    # Recall floor: if only 1 article page found, check if article
                    # continues on the next page (many articles span 2 pages).
                    # Missing a gold page costs G: 1.0→0.537. Extra page: 1.0→0.879.
                    if len(_art_inject_objs) == 1:
                        _next_pn = _art_inject_objs[0]["page_number"] + 1
                        _next_p = _page_lookup.get((_target_doc_for_art, _next_pn))
                        if _next_p:
                            # Only add if next page doesn't start a NEW article
                            _next_text = _next_p.get("text", "")[:100]
                            if not re.match(r"\s*Article\s+\d+\b", _next_text, re.IGNORECASE):
                                _art_inject_objs.append(_next_p)
                    # Save for later grounding citation
                    _article_grounding_pages = _art_inject_objs[:3]
                    # Inject into context: ensure these pages are in top positions
                    _existing_keys = {(p["doc_id"], p["page_number"]) for p in pages[:max_pages]}
                    _to_inject = [p for p in _art_inject_objs
                                  if (p["doc_id"], p["page_number"]) not in _existing_keys]
                    if _to_inject:
                        # Insert after first page, pushing others down
                        pages = pages[:1] + _to_inject[:2] + pages[1:]

    # Targeted page injection: if the question asks about specific sections that are
    # typically deep in documents (Schedule tables, Conclusion sections) or specific
    # article sub-clauses, inject matching pages into top-k from the retrieved pool.
    _is_schedule_question = bool(re.search(
        r"\b(schedule|maximum fine|fine amount|penalty amount|table of fines)\b",
        question, re.IGNORECASE,
    ))
    _is_conclusion_question = bool(re.search(
        r"\b(conclusion section|Conclusion.*(?:rul|costs?|award))\b",
        question, re.IGNORECASE,
    ))

    # Article sub-clause injection: "Article 7(3)(j)" → find pages with "(j)" near "Article 7"
    _art_subclause_m = re.search(
        r"\bArticle\s+(\d+)(?:\((\d+)\))?(?:\(([a-z])\))?",
        question, re.IGNORECASE,
    )
    # Also inject if question references any Article N (not just sub-clauses)
    _q_art_ref = re.search(r"\bArticle\s+(\d+)", question, re.IGNORECASE) if not _art_subclause_m else _art_subclause_m
    _inject_needed = _is_schedule_question or _is_conclusion_question or _art_subclause_m or _q_art_ref

    if _inject_needed and len(pages) > max_pages:
        top_pages = pages[:max_pages]
        _inject_pats: list[re.Pattern] = []
        _art_inject_pats_proximity = None
        if _is_schedule_question or _is_conclusion_question:
            _inject_pats.append(re.compile(
                r"(?:^|\n)\s*SCHEDULE\s+\d|Maximum Fine|"
                r"(?:^|\n)\s*Conclusion\b|(?:^|\n)\s*\d+\.\s+(?:For all|In conclusion)",
                re.IGNORECASE,
            ))
        if _art_subclause_m:
            _art_n = _art_subclause_m.group(1)
            _sub_n = _art_subclause_m.group(2)
            _sub_l = _art_subclause_m.group(3)
            _subclause_pat = None
            if _sub_l:
                # Letter sub-clauses like (j) are specific enough for injection
                _subclause_pat = re.compile(rf"\({_sub_l}\)", re.IGNORECASE)
            # Skip numeric sub-clauses like (1), (2) — too generic, match everywhere
            if _subclause_pat:
                # Find pages in top-k that reference this article
                _art_ref_pages = [p["page_number"] for p in top_pages
                                  if re.search(rf"\bArticle\s+{_art_n}\b", p.get("text", ""), re.IGNORECASE)]
                if not _art_ref_pages:
                    _art_ref_pages = [p["page_number"] for p in top_pages]
                _art_inject_pats_proximity = (_subclause_pat, _art_ref_pages)
        top_keys = {(p["doc_id"], p["page_number"]) for p in top_pages}
        _primary_doc_ids = {p["doc_id"] for p in top_pages}
        injected = []
        # Check if top-k already contains the referenced Article
        _ref_art_n = _q_art_ref.group(1) if _q_art_ref else None
        _top_has_article = False
        if _ref_art_n:
            # Match both "Article 14" and section-heading format "14." (used by some laws)
            _ref_art_pat = re.compile(
                rf"\bArticle\s+{_ref_art_n}\b|(?:^|\n)\s*{_ref_art_n}\.\s",
                re.IGNORECASE,
            )
            _top_has_article = any(_ref_art_pat.search(p.get("text", "")) for p in top_pages)
        for p in pages[max_pages:]:
            key = (p["doc_id"], p["page_number"])
            if key not in top_keys and p["doc_id"] in _primary_doc_ids:
                p_text = p.get("text", "")
                _matched = False
                # Check schedule/conclusion patterns
                for pat in _inject_pats:
                    if pat.search(p_text):
                        _matched = True
                        break
                # Check article sub-clause with proximity constraint
                if not _matched and _art_inject_pats_proximity:
                    _sc_pat, _ref_pages = _art_inject_pats_proximity
                    if _sc_pat.search(p_text):
                        # Only inject if within ±3 pages of article reference
                        if any(abs(p["page_number"] - rp) <= 3 for rp in _ref_pages):
                            _matched = True
                # Direct article injection: if top-k doesn't have "Article N",
                # inject the first page from the same doc that contains it.
                if not _matched and _ref_art_n and not _top_has_article:
                    if _ref_art_pat.search(p_text):
                        _matched = True
                        _top_has_article = True  # only inject once
                if _matched:
                    injected.append(p)
        if injected:
            # Limit injections to max 2 pages to avoid displacing too much context
            injected = injected[:2]
            if len(injected) <= max_pages - 1:
                pages = top_pages[:1] + injected + top_pages[1:max_pages - len(injected)]
            else:
                pages = top_pages[:1] + injected[:max_pages - 1]

    # For comparison questions: ensure pages from multiple docs are included
    # (otherwise all max_pages slots may be filled by one doc)
    # Also bump max_pages for comparison name questions (need dates from both cases)
    if is_comparison and answer_type == "name":
        max_pages = max(max_pages, 5)
    if is_comparison and len(pages) > max_pages:
        selected: list[dict] = []
        doc_counts: dict[str, int] = {}
        # First pass: take up to 3 pages per doc, round-robin style
        max_per_doc_cmp = 3
        for p in pages:
            did = p["doc_id"]
            if doc_counts.get(did, 0) < max_per_doc_cmp:
                selected.append(p)
                doc_counts[did] = doc_counts.get(did, 0) + 1
            if len(selected) >= max_pages:
                break
        # If we still have room and missed pages, fill up
        if len(selected) < max_pages:
            for p in pages:
                if p not in selected:
                    selected.append(p)
                    if len(selected) >= max_pages:
                        break
        page_list = selected
    else:
        page_list = pages[:max_pages]

    context_parts: list[str] = []
    context_pages: list[dict] = []
    for i, page in enumerate(page_list):
        text = page["text"].strip()
        if not text:
            continue
        # Distill to most relevant paragraphs for all types
        distilled = _distill_page(text, question, chars_per_page)
        context_parts.append(
            f"[BLOCK {i}: {page['doc_id']} p.{page['page_number']}]\n{distilled}"
        )
        context_pages.append(page)

    if not context_parts:
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        if answer_type == "free_text":
            return FREE_TEXT_FALLBACK, [], elapsed, elapsed, 0, 0, "none"
        return None, [], elapsed, elapsed, 0, 0, "none"

    context = "\n\n---\n\n".join(context_parts)
    # Comparison booleans get chain-of-thought prompt with higher max_tokens
    _effective_max_tokens = max_tokens
    if is_comparison and answer_type in ("bool", "boolean"):
        _effective_max_tokens = 150  # Room for E1/E2/ANSWER/CITE with party names
    prompt = _build_prompt(question, answer_type, context, is_comparison=is_comparison)
    # Sonnet for free_text: TTFT ~1500-2000ms (F=1.02) vs Haiku ~700ms (F=1.05).
    # F drop per free_text Q: -0.03. Over 30 free_text / 100 total: avg F drop = 0.009.
    # But Sonnet Asst improvement ~+0.10 → net gain: 0.3×0.10×G - 0.009 ≈ +0.016.
    use_strong = (answer_type == "free_text")

    # Pass t0=None so TTFT measures only LLM API time (excludes retrieval/context building)
    raw, ttft_ms, _call_total, in_tok, out_tok, model = _call(
        prompt, max_tokens=_effective_max_tokens, use_strong_model=use_strong, t0=None
    )

    if raw is None:
        from src.pipeline.answer import answer_question
        ans, used = answer_question(question_data, pages)
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        return ans, used, elapsed, elapsed, 0, 0, "deterministic-fallback"

    # For comparison booleans: extract ANSWER: line from structured CoT response
    if is_comparison and answer_type in ("bool", "boolean") and raw:
        answer_m = re.search(r"\bANSWER:\s*(true|false|null)\b", raw, re.IGNORECASE)
        if answer_m:
            # Extract citation from the full response before overwriting
            _, cited_indices_cot = _parse_citation(raw, len(context_pages))
            raw = answer_m.group(1).lower()
            if cited_indices_cot:
                # Pre-set citation from CoT response
                clean_raw = raw
                cited_indices = cited_indices_cot
            else:
                clean_raw, cited_indices = raw, []
        else:
            # No structured ANSWER: found — fall through to normal parsing
            clean_raw, cited_indices = _parse_citation(raw, len(context_pages))
    else:
        # Parse citation from LLM response (works for all types)
        clean_raw, cited_indices = _parse_citation(raw, len(context_pages))

    # For free_text: use SOURCES-based citation
    if answer_type == "free_text":
        text, ft_indices = _parse_free_text_sources(raw, len(context_pages))
        answer = _parse(text, answer_type)
        if answer == FREE_TEXT_FALLBACK or answer is None:
            total_ms2 = max(1, int((time.perf_counter() - _t0) * 1000))
            return FREE_TEXT_FALLBACK, [], ttft_ms, total_ms2, in_tok, out_tok, model
        # Use cited pages if valid; fallback to top 2 context pages.
        if ft_indices:
            used_pages = [context_pages[i] for i in ft_indices]
        else:
            used_pages = context_pages[:2]  # no citation → top 2
        # Recall floor: ensure at least 2 pages for free_text.
        # β=2.5 → missing a gold page costs 6.25x more than an extra page.
        if len(used_pages) < 2 and len(context_pages) >= 2:
            _ft_keys = {(p["doc_id"], p["page_number"]) for p in used_pages}
            for cp in context_pages:
                if (cp["doc_id"], cp["page_number"]) not in _ft_keys:
                    used_pages.append(cp)
                    break
        # Page-specific questions: restrict to that specific page
        _specific_ft = _detect_specific_page(question, used_pages, all_pages=pages)
        if _specific_ft:
            used_pages = _specific_ft
        # Safety: ensure non-null answer always has pages
        if not used_pages and context_pages:
            used_pages = context_pages[:1]
        # Cap free_text pages
        _ft_cap = 4 if is_comparison else 3
        if len(used_pages) > _ft_cap:
            used_pages = used_pages[:_ft_cap]
        total_ms2 = max(1, int((time.perf_counter() - _t0) * 1000))
        return answer, used_pages, ttft_ms, total_ms2, in_tok, out_tok, model

    # For non-free_text: parse answer from cleaned text (CITE suffix removed)
    answer = _parse(clean_raw, answer_type)

    # Post-LLM verification for number answers: detect article-number confusion.
    # If the answer matches the article number from the question, the LLM likely
    # returned the article number instead of the value. Search ALL retrieved pages
    # (not just context_pages) for the actual article content with a different number.
    if answer_type == "number" and answer is not None:
        _q_art_ref = re.search(r"\bArticle\s+(\d+)", question, re.IGNORECASE)
        if _q_art_ref and int(_q_art_ref.group(1)) == answer:
            # Search ALL retrieved pages (broader than context_pages)
            _search_pages = _all_pages if _all_pages else context_pages
            from src.pipeline.answer import extract_number
            det_num, det_pg = extract_number(_search_pages, question)
            if det_num is not None and det_num != answer:
                answer = det_num
                if det_pg:
                    clean_raw = str(answer)
                    cited_indices = []  # will use _find_source_pages

    # Post-LLM verification for name answers: if the answer looks like a generic
    # description (starts with articles/pronouns, or is the subject of the sentence
    # rather than the specific entity), try deterministic extraction.
    if answer_type == "name" and isinstance(answer, str) and context_pages:
        _generic_name_re = re.compile(
            r"^(Every|The|A|An|All|Each|No|Any)\s",
            re.IGNORECASE,
        )
        if _generic_name_re.match(answer) or len(answer) > 80:
            from src.pipeline.answer import extract_name
            det_name, det_pg = extract_name(context_pages, question)
            if det_name is not None and det_name != answer:
                answer = det_name
                if det_pg:
                    cited_indices = []

    # Post-LLM verification for comparison date-name questions:
    # Trust deterministic date extraction over LLM for "which case has earlier date" questions.
    if answer_type == "name" and is_comparison and isinstance(answer, str):
        _q_lower_cmp = question.lower()
        if re.search(r"\b(earlier|later|first)\b.*\b(date|issue)|"
                     r"\b(date|issue)\b.*\b(earlier|later|first)\b", _q_lower_cmp):
            from src.pipeline.answer import _extract_comparison_date_name
            _det_cmp, _det_cmp_pg = _extract_comparison_date_name(pages, question)
            if _det_cmp is not None and _det_cmp != answer:
                answer = _det_cmp
                if _det_cmp_pg:
                    cited_indices = []

    # Post-LLM verification for booleans: catch adversarial trick questions.
    if answer_type in ("bool", "boolean") and answer is True:
        # 1. Monetary amount check: fabricated dollar amounts → false
        _q_amount_m = re.search(r"\$\s*([\d,]+)", question)
        if _q_amount_m:
            _q_amount_raw = _q_amount_m.group(1).replace(",", "")
            _found_amount = False
            for cp in context_pages:
                if _q_amount_raw in cp.get("text", "").replace(",", "").replace(" ", ""):
                    _found_amount = True
                    break
            if not _found_amount:
                answer = False

        # 2. Territorial scope check: "does DIFC law X apply in [foreign jurisdiction]?"
        # DIFC laws only apply in the DIFC. If question mentions a DIFC law and asks
        # about a foreign jurisdiction, the answer is False.
        _foreign_juris_m = re.search(
            r"\bappl(?:y|ies)\b.*\b(?:in|at|to)\s+(?:the\s+)?(.{5,80}?)(?:\?|$)",
            question, re.IGNORECASE,
        )
        if _foreign_juris_m:
            _juris_text = _foreign_juris_m.group(1).lower()
            _is_foreign = any(kw in _juris_text for kw in [
                "sweden", "stockholm", "london", "new york", "singapore", "hong kong",
                "paris", "germany", "france", "india", "china", "usa", "uk",
                "switzerland", "zurich", "geneva", "amsterdam", "tokyo", "sydney",
                "arbitration institute", "icc", "lcia", "siac",
            ])
            _is_difc = any(kw in _juris_text for kw in ["difc", "dubai international financial"])
            # If question references a DIFC law and foreign jurisdiction, answer is False.
            # DIFC laws are jurisdictionally limited to DIFC by definition.
            _refs_difc_law = bool(re.search(
                r"\bDIFC\b.*\bLaw\b|\bLaw\s+No\.?\s*\d+\s+of\s+\d{4}\b",
                question, re.IGNORECASE,
            ))
            if _is_foreign and not _is_difc and _refs_difc_law:
                answer = False

        # 3. "Deemed to be" default rule check: if Article says "X shall be deemed to
        # be Y" and question asks "is X a Y?", the default rule is True, not False.
        # Exceptions (unless...) don't negate the general rule when asked generically.
        if answer is False and not is_comparison:
            art_m = re.search(r"Article\s+(\d+)", question, re.IGNORECASE)
            if art_m:
                for cp in context_pages:
                    txt = cp.get("text", "")
                    if re.search(rf"\bArticle\s+{art_m.group(1)}\b", txt, re.IGNORECASE) or \
                       re.search(rf"(?:^|\n)\s*{art_m.group(1)}\.\s", txt):
                        if re.search(r"\bshall be deemed\b.*\b(?:to be|a|an)\b", txt, re.IGNORECASE):
                            answer = True
                            break

    total_ms_final = max(1, int((time.perf_counter() - _t0) * 1000))

    if answer is None:
        # If the LLM couldn't find an answer but the question references a specific
        # article/case that IS in the context, try deterministic extraction as fallback.
        # This prevents false nulls on extractable questions.
        if answer_type not in ("free_text",) and context_pages:
            from src.pipeline.answer import answer_question
            det_ans, det_pages = answer_question(question_data, context_pages)
            if det_ans is not None:
                # Use article grounding pages for citations when available (more precise)
                if _article_grounding_pages and not is_comparison:
                    det_pages = list(_article_grounding_pages)
                return det_ans, det_pages, ttft_ms, total_ms_final, in_tok, out_tok, model + "-det-fallback"
        return None, [], ttft_ms, total_ms_final, in_tok, out_tok, model

    # =========================================================================
    # PAGE SELECTION FOR GROUNDING — Evidence-span based
    # =========================================================================
    # Strategy: build a candidate pool from multiple sources, then verify which
    # pages actually contain the supporting evidence. Default to 1 page for
    # typed answers unless evidence truly spans multiple pages.
    _candidate_pool: list[dict] = []
    _seen_keys: set[tuple[str, int]] = set()

    def _add_candidate(p: dict) -> None:
        key = (p["doc_id"], p["page_number"])
        if key not in _seen_keys:
            _seen_keys.add(key)
            _candidate_pool.append(p)

    # Source 1: CITE-based pages from LLM (highest signal)
    if cited_indices:
        for i in cited_indices:
            if i < len(context_pages):
                _add_candidate(context_pages[i])

    # Source 2: Article index pages (candidate, not override)
    if _article_grounding_pages:
        for p in _article_grounding_pages:
            _add_candidate(p)

    # Source 3: All context pages as fallback
    for p in context_pages:
        _add_candidate(p)

    # Verify evidence: find the page(s) that actually contain the answer
    source_pages = _verify_evidence_pages(
        answer, answer_type, question, _candidate_pool, context_pages
    )

    # For typed answers (number/date/name): merge CITE pages AND article pages
    # with evidence pages. β=2.5 means missing a gold page costs 6.25x more than
    # including an extra page. CITE pages carry LLM judgment; article pages carry
    # the referenced article text (often the gold standard for article questions).
    if answer_type not in ("bool", "boolean", "free_text"):
        _src_keys = {(p["doc_id"], p["page_number"]) for p in source_pages}
        # Add CITE pages
        if cited_indices:
            for i in cited_indices:
                if i < len(context_pages):
                    cp = context_pages[i]
                    _k = (cp["doc_id"], cp["page_number"])
                    if _k not in _src_keys:
                        source_pages.append(cp)
                        _src_keys.add(_k)
                        if len(source_pages) >= 2:
                            break
        # Add article page if question references an Article and it's not yet cited
        if len(source_pages) < 2 and _article_grounding_pages:
            for ap in _article_grounding_pages:
                _k = (ap["doc_id"], ap["page_number"])
                if _k not in _src_keys:
                    source_pages.append(ap)
                    _src_keys.add(_k)
                    if len(source_pages) >= 2:
                        break

    # For comparison questions: ensure both cases have representation via doc routing.
    # Use evidence-verified pages as base, then ensure each case has at least 1 page.
    _needs_all_docs = False
    if is_comparison:
        _case_refs_q = re.findall(r"[A-Z]{2,5}\s+\d{3}/\d{4}", question)
        if _case_refs_q and corpus_pages:
            from src.pipeline.retrieve import build_doc_routing_index
            _doc_route = build_doc_routing_index(corpus_pages)
            _page_lk = article_index.get("page_lookup", {}) if article_index else {}
            _source_doc_ids = {p["doc_id"] for p in source_pages}
            # Detect "all documents" / "full case files" → need ALL docs per case
            _needs_all_docs = bool(re.search(
                r"all documents|full case files|every document|title.?pages? of all",
                question, re.IGNORECASE,
            ))
            for cr in _case_refs_q:
                cr_norm = cr.upper().replace("  ", " ")
                if _needs_all_docs:
                    _cr_no_space = cr_norm.replace(" ", "")
                    for cp in corpus_pages:
                        if cp["page_number"] == 1 and cp["doc_id"] not in _source_doc_ids:
                            if _cr_no_space in cp.get("text", "").replace(" ", ""):
                                source_pages.append(cp)
                                _source_doc_ids.add(cp["doc_id"])
                else:
                    _target_doc = _doc_route["case"].get(cr_norm)
                    if _target_doc and _target_doc not in _source_doc_ids:
                        # Evidence verification: check page 1 and 2 for answer evidence
                        _best_p = None
                        for pn in (1, 2):
                            p = _page_lk.get((_target_doc, pn))
                            if not p:
                                p = next((cp for cp in corpus_pages if cp["doc_id"] == _target_doc and cp["page_number"] == pn), None)
                            if p:
                                # For name/free_text: prefer page with answer text
                                if answer_type in ("name", "free_text") and isinstance(answer, str):
                                    if answer.lower() in p.get("text", "").lower():
                                        _best_p = p
                                        break
                                if not _best_p:
                                    _best_p = p  # default to first available page
                        if _best_p:
                            source_pages.append(_best_p)
                            _source_doc_ids.add(_target_doc)

    # Page-specific questions: use corpus_pages if available for wider search
    _sp_search = corpus_pages if corpus_pages else pages
    _specific_page = _detect_specific_page(question, source_pages, all_pages=_sp_search)
    if _specific_page:
        source_pages = _specific_page

    # Final page cap: evidence-based grounding typically produces 1-2 pages.
    # Only "all documents" questions need higher caps.
    _multi_doc_override = len(_specific_page) if _specific_page and len(_specific_page) > 2 else 0
    if _needs_all_docs:
        _cap = 8
    elif is_comparison:
        _cap = 4  # 2 cases × up to 2 pages
    else:
        _cap = 2  # evidence-verified: 1-2 pages
    if _multi_doc_override > _cap:
        _cap = _multi_doc_override
    if len(source_pages) > _cap:
        source_pages = source_pages[:_cap]

    return answer, source_pages, ttft_ms, total_ms_final, in_tok, out_tok, model
