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
_TYPE_CONFIG = {
    "bool":     {"max_pages": 5, "chars": 3000, "max_tokens": 20},
    "boolean":  {"max_pages": 5, "chars": 3000, "max_tokens": 20},
    "number":   {"max_pages": 3, "chars": 3000, "max_tokens": 20},
    "date":     {"max_pages": 3, "chars": 2000, "max_tokens": 20},
    "name":     {"max_pages": 4, "chars": 2500, "max_tokens": 50},
    "names":    {"max_pages": 4, "chars": 2500, "max_tokens": 150},
    "free_text":{"max_pages": 4, "chars": 3000, "max_tokens": 300},
}
_DEFAULT_CONFIG = {"max_pages": 3, "chars": 1200, "max_tokens": 256}

_NOT_FOUND_RE = re.compile(
    r"(does not (identify|contain|mention|provide|include|specify)|"
    r"no (information|mention|reference|record)|"
    r"cannot (be|find|determine)|"
    r"not (found|available|present|identified|mentioned|specified|stated))",
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


def _provider() -> Optional[str]:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


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
    """
    _t0 = t0 if t0 is not None else time.perf_counter()
    p = _provider()

    try:
        if p == "anthropic":
            import anthropic
            model = "claude-sonnet-4-6" if use_strong_model else "claude-haiku-4-5-20251001"
            client = anthropic.Anthropic()
            ttft_ms: Optional[int] = None
            chunks: list[str] = []
            input_tokens = 0
            output_tokens = 0

            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
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
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
            return text, total_ms, total_ms, 0, 0, model

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
        print(f"  LLM error ({p}): {e}")

    total_ms = max(1, int((time.perf_counter() - _t0) * 1000))
    return None, total_ms, total_ms, 0, 0, "unknown"


_TYPE_INSTRUCTIONS = {
    "number": (
        "Return ONLY a single numeric value (integer or decimal). "
        "No units, no currency symbols, no extra text. "
        "Examples: 1500000 or 3.5 or 42. "
        "If not found in context: null"
    ),
    "bool": (
        "Return ONLY the word true, false, or null (lowercase, nothing else). "
        "true = yes / granted / approved / allowed / upheld / same. "
        "false = no / dismissed / denied / rejected / refused / different. "
        "null = the context does not clearly and directly answer this yes/no question. "
        "IMPORTANT: Only return true or false if you find CLEAR, DIRECT evidence. "
        "When in doubt or if the topic is absent, return null."
    ),
    "boolean": (
        "Return ONLY the word true, false, or null (lowercase, nothing else). "
        "true = yes / granted / approved / allowed / upheld / same. "
        "false = no / dismissed / denied / rejected / refused / different. "
        "null = the context does not clearly and directly answer this yes/no question. "
        "IMPORTANT: Only return true or false if you find CLEAR, DIRECT evidence. "
        "When in doubt or if the topic is absent, return null."
    ),
    "date": (
        "Return ONLY a date in YYYY-MM-DD format. "
        "Convert any date format (e.g. '15 March 2024' → '2024-03-15'). "
        "If not found in context: null"
    ),
    "name": (
        "Return ONLY the name or entity as a short phrase. "
        "Include the full legal name if available. "
        "No explanation, no extra text. "
        "If not found in context: null"
    ),
    "names": (
        "Return ONLY a JSON array of strings. "
        'Example: ["John Smith", "Acme Corp Ltd"]. '
        "No markdown code blocks. No explanations. "
        "Include every person, company or entity that answers the question. "
        "If genuinely not found: null"
    ),
    "free_text": (
        f"Answer in 1-3 clear, precise sentences using ONLY the provided context. "
        f"Maximum {FREE_TEXT_MAX} characters. Do not hallucinate or speculate. "
        "Be specific: include relevant legal names, article numbers, amounts, dates. "
        "After your answer write exactly: SOURCES: then comma-separated 0-based "
        "block numbers you used (e.g. SOURCES: 0,2). "
        f"If the answer is absent from context, write only: {FREE_TEXT_FALLBACK}"
    ),
}


def _build_prompt(question: str, answer_type: str, context: str) -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        "Extract the answer from the following legal document context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Instruction: {instruction}\n\n"
        "Answer:"
    )


def _parse_free_text_sources(raw: str, num_pages: int) -> tuple[str, list[int]]:
    """Parse 'SOURCES: 0,1,2' suffix from free_text response. Returns (answer, page_indices)."""
    m = re.search(r"\bSOURCES?:\s*([\d,\s]+)\s*$", raw.strip(), re.IGNORECASE)
    indices: list[int] = []
    if m:
        idx_strs = re.findall(r"\d+", m.group(1))
        indices = [int(i) for i in idx_strs if int(i) < num_pages]
        raw = raw[: m.start()].strip()
    return raw[:FREE_TEXT_MAX], indices


def _parse(raw: str, answer_type: str) -> Any:
    raw = raw.strip()
    if not raw or raw.lower() in ("null", "none", "n/a"):
        return None

    if answer_type in ("bool", "boolean"):
        low = raw.lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
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
        return parse_number(raw)

    if answer_type == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if m:
            y, mo, d = (int(x) for x in m.group().split("-"))
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return m.group()
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
        result = raw.strip("\"' ").strip()[:200]
        return result or None

    if answer_type == "free_text":
        text = raw[:FREE_TEXT_MAX]
        if text.startswith(FREE_TEXT_FALLBACK[:30]):
            return FREE_TEXT_FALLBACK
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


def _number_search_variants(v: Any) -> list[str]:
    try:
        n = float(v)
        if n == int(n):
            i = int(n)
            # Small numbers (< 100) appear too frequently; rely on top-page fallback
            if i < 100:
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
        return pages[:2]

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_with_llm(
    question_data: dict,
    pages: list[dict],
    t0: Optional[float] = None,
) -> tuple[Any, list[dict], int, int, int, int, str]:
    """
    Returns: (answer, used_pages, ttft_ms, total_ms, input_tokens, output_tokens, model_name)
    t0: perf_counter timestamp from before retrieval (for accurate TTFT measurement).
    """
    answer_type = question_data.get("answer_type", "free_text")
    question = question_data.get("question", "")

    _t0 = t0 if t0 is not None else time.perf_counter()

    if _provider() is None:
        from src.pipeline.answer import answer_question
        ans, used = answer_question(question_data, pages)
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        return ans, used, elapsed, elapsed, 0, 0, "deterministic"

    cfg = _TYPE_CONFIG.get(answer_type, _DEFAULT_CONFIG)
    max_pages = cfg["max_pages"]
    chars_per_page = cfg["chars"]
    max_tokens = cfg["max_tokens"]

    context_parts: list[str] = []
    context_pages: list[dict] = []
    for i, page in enumerate(pages[:max_pages]):
        text = page["text"].strip()
        if not text:
            continue
        context_parts.append(
            f"[BLOCK {i}: {page['doc_id']} p.{page['page_number']}]\n{text[:chars_per_page]}"
        )
        context_pages.append(page)

    if not context_parts:
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        if answer_type == "free_text":
            return FREE_TEXT_FALLBACK, [], elapsed, elapsed, 0, 0, "none"
        return None, [], elapsed, elapsed, 0, 0, "none"

    context = "\n\n---\n\n".join(context_parts)
    prompt = _build_prompt(question, answer_type, context)
    # Use Haiku for all types (fast TTFT → F=1.05 bonus)
    # Sonnet is ~3× slower to first token; Haiku is sufficient for structured extraction
    use_strong = False

    raw, ttft_ms, total_ms, in_tok, out_tok, model = _call(
        prompt, max_tokens=max_tokens, use_strong_model=use_strong, t0=_t0
    )

    if raw is None:
        from src.pipeline.answer import answer_question
        ans, used = answer_question(question_data, pages)
        elapsed = max(1, int((time.perf_counter() - _t0) * 1000))
        return ans, used, elapsed, elapsed, 0, 0, "deterministic-fallback"

    # For free_text: parse LLM-cited source blocks
    if answer_type == "free_text":
        text, cited_indices = _parse_free_text_sources(raw, len(context_pages))
        answer = _parse(text, answer_type)
        if answer == FREE_TEXT_FALLBACK or answer is None:
            total_ms2 = max(1, int((time.perf_counter() - _t0) * 1000))
            return FREE_TEXT_FALLBACK if answer_type == "free_text" else None, [], ttft_ms, total_ms2, in_tok, out_tok, model
        # Use cited pages if valid; fallback to top-2
        if cited_indices:
            used_pages = [context_pages[i] for i in cited_indices]
        else:
            used_pages = context_pages[:2]
        total_ms2 = max(1, int((time.perf_counter() - _t0) * 1000))
        return answer, used_pages, ttft_ms, total_ms2, in_tok, out_tok, model

    answer = _parse(raw, answer_type)

    # Retry with more pages if first attempt returns null
    if answer is None and answer_type in ("number", "date", "names") and len(pages) > max_pages:
        retry_pages = min(max_pages + 3, len(pages))
        retry_parts: list[str] = []
        retry_context_pages: list[dict] = []
        for i, page in enumerate(pages[:retry_pages]):
            text = page["text"].strip()
            if not text:
                continue
            retry_parts.append(
                f"[BLOCK {i}: {page['doc_id']} p.{page['page_number']}]\n{text[:chars_per_page]}"
            )
            retry_context_pages.append(page)
        if len(retry_parts) > len(context_parts):
            retry_context = "\n\n---\n\n".join(retry_parts)
            retry_prompt = _build_prompt(question, answer_type, retry_context)
            raw2, ttft_ms2, total_ms2, in_tok2, out_tok2, model2 = _call(
                retry_prompt, max_tokens=max_tokens, t0=_t0
            )
            if raw2 is not None:
                answer2 = _parse(raw2, answer_type)
                if answer2 is not None:
                    answer = answer2
                    context_pages = retry_context_pages
                    in_tok += in_tok2
                    out_tok += out_tok2

    total_ms_final = max(1, int((time.perf_counter() - _t0) * 1000))

    if answer is None:
        return None, [], ttft_ms, total_ms_final, in_tok, out_tok, model

    source_pages = _find_source_pages(answer, context_pages, answer_type)
    return answer, source_pages, ttft_ms, total_ms_final, in_tok, out_tok, model
