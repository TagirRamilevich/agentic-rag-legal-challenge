import json
import os
import re
from typing import Any, Optional

from src.utils.number_parse import parse_number

FREE_TEXT_FALLBACK = "There is no information on this question in the provided documents."
FREE_TEXT_MAX = 280

# Context size per answer type
_TYPE_CONFIG = {
    "bool":     {"max_pages": 5, "chars": 1500, "max_tokens": 20},
    "boolean":  {"max_pages": 5, "chars": 1500, "max_tokens": 20},
    "number":   {"max_pages": 3, "chars": 1500, "max_tokens": 20},
    "date":     {"max_pages": 3, "chars": 1200, "max_tokens": 20},
    "name":     {"max_pages": 4, "chars": 1200, "max_tokens": 50},
    "names":    {"max_pages": 4, "chars": 1200, "max_tokens": 150},
    "free_text":{"max_pages": 3, "chars": 1500, "max_tokens": 280},
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


def _call(prompt: str, max_tokens: int = 256, use_strong_model: bool = False) -> Optional[str]:
    p = _provider()
    try:
        if p == "anthropic":
            import anthropic
            model = "claude-sonnet-4-6" if use_strong_model else "claude-haiku-4-5-20251001"
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()

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
            return r.json()["choices"][0]["message"]["content"].strip()

        if p == "openai":
            import openai
            model = "gpt-4o" if use_strong_model else "gpt-4o-mini"
            client = openai.OpenAI()
            r = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()

    except Exception as e:
        print(f"  LLM error ({p}): {e}")
    return None


_TYPE_INSTRUCTIONS = {
    "number": (
        "Return ONLY a single numeric value (integer or decimal). "
        "No units, no currency symbols, no extra text. "
        "Examples: 1500000 or 3.5 or 42. "
        "If not found in context: null"
    ),
    "bool": (
        "Return ONLY the word true or the word false (lowercase, nothing else). "
        "true = yes / granted / approved / allowed / upheld / correct / same. "
        "false = no / dismissed / denied / rejected / refused / different / earlier / later. "
        "Look for court rulings, comparisons, or factual statements. "
        "Make your best determination from the available context. "
        "Only return null if the topic is completely absent from the context."
    ),
    "boolean": (
        "Return ONLY the word true or the word false (lowercase, nothing else). "
        "true = yes / granted / approved / allowed / upheld / correct / same. "
        "false = no / dismissed / denied / rejected / refused / different / earlier / later. "
        "Look for court rulings, comparisons, or factual statements. "
        "Make your best determination from the available context. "
        "Only return null if the topic is completely absent from the context."
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
        f"Answer in 1–3 sentences, maximum {FREE_TEXT_MAX} characters total. "
        "Use ONLY information from the provided context. Do not hallucinate. "
        f"If the answer is absent: {FREE_TEXT_FALLBACK}"
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
        t = len(_TRUE_RE.findall(raw))
        f = len(_FALSE_RE.findall(raw))
        if t > f:
            return True
        if f > t:
            return False
        return None

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
        # Detect "cannot find" responses before trying to parse
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
        # Split only if it looks like a list (contains quotes or short items)
        if '"' in clean or clean.startswith("["):
            items = [x.strip().strip("\"'[] ") for x in re.split(r'[,\n]', clean)]
            items = [x for x in items if x and len(x) < 150 and not _NOT_FOUND_RE.search(x)]
            return items or None
        # Single value — treat as a one-item list
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
        # If LLM starts with fallback then keeps writing, truncate at fallback
        if text.startswith(FREE_TEXT_FALLBACK[:30]):
            return FREE_TEXT_FALLBACK
        return text

    return raw or None


def _find_source_pages(answer: Any, pages: list[dict], answer_type: str = "") -> list[dict]:
    """Return only the 1–2 pages most likely to contain the answer.
    Grounding is F-beta(β=2.5): recall-weighted but precision still penalised.
    Returning fewer pages keeps precision high while recall stays 1 when we pick right.
    """
    if answer is None or not pages:
        return []
    answer_str = str(answer).lower() if not isinstance(answer, list) else " ".join(answer).lower()
    words = [w for w in answer_str.split() if len(w) > 4][:12]

    # For names/free_text allow up to 2 source pages (answer may span docs)
    max_src = 2 if answer_type in ("names", "free_text") else 1

    if not words:
        return pages[:max_src]

    scored = []
    for page in pages:
        page_lower = page["text"].lower()
        hits = sum(1 for w in words if w in page_lower)
        if hits > 0:
            scored.append((hits, page))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:max_src]]
    return pages[:max_src]


def answer_with_llm(
    question_data: dict, pages: list[dict]
) -> tuple[Any, list[dict]]:
    answer_type = question_data.get("answer_type", "free_text")
    question = question_data.get("question", "")

    if _provider() is None:
        from src.pipeline.answer import answer_question
        return answer_question(question_data, pages)

    cfg = _TYPE_CONFIG.get(answer_type, _DEFAULT_CONFIG)
    max_pages = cfg["max_pages"]
    chars_per_page = cfg["chars"]
    max_tokens = cfg["max_tokens"]

    context_parts: list[str] = []
    context_pages: list[dict] = []
    for page in pages[:max_pages]:
        text = page["text"].strip()
        if not text:
            continue
        context_parts.append(f"[{page['doc_id']} p.{page['page_number']}]\n{text[:chars_per_page]}")
        context_pages.append(page)

    if not context_parts:
        if answer_type == "free_text":
            return FREE_TEXT_FALLBACK, []
        return None, []

    context = "\n\n---\n\n".join(context_parts)
    prompt = _build_prompt(question, answer_type, context)
    raw = _call(prompt, max_tokens=max_tokens)

    if raw is None:
        from src.pipeline.answer import answer_question
        return answer_question(question_data, pages)

    answer = _parse(raw, answer_type)

    # Retry with more pages if first attempt returns null (Habr: reparser fallback)
    # Only for types where extra context clearly helps; bool/name can hallucinate on retry
    if answer is None and answer_type in ("number", "date", "names") and len(pages) > max_pages:
        retry_pages = min(max_pages + 3, len(pages))
        retry_parts: list[str] = []
        retry_context_pages: list[dict] = []
        for page in pages[:retry_pages]:
            text = page["text"].strip()
            if not text:
                continue
            retry_parts.append(f"[{page['doc_id']} p.{page['page_number']}]\n{text[:chars_per_page]}")
            retry_context_pages.append(page)
        if len(retry_parts) > len(context_parts):
            retry_context = "\n\n---\n\n".join(retry_parts)
            retry_prompt = _build_prompt(question, answer_type, retry_context)
            raw2 = _call(retry_prompt, max_tokens=max_tokens)
            if raw2 is not None:
                answer = _parse(raw2, answer_type)
                if answer is not None:
                    context_pages = retry_context_pages

    if answer is None:
        return None, []

    if answer_type == "free_text" and answer == FREE_TEXT_FALLBACK:
        return answer, []

    source_pages = _find_source_pages(answer, context_pages, answer_type)
    return answer, source_pages
