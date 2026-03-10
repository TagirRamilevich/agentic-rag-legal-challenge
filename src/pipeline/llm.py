import json
import os
import re
from typing import Any, Optional

FREE_TEXT_FALLBACK = "There is no information on this question in the provided documents."
FREE_TEXT_MAX = 280
CONTEXT_CHARS_PER_PAGE = 2500
MAX_CONTEXT_PAGES = 5


def _provider() -> Optional[str]:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def _call(prompt: str, max_tokens: int = 256) -> Optional[str]:
    p = _provider()
    try:
        if p == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()

        if p == "openrouter":
            import requests as req
            r = req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json={
                    "model": "anthropic/claude-haiku-4-5",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        if p == "openai":
            import openai
            client = openai.OpenAI()
            r = client.chat.completions.create(
                model="gpt-4o-mini",
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
        "No units, no currency symbols, no text whatsoever. "
        "If the answer is not present in the context, return exactly: null"
    ),
    "bool": (
        "Return ONLY true or false (lowercase). "
        "If the answer is not present in the context, return exactly: null"
    ),
    "boolean": (
        "Return ONLY true or false (lowercase). "
        "If the answer is not present in the context, return exactly: null"
    ),
    "date": (
        "Return ONLY a date in YYYY-MM-DD format. "
        "If the answer is not present in the context, return exactly: null"
    ),
    "name": (
        "Return ONLY the name or entity as a short phrase (no explanation, no punctuation around it). "
        "If the answer is not present in the context, return exactly: null"
    ),
    "names": (
        'Return ONLY a JSON array of strings, e.g. ["Alice Corp", "Beta Ltd"]. '
        "If the answer is not present in the context, return exactly: null"
    ),
    "free_text": (
        f"Answer in 1–3 sentences, maximum {FREE_TEXT_MAX} characters total. "
        "Base your answer ONLY on the information in the provided context. "
        "Do not hallucinate. "
        f"If the answer is not in the context, return exactly: {FREE_TEXT_FALLBACK}"
    ),
}


def _build_prompt(question: str, answer_type: str, context: str) -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        "Extract the answer from the following legal document context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer type: {answer_type}\n"
        f"Instruction: {instruction}\n\n"
        "Answer:"
    )


def _parse(raw: str, answer_type: str) -> Any:
    raw = raw.strip()
    if raw.lower() in ("null", "none", "n/a", ""):
        return None

    if answer_type in ("bool", "boolean"):
        if raw.lower() in ("true", "yes"):
            return True
        if raw.lower() in ("false", "no"):
            return False
        return None

    if answer_type == "number":
        clean = re.sub(r"[$€£,\s]", "", raw)
        m = re.search(r"-?\d+(?:\.\d+)?", clean)
        if m:
            try:
                v = float(m.group())
                return int(v) if v == int(v) else v
            except (ValueError, OverflowError):
                pass
        return None

    if answer_type == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if m:
            y, mo, d = (int(x) for x in m.group().split("-"))
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return m.group()
        return None

    if answer_type == "names":
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x] or None
        except json.JSONDecodeError:
            pass
        items = [x.strip().strip("\"'") for x in re.split(r"[,\n;]", raw) if x.strip()]
        return items or None

    if answer_type == "name":
        result = raw.strip("\"' ").strip()[:200]
        return result or None

    if answer_type == "free_text":
        return raw[:FREE_TEXT_MAX]

    return raw or None


def _find_source_pages(answer: Any, pages: list[dict]) -> list[dict]:
    if answer is None or not pages:
        return []
    answer_str = str(answer).lower() if not isinstance(answer, list) else " ".join(answer).lower()
    words = [w for w in answer_str.split() if len(w) > 4][:12]
    if not words:
        return pages

    scored = []
    for page in pages:
        page_lower = page["text"].lower()
        hits = sum(1 for w in words if w in page_lower)
        if hits > 0:
            scored.append((hits, page))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]
    return pages


def answer_with_llm(
    question_data: dict, pages: list[dict]
) -> tuple[Any, list[dict]]:
    answer_type = question_data.get("answer_type", "free_text")
    question = question_data.get("question", "")

    if _provider() is None:
        from src.pipeline.answer import answer_question
        return answer_question(question_data, pages)

    context_parts: list[str] = []
    context_pages: list[dict] = []
    for page in pages[:MAX_CONTEXT_PAGES]:
        text = page["text"].strip()
        if not text:
            continue
        context_parts.append(f"[{page['doc_id']} p.{page['page_number']}]\n{text[:CONTEXT_CHARS_PER_PAGE]}")
        context_pages.append(page)

    if not context_parts:
        if answer_type == "free_text":
            return FREE_TEXT_FALLBACK, []
        return None, []

    context = "\n\n---\n\n".join(context_parts)
    prompt = _build_prompt(question, answer_type, context)
    raw = _call(prompt)

    if raw is None:
        from src.pipeline.answer import answer_question
        return answer_question(question_data, pages)

    answer = _parse(raw, answer_type)

    if answer is None:
        return None, []

    if answer_type == "free_text" and answer == FREE_TEXT_FALLBACK:
        return answer, []

    source_pages = _find_source_pages(answer, context_pages)
    return answer, source_pages
