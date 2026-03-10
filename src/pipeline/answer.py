import re
from typing import Any, Optional

FREE_TEXT_FALLBACK = (
    "There is no information on this question in the provided documents."
)

_STOPWORDS = frozenset(
    {
        "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "that", "this", "how", "when",
        "where", "who", "which", "does", "do", "did", "has", "have", "had",
        "and", "or", "but", "not", "be", "been", "being", "its", "it", "any",
    }
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_PATTERN = "|".join(_MONTH_MAP.keys())

_DATE_PATTERNS = [
    (
        r"\b(\d{4})[/\-](\d{2})[/\-](\d{2})\b",
        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
    ),
    (
        r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b",
        lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}",
    ),
    (
        rf"\b(\d{{1,2}})\s+({_MONTH_PATTERN})\s+(\d{{4}})\b",
        lambda m: f"{m.group(3)}-{_MONTH_MAP.get(m.group(2).lower(), 1):02d}-{int(m.group(1)):02d}",
    ),
    (
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        lambda m: f"{m.group(3)}-{_MONTH_MAP.get(m.group(1).lower(), 1):02d}-{int(m.group(2)):02d}",
    ),
    (
        rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b",
        lambda m: f"{m.group(2)}-{_MONTH_MAP.get(m.group(1).lower(), 1):02d}-01",
    ),
]

_AFFIRM_RE = re.compile(
    r"\b(yes|true|approved|confirmed|granted|authorized|permitted|valid|correct|agreed|comply|complied|satisf\w*|eligible|qualif\w*|entitl\w*)\b",
    re.IGNORECASE,
)
_DENY_RE = re.compile(
    r"\b(no\b|false|denied|rejected|prohibited|invalid|incorrect|wrong|disagree|non.?compli\w*|ineligible|cannot|shall not|may not|does not|did not|is not|was not|are not|were not|not authorized|not permitted|not entitl\w*)\b",
    re.IGNORECASE,
)
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUMBER_RE = re.compile(
    r"[\$€£]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:%|percent)?"
)


def _keywords(question: str) -> list[str]:
    words = re.sub(r"[^\w\s]", " ", question.lower()).split()
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _sentences(text: str) -> list[str]:
    return re.split(r"[.!?\n]", text)


def _relevance(sentence: str, keywords: list[str]) -> int:
    low = sentence.lower()
    return sum(1 for kw in keywords if kw in low)


def extract_bool(
    pages: list[dict], question: str
) -> tuple[Optional[bool], list[dict]]:
    keywords = _keywords(question)
    best_val, best_page, best_score = None, None, -1

    for page in pages:
        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            if score > 0:
                a = len(_AFFIRM_RE.findall(sent))
                d = len(_DENY_RE.findall(sent))
                if a != d and score > best_score:
                    best_score = score
                    best_val = a > d
                    best_page = page

    if best_page is not None:
        return best_val, [best_page]
    return None, []


def extract_number(
    pages: list[dict], question: str
) -> tuple[Optional[Any], list[dict]]:
    keywords = _keywords(question)
    best_num, best_page, best_score = None, None, -1

    for page in pages:
        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            if score > best_score:
                matches = _NUMBER_RE.findall(sent)
                if matches:
                    try:
                        val = float(matches[0].replace(",", ""))
                        best_score = score
                        best_num = int(val) if val == int(val) else val
                        best_page = page
                    except (ValueError, OverflowError):
                        pass

    if best_page is not None:
        return best_num, [best_page]
    return None, []


def extract_date(
    pages: list[dict], question: str
) -> tuple[Optional[str], list[dict]]:
    keywords = _keywords(question)
    best_date, best_page, best_score = None, None, -1

    for page in pages:
        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            for pattern, formatter in _DATE_PATTERNS:
                m = re.search(pattern, sent, re.IGNORECASE)
                if m:
                    try:
                        date_str = formatter(m)
                        y, mo, d = (int(x) for x in date_str.split("-"))
                        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                            if score >= best_score:
                                best_score = score
                                best_date = date_str
                                best_page = page
                    except Exception:
                        pass

    if best_page is not None:
        return best_date, [best_page]
    return None, []


def extract_name(
    pages: list[dict], question: str
) -> tuple[Optional[str], list[dict]]:
    keywords = _keywords(question)
    best_name, best_page, best_score = None, None, -1

    for page in pages:
        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            if score > 0:
                candidates = [
                    n for n in _PROPER_NOUN_RE.findall(sent)
                    if n.lower() not in _STOPWORDS and len(n) > 3
                ]
                if candidates and score > best_score:
                    best_score = score
                    best_name = max(candidates, key=len)
                    best_page = page

    if best_page is not None:
        return best_name, [best_page]
    return None, []


def extract_names(
    pages: list[dict], question: str
) -> tuple[Optional[list], list[dict]]:
    keywords = _keywords(question)
    collected: list[str] = []
    used_pages: list[dict] = []

    for page in pages:
        text = page["text"]
        page_names: list[str] = []

        for sent in _sentences(text):
            if _relevance(sent, keywords) > 0:
                candidates = [
                    n for n in _PROPER_NOUN_RE.findall(sent)
                    if n.lower() not in _STOPWORDS and len(n) > 3
                ]
                page_names.extend(candidates)

        list_items = re.findall(
            r"(?:^|\n)\s*(?:\d+[.)]\s+|-\s+|\*\s+|•\s+)([^\n]{3,150})",
            text,
            re.MULTILINE,
        )
        for item in list_items:
            item = item.strip()
            if not keywords or any(kw in item.lower() for kw in keywords):
                page_names.extend(
                    n for n in _PROPER_NOUN_RE.findall(item)
                    if n.lower() not in _STOPWORDS and len(n) > 3
                )

        if page_names:
            seen_on_page = []
            for n in page_names:
                if n not in seen_on_page:
                    seen_on_page.append(n)
            collected.extend(seen_on_page)
            used_pages.append(page)

    if collected:
        deduped = []
        for item in collected:
            if item not in deduped:
                deduped.append(item)
        return deduped[:30], used_pages

    return None, []


def answer_question(
    question_data: dict, pages: list[dict]
) -> tuple[Any, list[dict]]:
    answer_type = question_data.get("answer_type", "free_text")
    question_text = question_data.get("question", "")

    if answer_type == "free_text":
        return FREE_TEXT_FALLBACK, []

    if answer_type in ("bool", "boolean"):
        return extract_bool(pages, question_text)

    if answer_type == "number":
        return extract_number(pages, question_text)

    if answer_type == "date":
        return extract_date(pages, question_text)

    if answer_type == "name":
        return extract_name(pages, question_text)

    if answer_type == "names":
        val, used = extract_names(pages, question_text)
        if val is None:
            single, used = extract_name(pages, question_text)
            if single is not None:
                val = [single]
        return val, used

    return None, []
