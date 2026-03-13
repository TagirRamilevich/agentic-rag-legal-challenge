import re
from typing import Any, Optional

from src.utils.number_parse import parse_number

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
    q_lower = question.lower()
    best_num, best_page, best_score = None, None, -1

    # For "law number" questions, try specific pattern first
    _is_law_num_q = bool(re.search(r"\b(law\s+number|official.*number|difc\s+law\s+no)\b", q_lower))

    for page in pages:
        if _is_law_num_q:
            # Try structured extraction: "Law No. X of YYYY" or "DIFC Law No. X"
            m = re.search(r"(?:DIFC\s+)?Law\s+No\.?\s*(\d+)", page["text"], re.IGNORECASE)
            if m:
                val = int(m.group(1))
                return val, [page]

        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            if score > best_score:
                # Strip article/section sub-references before parsing to avoid
                # extracting article numbers as answers (e.g. "Article 11(1)" → 11)
                clean_sent = re.sub(
                    r"\b(?:Article|Section|Clause|Schedule|Part)\s+\d+(?:\(\w+\))*",
                    "", sent, flags=re.IGNORECASE,
                )
                # Also strip bare section-heading numbers (Operating Law format: "11\n(1)...")
                clean_sent = re.sub(r"(?:^|\n)\s*(\d+)\s*(?:\n|$)", " ", clean_sent)
                val = parse_number(clean_sent)
                if val is not None:
                    # Legal Q&A about counts/durations: always positive
                    if val < 0 and "-" not in sent:
                        val = -val
                    best_score = score
                    best_num = val
                    best_page = page

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
                            break  # take most specific pattern first
                    except Exception:
                        pass

    if best_page is not None:
        return best_date, [best_page]
    return None, []


def _extract_date_from_text(text: str) -> Optional[str]:
    """Extract the first valid date from text, trying all patterns."""
    for pattern, formatter in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                date_str = formatter(m)
                y, mo, d = (int(x) for x in date_str.split("-"))
                if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return date_str
            except Exception:
                pass
    return None


def _extract_comparison_date_name(
    pages: list[dict], question: str
) -> tuple[Optional[str], list[dict]]:
    """For 'which case has earlier date' questions: extract dates from docs, compare."""
    # Extract case references from question
    case_refs = re.findall(r"\b([A-Z]{2,5}\s+\d{3}/\d{4})\b", question)
    if len(case_refs) < 2:
        return None, []

    # Find dates for each case reference from their documents
    case_dates: dict[str, tuple[str, dict]] = {}  # case_ref -> (date_str, page)
    for page in pages:
        text = page["text"]
        for ref in case_refs:
            if ref in text or ref.replace(" ", "") in text.replace(" ", ""):
                if ref in case_dates:
                    continue
                # 1) Look for explicit "Date of Issue"
                doi_m = re.search(r"Date\s+of\s+Issue[:\s]*([^\n]+)", text, re.IGNORECASE)
                if doi_m:
                    d = _extract_date_from_text(doi_m.group(1))
                    if d:
                        case_dates[ref] = (d, page)
                        continue
                # 2) Look for header date: "MONTH DD, YYYY COURT/SCT/..." (common case doc format)
                hdr_m = re.search(
                    rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\s+(?:COURT|SMALL|SCT|ARBITRATION|ENFORCEMENT)",
                    text[:400], re.IGNORECASE,
                )
                if hdr_m:
                    mo = _MONTH_MAP.get(hdr_m.group(1).lower(), 1)
                    d_str = f"{hdr_m.group(3)}-{mo:02d}-{int(hdr_m.group(2)):02d}"
                    case_dates[ref] = (d_str, page)
                    continue
                # 3) Fallback: first date in first 500 chars
                d = _extract_date_from_text(text[:500])
                if d:
                    case_dates[ref] = (d, page)

    if len(case_dates) >= 2:
        q_lower = question.lower()
        want_earlier = "earlier" in q_lower or "first" in q_lower
        sorted_cases = sorted(case_dates.items(), key=lambda x: x[1][0])
        chosen = sorted_cases[0] if want_earlier else sorted_cases[-1]
        ref, (date_str, page) = chosen
        return ref, [case_dates[r][1] for r in case_dates]

    return None, []


def extract_name(
    pages: list[dict], question: str
) -> tuple[Optional[str], list[dict]]:
    q_lower = question.lower()

    # For "which case has earlier/later date" comparison questions, use structured extraction
    if re.search(r"\b(earlier|later|first|which case|which document)\b.*\b(date|issue)", q_lower):
        result, used = _extract_comparison_date_name(pages, question)
        if result is not None:
            return result, used

    # For "what document/form/statement must" questions: extract the specific document type
    if re.search(r"\b(what\s+document|what\s+form|what\s+statement|what\s+report|what\s+certificate)\b", q_lower):
        # Look for patterns like "file... a [DOCUMENT]", "submit a [DOCUMENT]"
        # Limit to 1-3 capitalized words (document names are short)
        _doc_pat = re.compile(
            r"(?:file|submit|produce|deliver|furnish|provide|prepare|lodge)\s+"
            r"(?:with\s+\w+\s*,?\s*)?(?:a|an|the)\s+"
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
        )
        for page in pages:
            for sent in _sentences(page["text"]):
                m = _doc_pat.search(sent)
                if m:
                    doc_name = m.group(1).strip()
                    if len(doc_name) > 3 and doc_name.lower() not in _STOPWORDS:
                        return doc_name, [page]

    keywords = _keywords(question)
    best_name, best_page, best_score = None, None, -1

    # Filter out generic subjects that are not entity names
    _generic_prefixes = {"Every", "The", "A", "An", "All", "Each", "No", "Any"}

    for page in pages:
        for sent in _sentences(page["text"]):
            score = _relevance(sent, keywords)
            if score > 0:
                candidates = [
                    n for n in _PROPER_NOUN_RE.findall(sent)
                    if n.lower() not in _STOPWORDS
                    and len(n) > 3
                    and n.split()[0] not in _generic_prefixes
                ]
                if candidates and score > best_score:
                    best_score = score
                    best_name = max(candidates, key=len)
                    best_page = page

    if best_page is not None:
        return best_name, [best_page]
    return None, []


_PARTY_NOISE = frozenset({
    "court", "appeal", "order", "orders", "upon", "and", "between", "claim",
    "claimant", "defendant", "respondent", "appellant", "applicant",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "dubai", "difc", "courts", "justice", "judge", "honour",
    "arbitration", "division", "instance", "section", "part",
    "claim no", "case no", "renewed", "application", "disbursements",
})


def _extract_parties_from_between(text: str) -> list[str]:
    """Extract party names from BETWEEN...and... sections in court documents."""
    m = re.search(r"BETWEEN\s*\n(.+?)(?:\nORDER|\nUPON|\nBEFORE)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    between_text = m.group(1)
    # Split on "and" separator
    parts = re.split(r"\n\s*and\s*\n", between_text, flags=re.IGNORECASE)
    parties = []
    for part in parts:
        # Get the first line (party name), skip role labels
        lines = [l.strip() for l in part.strip().split("\n") if l.strip()]
        for line in lines:
            # Skip role labels
            if re.match(r"^(Claimant|Defendant|Respondent|Appellant|Applicant)", line, re.IGNORECASE):
                continue
            # Skip numbered references like "(1)" at start
            name = re.sub(r"^\(\d+\)\s*", "", line).strip()
            if name and len(name) > 3 and name.lower() not in _PARTY_NOISE:
                parties.append(name)
    return parties


def extract_names(
    pages: list[dict], question: str
) -> tuple[Optional[list], list[dict]]:
    keywords = _keywords(question)
    q_lower = question.lower()

    # For party-related questions, try structured extraction first
    is_party_q = any(w in q_lower for w in ["claimant", "defendant", "respondent", "party", "parties"])
    if is_party_q:
        for page in pages:
            parties = _extract_parties_from_between(page["text"])
            if parties:
                # Filter based on role mentioned in question
                if "claimant" in q_lower:
                    # First party in BETWEEN is typically the claimant
                    return parties[:1] if len(parties) >= 2 else parties, [page]
                if "defendant" in q_lower or "respondent" in q_lower:
                    return parties[1:] if len(parties) >= 2 else parties, [page]
                return parties, [page]

    collected: list[str] = []
    used_pages: list[dict] = []

    for page in pages:
        text = page["text"]
        page_names: list[str] = []

        for sent in _sentences(text):
            if _relevance(sent, keywords) > 0:
                candidates = [
                    n for n in _PROPER_NOUN_RE.findall(sent)
                    if n.lower() not in _PARTY_NOISE and n.lower() not in _STOPWORDS and len(n) > 3
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
                    if n.lower() not in _PARTY_NOISE and n.lower() not in _STOPWORDS and len(n) > 3
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
