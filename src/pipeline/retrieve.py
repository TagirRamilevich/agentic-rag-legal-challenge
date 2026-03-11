import re
from src.pipeline.index import tokenize
from src.pipeline.query_expand import expand_query

_STOPWORDS = frozenset(
    {
        "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "that", "this", "how", "when",
        "where", "who", "which", "does", "do", "did", "has", "have", "had",
        "and", "or", "but", "not", "be", "been", "being", "its", "it", "any",
        "their", "they", "we", "you", "he", "she", "i", "me", "him", "her",
        "us", "them", "my", "your", "his", "our",
    }
)

# Case/law reference patterns
_CASE_RE = re.compile(
    r"\b([A-Z]{2,5}\s+\d{3}/\d{4})\b"
)
_LAW_REF_RE = re.compile(
    r"\b(?:Law No\.\s*\d+\s+of\s+\d{4}|Employment Law|Intellectual Property Law|Trust Law|"
    r"Arbitration Law|Companies Law|Contract Law|Data Protection Law|Strata Title Law|"
    r"Employment Regulations?|Strata Title Regulations?|[A-Z][a-z]+ Law)\b",
    re.IGNORECASE,
)

# Questions about law identity/title/number → first page has the answer
_LAW_IDENTITY_RE = re.compile(
    r"\b(law number|law no\.?|title of|enacted|commencement date|what is the .{0,30} law)\b",
    re.IGNORECASE,
)

# Article reference in question → need to find that specific article page
_ARTICLE_RE = re.compile(r"\bArticle\s+(\d+)(?:\s*\(\d+\))?\b", re.IGNORECASE)

# Indicators that a question compares two different documents/entities
_COMPARISON_RE = re.compile(
    r"\b(same (year|date|entity|authority|minister|body|person|time|month|day|judge|law)|"
    r"(earlier|later|older|newer) (in (the )?year|than)|"
    r"both .{3,60} and|"
    r"administered by (the )?same|"
    r"governed by (the )?same|"
    r"common to both|"
    r"any of the same|"
    r"in both cases?)\b",
    re.IGNORECASE,
)


def _extract_multi_entity_queries(question: str):
    """Return list of sub-queries if question compares 2+ entities, else empty list."""
    # Strategy 1: two or more case numbers → one query per case
    cases = _CASE_RE.findall(question)
    if len(cases) >= 2:
        return list(dict.fromkeys(cases))  # deduplicated, order-preserved

    # Strategy 2: two law references → one query per law
    laws = _LAW_REF_RE.findall(question)
    if len(laws) >= 2:
        return list(dict.fromkeys(laws))

    # Strategy 3: split around comparison connectives
    for splitter in [
        r"\bsame \w+(?: \w+)? (?:as|that\b)",
        r"\bearlier (?:in (?:the )?year )?than\b",
        r"\blater (?:in (?:the )?year )?than\b",
        r"\bboth\b.{3,60}\band\b",
    ]:
        m = re.search(splitter, question, re.IGNORECASE)
        if m:
            left = question[: m.start()].strip()
            right = question[m.end() :].strip()
            if len(left) > 8 and len(right) > 8:
                return [left, right]

    return []


def _bm25_top(bm25, pages, query_str: str, top_k: int) -> list:
    tokens = expand_query(tokenize(query_str))
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_indices = [i for i in ranked[:top_k] if scores[i] > 0]
    if not top_indices:
        top_indices = ranked[: min(3, len(ranked))]
    return top_indices


def is_comparison_question(question: str) -> bool:
    """Return True if question compares two distinct entities requiring dual retrieval."""
    if not _COMPARISON_RE.search(question):
        return False
    # Must also have two distinct references (case numbers OR law names)
    cases = _CASE_RE.findall(question)
    if len(cases) >= 2:
        return True
    laws = _LAW_REF_RE.findall(question)
    if len(laws) >= 2:
        return True
    # Generic comparison connective with enough context on both sides
    for splitter in [r"\bsame \w+(?: \w+)? (?:as|that\b)", r"\bearlier (?:in (?:the )?year )?than\b", r"\blater (?:in (?:the )?year )?than\b"]:
        m = re.search(splitter, question, re.IGNORECASE)
        if m and len(question[: m.start()].strip()) > 8 and len(question[m.end():].strip()) > 8:
            return True
    return False


# Extra BM25 anchor terms appended per answer_type to boost relevant pages
_TYPE_ANCHORS: dict[str, str] = {
    "number": "AED",
    "date":   "enacted commencement",
}


def retrieve_pages(
    bm25,
    pages: list[dict],
    question: str,
    top_k: int = 20,
    add_neighbors: bool = True,
    answer_type: str = "",
) -> list[dict]:
    sub_queries = []
    if _COMPARISON_RE.search(question):
        sub_queries = _extract_multi_entity_queries(question)

    result: dict[int, dict] = {}
    top_indices: list[int] = []
    doc_page_index = {(p["doc_id"], p["page_number"]): j for j, p in enumerate(pages)}

    if sub_queries:
        # For comparison questions: INTERLEAVE pages from each entity so that
        # each entity is represented early (critical when max_pages is small)
        per_q = max(5, top_k // max(len(sub_queries), 1))
        per_q_early_lists: list[list[int]] = []
        per_q_bm25_lists: list[list[int]] = []
        for sq in sub_queries:
            idxs = _bm25_top(bm25, pages, sq, per_q)
            early: list[int] = []
            if idxs:
                top_doc_id = pages[idxs[0]]["doc_id"]
                for early_pg in (1, 2, 3, 4):
                    j = doc_page_index.get((top_doc_id, early_pg))
                    if j is not None:
                        early.append(j)
            per_q_early_lists.append(early)
            per_q_bm25_lists.append(idxs)

        # Interleave: round-robin early pages from each entity first
        max_early = max(len(e) for e in per_q_early_lists) if per_q_early_lists else 0
        for slot in range(max_early):
            for early_list in per_q_early_lists:
                if slot < len(early_list):
                    j = early_list[slot]
                    if j not in result:
                        result[j] = pages[j]
                        top_indices.append(j)

        # Then interleave BM25 pages from each entity
        max_bm25 = max(len(b) for b in per_q_bm25_lists) if per_q_bm25_lists else 0
        for slot in range(max_bm25):
            for bm25_list in per_q_bm25_lists:
                if slot < len(bm25_list):
                    i = bm25_list[slot]
                    if i not in result and len(result) < top_k * 2:
                        result[i] = pages[i]
                        top_indices.append(i)

    # Full-question BM25 (fills remaining slots, may find cross-entity pages)
    anchors = _TYPE_ANCHORS.get(answer_type, "")
    augmented_question = f"{question} {anchors}".strip() if anchors else question
    tokens = expand_query(tokenize(augmented_question))
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    main_indices = [i for i in ranked[:top_k] if scores[i] > 0]
    if not main_indices:
        main_indices = ranked[: min(5, len(ranked))]
    for i in main_indices:
        if i not in result:
            result[i] = pages[i]
            top_indices.append(i)

    # Priority pages to prepend (appear first in returned list for the fallback reranker)
    priority: list[int] = []

    # Law-identity questions: include p.1 of the top matching document
    # (title pages always have the law number and enactment date)
    if _LAW_IDENTITY_RE.search(question) and main_indices:
        top_doc_id = pages[main_indices[0]]["doc_id"]
        for pg in (1, 2, 3):
            j = doc_page_index.get((top_doc_id, pg))
            if j is not None:
                priority.append(j)
                if j not in result:
                    result[j] = pages[j]
                    top_indices.append(j)

    # Article-specific questions: search for that article within the top doc
    article_m = _ARTICLE_RE.search(question)
    if article_m and main_indices:
        article_num = article_m.group(1)
        # Match "Article 14" OR "14." OR "14(" at start/middle of line (numbered sections)
        article_pat = re.compile(
            rf"(?:\bArticle\s+{article_num}\b|(?:^|\n)\s*{article_num}[.(])",
            re.IGNORECASE | re.MULTILINE,
        )
        top_doc_id = pages[main_indices[0]]["doc_id"]
        for idx, p in enumerate(pages):
            if p["doc_id"] == top_doc_id and article_pat.search(p["text"]):
                priority.append(idx)
                if idx not in result:
                    result[idx] = pages[idx]
                    top_indices.append(idx)
                if len(priority) >= 5:
                    break

    if add_neighbors:
        for i in list(top_indices[:8]):
            page = pages[i]
            for delta in (-1, 1):
                key = (page["doc_id"], page["page_number"] + delta)
                j = doc_page_index.get(key)
                if j is not None and j not in result:
                    result[j] = pages[j]

    # Build final ordered list: priority pages first, then remaining
    all_indices = list(dict.fromkeys(priority + list(result.keys())))
    return [pages[i] for i in all_indices if i in result]
