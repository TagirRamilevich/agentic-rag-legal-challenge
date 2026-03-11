import re
import numpy as np
from src.pipeline.index import tokenize, embed_query, _EMBED_AVAILABLE, _DEFAULT_EMBED_MODEL
from src.pipeline.query_expand import expand_query

# ---------------------------------------------------------------------------
# Document routing index: maps entity references → doc_ids
# ---------------------------------------------------------------------------
_DOC_ROUTE_CACHE: dict = {}


def build_doc_routing_index(pages: list[dict]) -> dict:
    """Build a lookup from case numbers and law names to doc_ids.
    Uses page 1 text from each document for entity extraction."""
    key = id(pages)
    if key in _DOC_ROUTE_CACHE:
        return _DOC_ROUTE_CACHE[key]

    case_to_doc: dict[str, str] = {}
    law_to_doc: dict[str, str] = {}

    _case_p = re.compile(r"\b([A-Z]{2,5}\s+\d{3}/\d{4})\b")
    _law_no_p = re.compile(r"\b((?:DIFC\s+)?Law No\.\s*\d+\s+of\s+\d{4})\b", re.IGNORECASE)
    _law_name_p = re.compile(
        r"\b((?:[A-Z][a-z]+\s+){1,5}(?:Law|Regulations?|Rules?)(?:\s+\d{4})?)\b"
    )
    # Also match ALL-CAPS law names from title pages (e.g. "OPERATING LAW")
    _law_name_caps_p = re.compile(
        r"\b((?:[A-Z]{3,}[ \t]+){1,5}(?:LAW|REGULATIONS?|RULES?))\b"
    )

    for p in pages:
        if p["page_number"] != 1:
            continue
        text = p["text"][:500]
        doc_id = p["doc_id"]

        # Extract case numbers
        for case in _case_p.findall(text):
            case_norm = case.upper().replace("  ", " ")
            if case_norm not in case_to_doc:
                case_to_doc[case_norm] = doc_id

        # Extract law references
        for law in _law_no_p.findall(text):
            law_norm = law.strip()
            if law_norm not in law_to_doc:
                law_to_doc[law_norm] = doc_id

        for law in _law_name_p.findall(text):
            law_norm = law.strip()
            if len(law_norm) > 8 and law_norm not in law_to_doc:
                law_to_doc[law_norm] = doc_id
        # Also extract ALL-CAPS law names and store as Title Case
        for law in _law_name_caps_p.findall(text):
            law_title = law.strip().title()  # "OPERATING LAW" → "Operating Law"
            if len(law_title) > 8 and law_title not in law_to_doc:
                law_to_doc[law_title] = doc_id

    index = {"case": case_to_doc, "law": law_to_doc}
    _DOC_ROUTE_CACHE[key] = index
    return index

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
    r"in both cases?|"
    r"which (case|document|party|judgment) (has|had|is|was|were) (the )?(earlier|later|higher|lower|greater|smaller|first|last|same)|"
    r"(between|comparing) .{3,60} (and|or|vs\.?|versus))\b",
    re.IGNORECASE,
)


def _extract_multi_entity_queries(question: str):
    """Return list of sub-queries if question compares 2+ entities, else empty list."""
    # Strategy 1: two or more case numbers → one query per case
    cases = _CASE_RE.findall(question)
    if len(cases) >= 2:
        return list(dict.fromkeys(cases))  # deduplicated, order-preserved

    # Strategy 2: two law references → one query per law.
    # First check for specific "Law No. X of Y" patterns (most discriminative).
    _law_no_re = re.compile(r"\bLaw No\.\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
    law_nos = list(dict.fromkeys(_law_no_re.findall(question)))
    if len(law_nos) >= 2:
        return law_nos[:2]
    # Then try multi-word law phrase extractor for named laws
    # (e.g. "Leasing Law", "Real Property Law Amendment Law")
    _mp_law_re = re.compile(
        r"\b(?:[A-Z][a-z]+\s+){1,4}(?:Law|Regulations?)\b"
    )
    law_phrases = [m.group(0) for m in _mp_law_re.finditer(question)]
    # Remove any phrase that is a suffix of a longer phrase already in the list
    unique_laws: list[str] = []
    for ph in law_phrases:
        dominated = any(ph != other and ph in other for other in law_phrases)
        if not dominated and ph not in unique_laws:
            unique_laws.append(ph)
    if len(unique_laws) >= 2:
        # Use the 2 longest phrases (most specific) rather than first 2
        # e.g. "Employment Regulations" preferred over "Employment Law" when both present
        return sorted(unique_laws, key=len, reverse=True)[:2]

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


def _embedding_top(
    embeddings: np.ndarray,
    query_str: str,
    top_k: int,
    model_name: str = _DEFAULT_EMBED_MODEL,
) -> list[int]:
    """Return top-k page indices by cosine similarity to query embedding."""
    q_vec = embed_query(query_str, model_name)          # (1, D)
    scores = (embeddings @ q_vec.T).flatten()            # (N,)
    ranked = np.argsort(-scores)[:top_k]
    return ranked.tolist()


def _rrf_fuse(
    *rankings: list[int],
    top_k: int = 20,
    k: int = 60,
) -> list[int]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    For each index, RRF score = sum over lists of 1 / (k + rank).
    k=60 is the standard default from the RRF paper.
    Returns fused ranking (top_k indices).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores, key=lambda i: -scores[i])
    return fused[:top_k]


def _hybrid_top(
    bm25,
    pages,
    embeddings,
    query_str: str,
    top_k: int,
) -> list[int]:
    """Hybrid retrieval: BM25 + embedding search fused with RRF.

    Falls back to BM25-only if embeddings are None.
    """
    bm25_indices = _bm25_top(bm25, pages, query_str, top_k * 2)
    if embeddings is None:
        return bm25_indices[:top_k]
    emb_indices = _embedding_top(embeddings, query_str, top_k * 2)
    return _rrf_fuse(bm25_indices, emb_indices, top_k=top_k)


def is_comparison_question(question: str) -> bool:
    """Return True if question compares two distinct entities requiring dual retrieval."""
    # Two case numbers alone → always a comparison question
    cases = _CASE_RE.findall(question)
    if len(cases) >= 2:
        return True
    if not _COMPARISON_RE.search(question):
        return False
    # Must also have two distinct references (case numbers OR law names)
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


_LAW_NO_EXACT_RE = re.compile(r"\bLaw No\.\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)


def _find_enactment_notice_doc(
    sub_q: str,
    pages: list[dict],
    doc_page_counts: dict,
    doc_page_index: dict,
) -> str:
    """For a 'Law No. X of YYYY' sub-query, return the enactment notice doc_id (1-page doc).
    Returns empty string if not found."""
    sq_lower = sub_q.lower()
    for p in pages:
        if p["page_number"] == 1 and doc_page_counts.get(p["doc_id"], 99) <= 2:
            t_lower = p["text"].lower()
            if "enactment notice" in t_lower and sq_lower in t_lower:
                return p["doc_id"]
    return ""


def retrieve_pages(
    bm25,
    pages: list[dict],
    question: str,
    top_k: int = 20,
    add_neighbors: bool = True,
    answer_type: str = "",
    embeddings=None,
) -> list[dict]:
    sub_queries = []
    # Two+ case refs always trigger dual-query even without explicit comparison language
    _has_multi_case = len(_CASE_RE.findall(question)) >= 2
    if _COMPARISON_RE.search(question) or _has_multi_case:
        sub_queries = _extract_multi_entity_queries(question)

    result: dict[int, dict] = {}
    top_indices: list[int] = []
    doc_page_index = {(p["doc_id"], p["page_number"]): j for j, p in enumerate(pages)}

    from collections import Counter as _CounterDP
    _doc_page_counts = _CounterDP(p["doc_id"] for p in pages)

    # Use document routing to boost known entities
    doc_route = build_doc_routing_index(pages)
    routed_doc_ids: set[str] = set()
    for case_ref in _CASE_RE.findall(question):
        case_norm = case_ref.upper().replace("  ", " ")
        if case_norm in doc_route["case"]:
            routed_doc_ids.add(doc_route["case"][case_norm])
    for law_ref in _LAW_REF_RE.findall(question):
        # Skip overly broad matches like "DIFC Law" that match everything
        if len(law_ref) < 10 and not re.match(r"Law No\.", law_ref, re.IGNORECASE):
            continue
        for key, doc_id in doc_route["law"].items():
            if law_ref.lower() in key.lower() or key.lower() in law_ref.lower():
                routed_doc_ids.add(doc_id)
    # Add page 1 of routed docs to ensure they're in the result
    for rdoc in routed_doc_ids:
        j = doc_page_index.get((rdoc, 1))
        if j is not None and j not in result:
            result[j] = pages[j]
            top_indices.append(j)

    if sub_queries:
        # For comparison questions: INTERLEAVE pages from each entity so that
        # each entity is represented early (critical when max_pages is small)
        per_q = max(5, top_k // max(len(sub_queries), 1))
        per_q_early_lists: list[list[int]] = []
        per_q_bm25_lists: list[list[int]] = []
        _date_q = re.search(
            r"\b(come into force|commencement|enacted|same date|same year|enact)\b",
            question, re.IGNORECASE,
        )
        for sq in sub_queries:
            idxs = _hybrid_top(bm25, pages, embeddings, sq, per_q)
            early: list[int] = []
            if idxs:
                top_doc_id = pages[idxs[0]]["doc_id"]
                # For date/commencement questions, prefer enactment notice (1-page doc).
                if _date_q:
                    notice_doc = _find_enactment_notice_doc(
                        sq, pages, _doc_page_counts, doc_page_index
                    )
                    if notice_doc:
                        top_doc_id = notice_doc
                else:
                    # Title-match fallback: if BM25 top doc's p.1 title doesn't contain
                    # the sub-query terms, find a doc whose title better matches.
                    # Helps when BM25 picks a large doc that merely references the law
                    # (e.g. "Employment Law" for "Employment Regulations" sub-query).
                    top_p1_j = doc_page_index.get((top_doc_id, 1))
                    top_title = pages[top_p1_j]["text"].lower()[:200] if top_p1_j is not None else ""
                    sq_terms = [w for w in sq.lower().split() if len(w) > 3]
                    title_match = sum(1 for t in sq_terms if t in top_title)
                    if sq_terms and title_match < len(sq_terms) * 0.7:
                        # Search for better doc: highest page-count doc whose p.1 title
                        # contains the majority of sub-query terms.
                        best_match_cnt = 0
                        best_match_doc = ""
                        best_match_pages = 0
                        for p in pages:
                            if p["page_number"] == 1:
                                t_low = p["text"].lower()[:300]
                                tc = sum(1 for t in sq_terms if t in t_low)
                                pc = _doc_page_counts.get(p["doc_id"], 0)
                                if tc >= len(sq_terms) * 0.7:
                                    if tc > best_match_cnt or (tc == best_match_cnt and pc > best_match_pages):
                                        best_match_cnt = tc
                                        best_match_doc = p["doc_id"]
                                        best_match_pages = pc
                        if best_match_doc:
                            top_doc_id = best_match_doc
                for early_pg in (1, 2, 3, 4):
                    j = doc_page_index.get((top_doc_id, early_pg))
                    if j is not None:
                        early.append(j)
            per_q_early_lists.append(early)
            per_q_bm25_lists.append(idxs)

        # Interleave: round-robin early pages from each entity first.
        # Tag p.1 of each entity as _priority so reranker won't drop them
        # (title pages contain enactment dates needed for law comparison questions).
        max_early = max(len(e) for e in per_q_early_lists) if per_q_early_lists else 0
        for slot in range(max_early):
            for early_list in per_q_early_lists:
                if slot < len(early_list):
                    j = early_list[slot]
                    page_obj = pages[j]
                    if slot == 0 and page_obj["page_number"] == 1:
                        # First page of each entity: tag as priority
                        tagged = dict(page_obj)
                        tagged["_priority"] = True
                        result[j] = tagged
                    elif j not in result:
                        result[j] = page_obj
                    if j not in top_indices:
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

    # Full-question hybrid retrieval (fills remaining slots, may find cross-entity pages)
    anchors = _TYPE_ANCHORS.get(answer_type, "")
    augmented_question = f"{question} {anchors}".strip() if anchors else question
    main_indices = _hybrid_top(bm25, pages, embeddings, augmented_question, top_k)
    if not main_indices:
        # Fallback: raw BM25 top-5
        tokens = expand_query(tokenize(augmented_question))
        scores = bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        main_indices = ranked[: min(5, len(ranked))]
    for i in main_indices:
        if i not in result:
            result[i] = pages[i]
            top_indices.append(i)

    # Priority pages to prepend (appear first in returned list for the fallback reranker)
    priority: list[int] = []

    # Article-specific questions: search for that article within the top doc.
    # Tag with _priority so reranker pins them (cross-encoder can misrank article pages).
    # Run article search BEFORE law-identity so article pages take precedence.
    article_m = _ARTICLE_RE.search(question)
    if article_m and main_indices:
        article_num = article_m.group(1)
        article_pat = re.compile(
            rf"(?:\bArticle\s+{article_num}\b|(?:^|\n)\s*{article_num}[.(])",
            re.IGNORECASE | re.MULTILINE,
        )
        # If question names a specific law, prefer that doc over the top BM25 result.
        # Find the longest (most pages) doc whose p.1 mentions the named law
        # (avoids short enactment notices that merely reference the law).
        top_doc_id = pages[main_indices[0]]["doc_id"]
        # Extract a multi-word law name: look for phrase "[Adj*] <CapWord>+ Law" in question
        # Use longer match for better discrimination (e.g. "General Partnership Law" > "Partnership Law")
        _law_phrase_re = re.compile(
            r"\b((?:[A-Z][a-z]+\s+){1,4}(?:Law|Regulations?))(?:\s+(?:Amendment\s+)?(?:Law|Regulations?))?\b"
        )
        best_law_name: str = ""
        for lm in _law_phrase_re.finditer(question):
            phrase = lm.group(0).strip()
            if len(phrase) > len(best_law_name):
                best_law_name = phrase
        if not best_law_name:
            # Fallback to _LAW_REF_RE
            named_law_m = _LAW_REF_RE.search(question)
            if named_law_m:
                best_law_name = named_law_m.group(0)
        if best_law_name:
            law_name = best_law_name.lower()
            best_doc: str = ""
            best_match_len: int = 0
            best_page_count: int = 0
            for p in pages:
                if p["page_number"] == 1:
                    text_lower = p["text"].lower()
                    # Score: length of matching prefix in title line
                    if law_name in text_lower:
                        cnt = _doc_page_counts[p["doc_id"]]
                        # Prefer exact match in title (first 100 chars) over body mentions
                        in_title = law_name in text_lower[:100]
                        score = (int(in_title) * 1000) + cnt
                        if score > best_match_len:
                            best_match_len = score
                            best_doc = p["doc_id"]
                            best_page_count = cnt
            if best_doc:
                top_doc_id = best_doc
        article_priority_pages: list[int] = []
        for idx, p in enumerate(pages):
            if p["doc_id"] == top_doc_id and article_pat.search(p["text"]):
                tagged = dict(p)
                tagged["_priority"] = True
                article_priority_pages.append(idx)
                priority.append(idx)
                result[idx] = tagged
                if idx not in top_indices:
                    top_indices.append(idx)
                # After inserting the first article-match page, immediately add its
                # successor (article sub-clauses often continue onto the next page).
                # Always overwrite result entry in case BM25 added it without _priority.
                if len(article_priority_pages) == 1:
                    next_key = (p["doc_id"], p["page_number"] + 1)
                    next_j = doc_page_index.get(next_key)
                    if next_j is not None:
                        ntagged = dict(pages[next_j])
                        ntagged["_priority"] = True
                        priority.append(next_j)
                        result[next_j] = ntagged  # overwrite even if already present
                        if next_j not in top_indices:
                            top_indices.append(next_j)
                if len(priority) >= 5:
                    break

    # "Administered by" questions: pin pages whose text explicitly states
    # which entity administers the law (these are often on page 4-5, not p.1).
    # Run BEFORE law-identity so admin pages take priority over title pages
    # (with max_per_doc=2, earlier priority pages fill slots first).
    # Also search the full corpus in case BM25 found enactment notices instead
    # of the main law documents for sub-queries.
    if re.search(r"\badministered by\b", question, re.IGNORECASE):
        _admin_pat = re.compile(r"\badministered by\b|\badministration of this law\b", re.IGNORECASE)
        # First: tag already-retrieved pages
        for idx in list(result.keys()):
            if _admin_pat.search(result[idx]["text"]):
                if not result[idx].get("_priority"):
                    tagged = dict(result[idx])
                    tagged["_priority"] = True
                    result[idx] = tagged
                    priority.append(idx)
        # Second: for each sub_q law, find the main law doc and search for admin pages.
        # (BM25 may have found the enactment notice instead of the main law doc.)
        for sq in (sub_queries if sub_queries else [question]):
            sq_lower = sq.lower()
            # Find the largest doc whose p.1 contains the law reference
            best_law_doc = ""
            best_cnt = 0
            for p in pages:
                if p["page_number"] == 1:
                    t_low = p["text"].lower()
                    if sq_lower in t_low or any(w in t_low for w in sq_lower.split() if len(w) > 3):
                        cnt = _doc_page_counts.get(p["doc_id"], 0)
                        if cnt > best_cnt and cnt > 2:  # prefer substantive docs
                            terms = [w for w in sq_lower.split() if len(w) > 3]
                            if sum(1 for t in terms if t in t_low) >= len(terms) * 0.6:
                                best_cnt = cnt
                                best_law_doc = p["doc_id"]
            if best_law_doc:
                for idx2, p2 in enumerate(pages):
                    if p2["doc_id"] == best_law_doc and _admin_pat.search(p2["text"]):
                        if idx2 not in result or not result[idx2].get("_priority"):
                            tagged2 = dict(p2)
                            tagged2["_priority"] = True
                            priority.append(idx2)
                            result[idx2] = tagged2
                            if idx2 not in top_indices:
                                top_indices.append(idx2)

    # Law-identity questions: include p.1 of the top matching document.
    # Skip when article search already found specific pages (article is more precise).
    # Only for actual metadata questions (law number, title, enactment date).
    # (title pages always have the law number and enactment date)
    # Tag with _priority to prevent cross-encoder from pushing them out.
    _is_metadata_q = bool(re.search(
        r"\b(what is the (?:law )?number|what is the title|official (?:law )?number|"
        r"enacted\b|commencement date|date (?:of|was) .{0,20}enacted|cover page)\b",
        question, re.IGNORECASE,
    ))
    if _is_metadata_q and main_indices and not article_m:
        # Find the best-matching doc: prefer docs whose p.1 title closely matches the law name
        # from the question (avoids picking the wrong law when many laws match BM25).
        top_doc_id = pages[main_indices[0]]["doc_id"]
        _mp_law_re2 = re.compile(
            r"\b(?:Law No\.\s*\d+\s+of\s+\d{4}|(?:[A-Z][a-z]+\s+){1,4}(?:Law|Regulations?))\b"
        )
        law_phrases_q = [m.group(0).lower() for m in _mp_law_re2.finditer(question)]
        if law_phrases_q:
            # Use the longest law phrase (most specific)
            best_law_q = max(law_phrases_q, key=len)
            best_doc2: str = ""
            best_score2: int = 0
            for p in pages:
                if p["page_number"] == 1:
                    text_lower = p["text"].lower()
                    if best_law_q in text_lower:
                        # "in_title": law appears before any "as amended by" / "consolidated" section
                        # This distinguishes the law itself from docs that merely list it as an amendment
                        cutoff = min(text_lower.find("as amended"), text_lower.find("consolidated"))
                        cutoff = max(cutoff, 0) if cutoff >= 0 else 300
                        in_title = best_law_q in text_lower[:cutoff] if cutoff > 10 else best_law_q in text_lower[:100]
                        cnt = _doc_page_counts[p["doc_id"]]
                        score = (int(in_title) * 10000) + cnt
                        if score > best_score2:
                            best_score2 = score
                            best_doc2 = p["doc_id"]
            if best_doc2:
                top_doc_id = best_doc2
        for pg in (1, 2):
            j = doc_page_index.get((top_doc_id, pg))
            if j is not None:
                tagged = dict(pages[j])
                tagged["_priority"] = True
                priority.append(j)
                result[j] = tagged
                if j not in top_indices:
                    top_indices.append(j)

    # "Same law number" / definition questions: pin pages that define "Law" with
    # a law number (e.g. '"Law" means the Strata Title Law DIFC Law No. 5 of 2007').
    # These pages are critical for comparing what law a term refers to across docs.
    if re.search(r"\bsame law number\b|\brefer to the same law\b|\blaw number\b", question, re.IGNORECASE):
        _lawdef_pat = re.compile(r'"[Ll]aw"\s+means\b|[Ll]aw\s+means\s+the\b|\bLaw No\.\s*\d', re.IGNORECASE)
        for idx in list(result.keys()):
            if _lawdef_pat.search(result[idx]["text"]):
                if not result[idx].get("_priority"):
                    tagged = dict(result[idx])
                    tagged["_priority"] = True
                    result[idx] = tagged
                    priority.append(idx)

    # "Enactment notice" questions: ensure at least one actual enactment-notice
    # page (1-2 page doc) is in the result so the LLM can reason about commencement text.
    if re.search(r"\benactment notice\b", question, re.IGNORECASE):
        _en_pat = re.compile(r"ENACTMENT NOTICE", re.IGNORECASE)
        found_en = False
        # Tag already-retrieved 1-page enactment notice docs as priority
        for idx in list(result.keys()):
            if _doc_page_counts.get(result[idx]["doc_id"], 99) <= 2 and _en_pat.search(result[idx]["text"]):
                if not result[idx].get("_priority"):
                    tagged = dict(result[idx])
                    tagged["_priority"] = True
                    result[idx] = tagged
                    priority.append(idx)
                found_en = True
        # If no enactment notice in result, add the first one found in corpus
        if not found_en:
            for p in pages:
                if p["page_number"] == 1 and _doc_page_counts.get(p["doc_id"], 99) <= 2:
                    if _en_pat.search(p["text"]):
                        j = doc_page_index.get((p["doc_id"], 1))
                        if j is not None:
                            tagged = dict(pages[j])
                            tagged["_priority"] = True
                            priority.append(j)
                            result[j] = tagged
                            if j not in top_indices:
                                top_indices.append(j)
                            break

    # Keyword-within-document search: find pages in top doc that contain
    # content words from the question (for cases where TOC pages dominate BM25).
    # Only run when no article-specific match and the top doc seems law-like.
    if not article_m and main_indices:
        # Extract content terms: non-stopword, non-case-ref, alphabetic tokens >= 4 chars
        q_tokens = [
            t for t in re.findall(r"[a-zA-Z]{4,}", question.lower())
            if t not in _STOPWORDS
        ]
        if len(q_tokens) >= 2:
            top_doc_id = pages[main_indices[0]]["doc_id"]
            # Check if top doc is a multi-page law/legal doc
            doc_pages_count = sum(1 for p in pages if p["doc_id"] == top_doc_id)
            if doc_pages_count >= 4:
                # Find pages in this doc that match multiple query terms
                for idx, p in enumerate(pages):
                    if p["doc_id"] == top_doc_id:
                        text_lower = p["text"].lower()
                        matches = sum(1 for t in q_tokens if t in text_lower)
                        if matches >= max(2, len(q_tokens) // 2):
                            if idx not in result:
                                result[idx] = pages[idx]
                                top_indices.append(idx)

    # Case-specific questions: for short case documents (<= 5 pages),
    # always include all pages when the question references the case number.
    # Tag them with _priority so the reranker won't drop them.
    case_refs = _CASE_RE.findall(question)
    # Extract question content words for case page relevance scoring
    _case_q_words = set(
        w.lower() for w in re.findall(r"[a-zA-Z]{4,}", question)
        if w.lower() not in _STOPWORDS
    )
    for case_ref in case_refs[:2]:
        # Find doc with this case ref in first page text
        for idx, p in enumerate(pages):
            if p["page_number"] == 1 and case_ref.replace(" ", "") in p["text"].replace(" ", ""):
                case_doc_id = p["doc_id"]
                case_pages_list = [j for j, pp in enumerate(pages) if pp["doc_id"] == case_doc_id]
                if len(case_pages_list) <= 6:
                    # Small case doc: include all pages as priority
                    for j in case_pages_list:
                        tagged = dict(pages[j])
                        tagged["_priority"] = True
                        priority.append(j)
                        result[j] = tagged
                        if j not in top_indices:
                            top_indices.append(j)
                else:
                    # Larger case doc: include p.1 as priority, add keyword-matching pages
                    j0 = doc_page_index.get((case_doc_id, 1))
                    if j0 is not None:
                        tagged = dict(pages[j0])
                        tagged["_priority"] = True
                        priority.append(j0)
                        result[j0] = tagged
                        if j0 not in top_indices:
                            top_indices.append(j0)
                    # Find pages matching question keywords (e.g. "conclusion", "ordered")
                    for j in case_pages_list:
                        page_text_l = pages[j]["text"].lower()
                        kw_matches = sum(1 for w in _case_q_words if w in page_text_l)
                        if kw_matches >= 2:
                            if j not in result:
                                result[j] = pages[j]
                                top_indices.append(j)
                break

    # Fine/penalty questions: find the schedule page with fine amounts
    if re.search(r"\b(maximum fine|penalty|fine for|liable to a fine)\b", question, re.IGNORECASE):
        # For the top-matching law doc, find schedule/fine pages
        if main_indices:
            top_doc_id = pages[main_indices[0]]["doc_id"]
            # Try routed doc first
            for ref in _LAW_REF_RE.findall(question):
                for key, doc_id in doc_route["law"].items():
                    if ref.lower() in key.lower() or key.lower() in ref.lower():
                        top_doc_id = doc_id
                        break
            _fine_pat = re.compile(r"\bschedule\b.*\bfine|fine.*\bschedule\b|\bmaximum fine\b|\bfines and fees\b", re.IGNORECASE)
            for idx2, p2 in enumerate(pages):
                if p2["doc_id"] == top_doc_id and _fine_pat.search(p2["text"]):
                    if idx2 not in result or not result.get(idx2, {}).get("_priority"):
                        tagged2 = dict(p2)
                        tagged2["_priority"] = True
                        priority.append(idx2)
                        result[idx2] = tagged2
                        if idx2 not in top_indices:
                            top_indices.append(idx2)

    # Cross-document search: for "which laws were amended by X" questions,
    # find all docs whose title page (p.1) mentions the amending law.
    _amend_q = re.search(
        r"\b(?:amended|enacted|modified|introduced)\s+by\s+(.{10,60})\b",
        question, re.IGNORECASE,
    )
    if not _amend_q:
        _amend_q = re.search(
            r"\b(DIFC Law No\.\s*\d+\s+of\s+\d{4})\b", question, re.IGNORECASE,
        )
    if _amend_q and re.search(r"\bwhich\b|\bwhat\b|\blist\b", question, re.IGNORECASE):
        amend_ref = _amend_q.group(1).strip().lower()
        # Search all doc title pages for this reference
        for idx, p in enumerate(pages):
            if p["page_number"] == 1 and amend_ref in p["text"].lower():
                if idx not in result:
                    tagged = dict(pages[idx])
                    tagged["_priority"] = True
                    priority.append(idx)
                    result[idx] = tagged
                    if idx not in top_indices:
                        top_indices.append(idx)

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
    return [result[i] for i in all_indices if i in result]
