try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_CACHE: dict = {}


def _model(name: str):
    if name not in _CACHE:
        _CACHE[name] = _CrossEncoder(name)
    return _CACHE[name]


def _apply_diversity(pages: list[dict], top_k: int, max_per_doc: int) -> list[dict]:
    """Return up to top_k pages, capping each doc_id at max_per_doc."""
    doc_counts: dict = {}
    result = []
    for page in pages:
        did = page["doc_id"]
        if doc_counts.get(did, 0) < max_per_doc:
            result.append(page)
            doc_counts[did] = doc_counts.get(did, 0) + 1
        if len(result) >= top_k:
            break
    return result


def rerank_pages(
    pages: list[dict],
    query: str,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_per_doc: int = 0,
) -> list[dict]:
    if not pages:
        return []

    if not _AVAILABLE:
        # No cross-encoder: use BM25 ordering; apply diversity if requested
        if max_per_doc > 0:
            return _apply_diversity(pages, top_k, max_per_doc)
        return pages[:top_k]

    # Limit reranker input: too many pages → slow CPU inference
    # Keep at most 3×top_k pages for scoring; priority pages always included first
    priority_pages = [p for p in pages if p.get("_priority")]
    non_priority = [p for p in pages if not p.get("_priority")]
    max_non_priority = max(top_k * 3, 12) - len(priority_pages)
    input_pages = priority_pages + non_priority[:max_non_priority]

    m = _model(model_name)
    pairs = [(query, p["text"][:512]) for p in input_pages]
    scores = list(m.predict(pairs, show_progress_bar=False))
    # Priority pages are pinned regardless of score
    for i, page in enumerate(input_pages):
        if page.get("_priority"):
            scores[i] = 1000.0
    ranked = [page for _, page in sorted(zip(scores, input_pages), key=lambda x: -float(x[0]))]

    if max_per_doc > 0:
        return _apply_diversity(ranked, top_k, max_per_doc)

    return ranked[:top_k]
