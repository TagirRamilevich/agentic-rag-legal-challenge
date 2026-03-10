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


def rerank_pages(
    pages: list[dict],
    query: str,
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    if not _AVAILABLE or not pages:
        return pages[:top_k]
    m = _model(model_name)
    pairs = [(query, p["text"][:512]) for p in pages]
    scores = m.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, pages), key=lambda x: -float(x[0]))
    return [p for _, p in ranked[:top_k]]
