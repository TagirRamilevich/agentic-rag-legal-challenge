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


def retrieve_pages(
    bm25,
    pages: list[dict],
    question: str,
    top_k: int = 20,
    add_neighbors: bool = True,
) -> list[dict]:
    tokens = expand_query(tokenize(question))
    scores = bm25.get_scores(tokens)

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_indices = [i for i in ranked[:top_k] if scores[i] > 0]
    if not top_indices:
        top_indices = ranked[: min(5, len(ranked))]

    result = {i: pages[i] for i in top_indices}

    if add_neighbors:
        doc_page_index = {(p["doc_id"], p["page_number"]): j for j, p in enumerate(pages)}
        for i in list(top_indices[:5]):
            page = pages[i]
            for delta in (-1, 1):
                key = (page["doc_id"], page["page_number"] + delta)
                j = doc_page_index.get(key)
                if j is not None and j not in result:
                    result[j] = pages[j]

    return list(result.values())
