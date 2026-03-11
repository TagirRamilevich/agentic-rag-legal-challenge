import time

MODEL_NAME = "bm25+rerank+claude"


def build_telemetry(
    used_pages: list[dict],
    ttft_ms: int,
    total_ms: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model_name: str = "",
) -> dict:
    """Build submission telemetry dict from timing and retrieval data."""
    by_doc: dict[str, set] = {}
    for page in used_pages:
        doc_id = page["doc_id"]
        if doc_id.endswith(".pdf"):
            doc_id = doc_id[:-4]
        by_doc.setdefault(doc_id, set()).add(page["page_number"])

    retrieved_chunk_pages = [
        {"doc_id": doc_id, "page_numbers": sorted(pns)}
        for doc_id, pns in sorted(by_doc.items())
    ]

    return {
        "timing": {
            "ttft_ms": max(1, ttft_ms),
            "tpot_ms": 0,
            "total_time_ms": max(1, total_ms),
        },
        "retrieval": {"retrieved_chunk_pages": retrieved_chunk_pages},
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "model_name": model_name or MODEL_NAME,
    }
