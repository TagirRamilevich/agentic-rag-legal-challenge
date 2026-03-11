import time

MODEL_NAME = "bm25+rules-v1"


def build_telemetry(start_time: float, used_pages: list[dict]) -> dict:
    elapsed_ms = max(1, int((time.perf_counter() - start_time) * 1000))

    by_doc: dict[str, set] = {}
    for page in used_pages:
        # Platform expects doc_id without file extension (e.g. "abc123", not "abc123.pdf")
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
            "ttft_ms": elapsed_ms,
            "tpot_ms": 0,
            "total_time_ms": elapsed_ms,
        },
        "retrieval": {"retrieved_chunk_pages": retrieved_chunk_pages},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "model_name": MODEL_NAME,
    }
