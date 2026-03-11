def chunk_pages(
    pages: list[dict],
    chunk_tokens: int = 400,
    overlap_tokens: int = 100,
) -> list[dict]:
    if chunk_tokens <= 0:
        return pages

    step = max(1, chunk_tokens - overlap_tokens)
    chunks = []

    for page in pages:
        words = page["text"].split()
        if len(words) <= chunk_tokens:
            chunks.append(
                {
                    "doc_id": page["doc_id"],
                    "page_number": page["page_number"],
                    "chunk_index": 0,
                    "text": page["text"],
                }
            )
            continue

        for i, start in enumerate(range(0, len(words), step)):
            window = words[start : start + chunk_tokens]
            if not window:
                break
            chunks.append(
                {
                    "doc_id": page["doc_id"],
                    "page_number": page["page_number"],
                    "chunk_index": i,
                    "text": " ".join(window),
                }
            )

    return chunks
