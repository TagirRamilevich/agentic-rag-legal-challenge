import os

from src.utils.pdf_text import extract_pages, get_page_count
from src.utils.ocr import ocr_page
from src.utils.cache import cache_exists, save_json, load_json


def ingest_corpus(
    docs_dir: str,
    cache_path: str,
    min_chars: int = 50,
    ocr_dpi: int = 300,
) -> list[dict]:
    if cache_exists(cache_path):
        return load_json(cache_path)

    pages = []
    pdf_files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))

    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_dir, pdf_file)
        try:
            pdf_pages = extract_pages(pdf_path)
        except Exception as exc:
            print(f"  Warning: failed to extract {pdf_file}: {exc}")
            continue

        for page in pdf_pages:
            text = page["text"]
            if len(text) < min_chars:
                try:
                    text = ocr_page(pdf_path, page["page_number"], ocr_dpi)
                except Exception:
                    text = ""
            pages.append(
                {
                    "doc_id": pdf_file,
                    "page_number": page["page_number"],
                    "text": text,
                }
            )

    save_json(pages, cache_path)
    return pages


def get_page_counts(docs_dir: str) -> dict:
    counts = {}
    for f in os.listdir(docs_dir):
        if f.lower().endswith(".pdf"):
            try:
                counts[f] = get_page_count(os.path.join(docs_dir, f))
            except Exception:
                counts[f] = 0
    return counts
