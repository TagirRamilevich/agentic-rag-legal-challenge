import re

from rank_bm25 import BM25Okapi

from src.utils.cache import cache_exists, save_pickle, load_pickle


def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return tokens if tokens else ["_empty_"]


def build_index(pages: list[dict]) -> BM25Okapi:
    corpus = [tokenize(p["text"]) for p in pages]
    return BM25Okapi(corpus)


def get_or_build_index(pages: list[dict], index_path: str) -> tuple:
    if cache_exists(index_path):
        data = load_pickle(index_path)
        return data["bm25"], data["pages"]
    bm25 = build_index(pages)
    save_pickle({"bm25": bm25, "pages": pages}, index_path)
    return bm25, pages
