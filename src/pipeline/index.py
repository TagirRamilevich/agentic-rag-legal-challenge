import os
import re
import numpy as np

from rank_bm25 import BM25Okapi

from src.utils.cache import cache_exists, save_pickle, load_pickle

# ---------------------------------------------------------------------------
# Sentence-transformers availability (used for dense embedding retrieval)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _EMBED_AVAILABLE = True
except ImportError:
    _EMBED_AVAILABLE = False

_EMBED_MODEL_CACHE: dict = {}

_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embed_model(model_name: str = _DEFAULT_EMBED_MODEL):
    """Load (and cache) a SentenceTransformer model."""
    if model_name not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE[model_name] = _SentenceTransformer(model_name)
    return _EMBED_MODEL_CACHE[model_name]


# ---------------------------------------------------------------------------
# Tokenizer (for BM25)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return tokens if tokens else ["_empty_"]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_index(pages: list[dict]) -> BM25Okapi:
    corpus = [tokenize(p["text"]) for p in pages]
    return BM25Okapi(corpus)


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

def build_embedding_index(
    pages: list[dict],
    model_name: str = _DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode all page texts into a normalized embedding matrix (N x D).

    Returns a float32 numpy array. Each row is L2-normalized so that
    dot product == cosine similarity.
    """
    model = _get_embed_model(model_name)
    # Truncate texts to first 512 chars to keep encoding fast
    texts = [p["text"][:512] for p in pages]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(
    query: str,
    model_name: str = _DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """Encode a single query string into a normalized embedding vector (1 x D)."""
    model = _get_embed_model(model_name)
    vec = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Hybrid index: BM25 + embeddings
# ---------------------------------------------------------------------------

def get_or_build_index(pages: list[dict], index_path: str) -> tuple:
    """Build or load BM25 index + optional embedding index.

    Returns (bm25, pages, embeddings_or_None).
    Callers that unpack only 2 values will get a backward-compatible tuple
    thanks to the wrapper — but updated callers should unpack 3 values.
    """
    emb_path = index_path.replace(".pkl", "_emb.npy")

    # --- BM25 ---
    if cache_exists(index_path):
        data = load_pickle(index_path)
        bm25, pages = data["bm25"], data["pages"]
    else:
        bm25 = build_index(pages)
        save_pickle({"bm25": bm25, "pages": pages}, index_path)

    # --- Embeddings (optional) ---
    embeddings = None
    if _EMBED_AVAILABLE:
        if cache_exists(emb_path):
            embeddings = np.load(emb_path)
            # Sanity check: if page count changed, rebuild
            if embeddings.shape[0] != len(pages):
                embeddings = build_embedding_index(pages)
                np.save(emb_path, embeddings)
        else:
            print("  Building embedding index (one-time) ...")
            embeddings = build_embedding_index(pages)
            os.makedirs(os.path.dirname(os.path.abspath(emb_path)), exist_ok=True)
            np.save(emb_path, embeddings)
            print(f"  Embedding index saved ({embeddings.shape})")

    return bm25, pages, embeddings
