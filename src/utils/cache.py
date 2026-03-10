import json
import os
import pickle


def cache_exists(path: str) -> bool:
    return os.path.exists(path)


def _ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(data, path: str):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data, path: str):
    _ensure_parent(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
