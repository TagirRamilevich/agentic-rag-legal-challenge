import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import yaml

from src.pipeline.ingest import ingest_corpus, get_page_counts
from src.pipeline.index import get_or_build_index
from src.pipeline.retrieve import retrieve_pages
from src.pipeline.answer import answer_question
from src.pipeline.telemetry import build_telemetry
from src.utils.json_schema import validate_submission
from arlc import EvaluationClient


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_code_archive(archive_path: str):
    client = EvaluationClient.from_env()
    include = [p for p in ["src", "scripts", "configs", "arlc", "requirements.txt", "CLAUDE.md"] if os.path.exists(p)]
    client.create_code_archive(include, archive_path)
    print(f"Code archive: {archive_path}")


def main(phase: str, config_path: str = "configs/rag.yaml", skip_validate: bool = False):
    cfg = load_config(config_path)

    docs_dir = os.path.join(cfg["docs_dir"], phase)
    questions_path = os.path.join(cfg["data_dir"], phase, "questions.json")
    cache_dir = os.path.join(cfg["cache_dir"], phase)
    os.makedirs(cache_dir, exist_ok=True)

    pages_cache = os.path.join(cache_dir, "pages.json")
    index_cache = os.path.join(cache_dir, "index.pkl")

    print(f"[1/4] Ingesting corpus from {docs_dir} ...")
    pages = ingest_corpus(
        docs_dir,
        pages_cache,
        min_chars=cfg["ingestion"]["min_page_text_chars"],
        ocr_dpi=cfg["ingestion"]["ocr_dpi"],
    )
    print(f"      {len(pages)} pages loaded")

    print("[2/4] Building / loading BM25 index ...")
    bm25, pages = get_or_build_index(pages, index_cache)

    print(f"[3/4] Loading questions from {questions_path} ...")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"      {len(questions)} questions")

    top_k = cfg["retrieval"]["top_k_bm25"]
    add_neighbors = cfg["retrieval"]["neighbor_pages"]
    answers = []

    print("[4/4] Answering questions ...")
    for q in questions:
        t0 = time.perf_counter()
        retrieved = retrieve_pages(bm25, pages, q["question"], top_k=top_k, add_neighbors=add_neighbors)
        answer, used_pages = answer_question(q, retrieved)
        telemetry = build_telemetry(t0, used_pages)
        answers.append({"question_id": q["question_id"], "answer": answer, "telemetry": telemetry})

    submission = {
        "architecture_summary": cfg.get("architecture_summary", "")[:500],
        "answers": answers,
    }

    if not skip_validate:
        page_counts = get_page_counts(docs_dir) if os.path.isdir(docs_dir) else {}
        errors = validate_submission(submission, docs_dir if os.path.isdir(docs_dir) else None, page_counts)
        if errors:
            print("Validation FAILED:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        print(f"Validation passed ({len(answers)} answers)")

    out_path = cfg.get("submission_path", "submission.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    print(f"Submission written to {out_path}")

    archive_path = cfg.get("code_archive_path", "code_archive.zip")
    _build_code_archive(archive_path)

    return out_path, archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["warmup", "final"])
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    main(args.phase, args.config, skip_validate=args.no_validate)
