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
from src.utils.chunker import chunk_pages
from src.pipeline.retrieve import retrieve_pages, is_comparison_question
from src.pipeline.rerank import rerank_pages
from src.pipeline.llm import answer_with_llm
from src.pipeline.answer import answer_question
from src.pipeline.telemetry import build_telemetry
from src.utils.json_schema import validate_submission
from arlc import EvaluationClient


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_code_archive(archive_path: str):
    client = EvaluationClient.from_env()
    include = [
        p for p in ["src", "scripts", "configs", "arlc", "requirements.txt", "CLAUDE.md"]
        if os.path.exists(p)
    ]
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

    chunk_cfg = cfg.get("chunking", {})
    if chunk_cfg.get("enabled", False):
        print("[2/4] Chunking pages + building BM25 index ...")
        units = chunk_pages(
            pages,
            chunk_tokens=chunk_cfg.get("chunk_tokens", 400),
            overlap_tokens=chunk_cfg.get("overlap_tokens", 100),
        )
        print(f"      {len(units)} chunks from {len(pages)} pages")
    else:
        print("[2/4] Building / loading BM25 index ...")
        units = pages

    bm25, units = get_or_build_index(units, index_cache)

    print(f"[3/4] Loading questions from {questions_path} ...")
    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"      {len(questions)} questions")

    top_k_bm25 = cfg["retrieval"]["top_k_bm25"]
    top_k_rerank = cfg["retrieval"]["top_k_rerank"]
    add_neighbors = cfg["retrieval"]["neighbor_pages"]
    use_reranker = cfg["retrieval"].get("use_reranker", True)
    reranker_model = cfg["retrieval"].get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_llm = cfg["llm"].get("use_llm", True)

    answers = []
    null_count = 0
    llm_count = 0

    # Pre-warm reranker so first question TTFT isn't distorted by model load
    if use_reranker:
        rerank_pages([{"text": "warmup", "doc_id": "x", "page_number": 1}], "warmup", top_k=1, model_name=reranker_model)

    print("[4/4] Answering questions ...")
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()

        retrieved = retrieve_pages(bm25, units, q["question"], top_k=top_k_bm25, add_neighbors=add_neighbors, answer_type=q.get("answer_type", ""))

        is_cmp = is_comparison_question(q["question"])
        _SKIP_RERANK = {"number", "date"}
        need_rerank = use_reranker and (q.get("answer_type") not in _SKIP_RERANK or is_cmp)
        if need_rerank:
            final_pages = rerank_pages(retrieved, q["question"], top_k=top_k_rerank, model_name=reranker_model, max_per_doc=2 if is_cmp else 0)
        else:
            final_pages = retrieved[:top_k_rerank]

        if use_llm:
            answer, used_pages = answer_with_llm(q, final_pages)
            llm_count += 1
        else:
            answer, used_pages = answer_question(q, final_pages)

        if answer is None:
            null_count += 1

        telemetry = build_telemetry(t0, used_pages)
        q_id = q.get("question_id") or q.get("id", "")
        answers.append({"question_id": q_id, "answer": answer, "telemetry": telemetry})

        if i % 10 == 0 or i == len(questions):
            print(f"      {i}/{len(questions)} done  (last ttft {telemetry['timing']['ttft_ms']}ms)")

    print(f"      null answers: {null_count}/{len(questions)}, llm calls: {llm_count}")

    submission = {
        "architecture_summary": cfg.get("architecture_summary", "")[:500],
        "answers": answers,
    }

    if not skip_validate:
        page_counts = get_page_counts(docs_dir) if os.path.isdir(docs_dir) else {}
        errors = validate_submission(
            submission,
            docs_dir if os.path.isdir(docs_dir) else None,
            page_counts,
        )
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
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic-only mode")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    if args.no_llm or args.no_rerank:
        cfg = load_config(args.config)
        if args.no_llm:
            cfg["llm"]["use_llm"] = False
        if args.no_rerank:
            cfg["retrieval"]["use_reranker"] = False
        import tempfile, yaml as _yaml
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            _yaml.dump(cfg, tmp)
            tmp_path = tmp.name
        main(args.phase, tmp_path, skip_validate=args.no_validate)
    else:
        main(args.phase, args.config, skip_validate=args.no_validate)
