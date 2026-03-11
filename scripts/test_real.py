"""
Test pipeline on the real public dataset from reference/.
No ground-truth answers available, so we show:
  - Retrieval stats (how many questions get non-null retrieval)
  - Answer type distribution
  - Sample answers for manual review
  - TTFT distribution

Usage:
    python scripts/test_real.py [--no-llm] [--no-rerank] [--limit N] [--verbose]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.ingest import ingest_corpus
from src.pipeline.index import get_or_build_index
from src.pipeline.retrieve import retrieve_pages, is_comparison_question
from src.pipeline.rerank import rerank_pages
from src.pipeline.llm import answer_with_llm
from src.pipeline.answer import answer_question

_PHASE_CONFIG = {
    "warmup": {
        "docs_dir": "reference/dataset_documents",
        "cache_dir": "data/reference_test/.cache",
        "questions_path": "reference/public_dataset.json",
    },
    "final": {
        "docs_dir": "docs_corpus/final",
        "cache_dir": "data/final/.cache",
        "questions_path": "data/final/questions.json",
    },
}


def main(use_llm: bool = True, use_rerank: bool = True, limit: int = 0, verbose: bool = False, phase: str = "warmup"):
    cfg = _PHASE_CONFIG[phase]
    DOCS_DIR = cfg["docs_dir"]
    CACHE_DIR = cfg["cache_dir"]
    QUESTIONS_PATH = cfg["questions_path"]

    os.makedirs(CACHE_DIR, exist_ok=True)
    pages_cache = os.path.join(CACHE_DIR, "pages.json")
    index_cache = os.path.join(CACHE_DIR, "index.pkl")

    print(f"Phase: {phase} | Ingesting {DOCS_DIR} ...")
    pages = ingest_corpus(DOCS_DIR, pages_cache)
    print(f"  {len(pages)} pages from {len(set(p['doc_id'] for p in pages))} docs")

    print("Building / loading BM25 index ...")
    result = get_or_build_index(pages, index_cache)
    bm25, pages = result[0], result[1]
    embeddings = result[2] if len(result) > 2 else None

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    # Pre-warm reranker model so first-question TTFT isn't distorted
    if use_rerank:
        from src.pipeline.rerank import rerank_pages as _rp
        _rp([{"text": "warmup", "doc_id": "x", "page_number": 1}], "warmup", top_k=1)

    print(f"Answering {len(questions)} questions (llm={use_llm}, rerank={use_rerank}) ...\n")

    results = []
    ttfts = []
    null_count = 0
    retrieved_count = 0
    type_null = defaultdict(int)
    type_total = defaultdict(int)

    for q in questions:
        t0 = time.perf_counter()
        q_id = q.get("question_id") or q.get("id", "")
        answer_type = q["answer_type"]

        retrieved = retrieve_pages(bm25, pages, q["question"], top_k=15, add_neighbors=True, answer_type=answer_type, embeddings=embeddings)

        is_cmp = is_comparison_question(q["question"])
        # Only rerank comparison questions: they need cross-encoder to
        # properly rank pages from different docs with max_per_doc diversity.
        # Non-comparison: BM25+embedding hybrid + priority pages is sufficient.
        need_rerank = use_rerank and is_cmp
        if need_rerank:
            final = rerank_pages(retrieved, q["question"], top_k=5, max_per_doc=2 if is_cmp else 0)
        else:
            final = retrieved[:5]

        if use_llm:
            answer, used_pages, ttft_ms, total_ms_q, *_ = answer_with_llm(q, final, t0=t0, is_comparison=is_cmp)
        else:
            answer, used_pages = answer_question(q, final)
            ttft_ms = int((time.perf_counter() - t0) * 1000)
            total_ms_q = ttft_ms

        ttfts.append(ttft_ms)

        type_total[answer_type] += 1
        if answer is None or (isinstance(answer, list) and not answer):
            null_count += 1
            type_null[answer_type] += 1
        if used_pages:
            retrieved_count += 1

        results.append({
            "id": q_id,
            "question": q["question"],
            "answer_type": answer_type,
            "answer": answer,
            "used_pages": [{"doc_id": p["doc_id"], "page_number": p["page_number"]} for p in used_pages],
            "ttft_ms": ttft_ms,
        })

        if verbose:
            ans_str = str(answer)[:80] if answer is not None else "NULL"
            doc = used_pages[0]["doc_id"][:20] + "..." if used_pages else "—"
            print(f"  [{answer_type:8s}] {q['question'][:55]:<55s} → {ans_str} | src:{doc} ttft={ttft_ms}ms total={total_ms_q}ms")

    n = len(questions)
    avg_ttft = sum(ttfts) / n if ttfts else 0
    p50 = sorted(ttfts)[n // 2] if ttfts else 0
    p95 = sorted(ttfts)[int(n * 0.95)] if ttfts else 0
    ttft_bonus = "+5%" if avg_ttft < 1000 else "+2%" if avg_ttft < 2000 else "0%" if avg_ttft < 3000 else f"-{int((avg_ttft/1000-3)*5)}%"

    print("\n" + "=" * 65)
    print(f"QUESTIONS: {n}  |  NULL answers: {null_count} ({100*null_count/n:.1f}%)")
    print(f"Non-null with sources: {retrieved_count} ({100*retrieved_count/n:.1f}%)")

    print("\nBY ANSWER TYPE:")
    print(f"  {'type':12s} {'total':>6s} {'null':>6s} {'null%':>7s}")
    for atype in sorted(type_total):
        total = type_total[atype]
        nulls = type_null[atype]
        print(f"  {atype:12s} {total:>6d} {nulls:>6d} {100*nulls/total:>6.0f}%")

    print(f"\nTTFT:  avg={avg_ttft:.0f}ms  p50={p50}ms  p95={p95}ms  → bonus: {ttft_bonus}")

    print("\nSAMPLE ANSWERS (first 5 per type):")
    shown = defaultdict(int)
    for r in results:
        if shown[r["answer_type"]] >= 3:
            continue
        shown[r["answer_type"]] += 1
        ans_str = str(r["answer"])[:100]
        src = r["used_pages"][0]["doc_id"][:30] + "..." if r["used_pages"] else "—"
        print(f"  [{r['answer_type']:8s}] Q: {r['question'][:55]}")
        print(f"             A: {ans_str}")
        print(f"             src: {src}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Test only first N questions")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--phase", default="warmup", choices=["warmup", "final"])
    args = parser.parse_args()
    main(use_llm=not args.no_llm, use_rerank=not args.no_rerank, limit=args.limit, verbose=args.verbose, phase=args.phase)
