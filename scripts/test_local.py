"""
Local test harness: creates synthetic legal PDF, runs the full pipeline,
and reports retrieval recall + extraction accuracy.

Usage:
    python scripts/test_local.py [--no-llm] [--no-rerank] [--verbose]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import fitz

from src.pipeline.ingest import ingest_corpus
from src.pipeline.index import get_or_build_index
from src.pipeline.retrieve import retrieve_pages, is_comparison_question
from src.pipeline.rerank import rerank_pages
from src.pipeline.llm import answer_with_llm
from src.pipeline.answer import answer_question


DOCS_DIR = "data/local_test/docs"
CACHE_DIR = "data/local_test/.cache"
PDF_NAME = "synthetic_contract.pdf"

TEST_CASES = [
    {
        "question_id": "t01",
        "question": "What is the total contract value?",
        "answer_type": "number",
        "expected": 2500000,
        "expected_page": 1,
    },
    {
        "question_id": "t02",
        "question": "What is the annual service fee?",
        "answer_type": "number",
        "expected": 150000,
        "expected_page": 2,
    },
    {
        "question_id": "t03",
        "question": "What is the penalty rate for late payment?",
        "answer_type": "number",
        "expected": 5,
        "expected_page": 3,
    },
    {
        "question_id": "t04",
        "question": "When was the agreement signed?",
        "answer_type": "date",
        "expected": "2024-03-15",
        "expected_page": 1,
    },
    {
        "question_id": "t05",
        "question": "When does the contract term expire?",
        "answer_type": "date",
        "expected": "2027-03-15",
        "expected_page": 2,
    },
    {
        "question_id": "t06",
        "question": "Is the agreement subject to automatic renewal?",
        "answer_type": "bool",
        "expected": True,
        "expected_page": 2,
    },
    {
        "question_id": "t07",
        "question": "Is the contractor permitted to subcontract work without approval?",
        "answer_type": "bool",
        "expected": False,
        "expected_page": 3,
    },
    {
        "question_id": "t08",
        "question": "What is the name of the service provider?",
        "answer_type": "name",
        "expected": "Acme Solutions Ltd",
        "expected_page": 1,
    },
    {
        "question_id": "t09",
        "question": "Who is the governing authority for dispute resolution?",
        "answer_type": "name",
        "expected": "London Court of International Arbitration",
        "expected_page": 4,
    },
    {
        "question_id": "t10",
        "question": "Who are the authorized signatories of the agreement?",
        "answer_type": "names",
        "expected": ["John Smith", "Mary Johnson"],
        "expected_page": 1,
    },
    {
        "question_id": "t11",
        "question": "What services are included in the scope of work?",
        "answer_type": "names",
        "expected": ["Software Development", "Quality Assurance", "Technical Support"],
        "expected_page": 2,
    },
    {
        "question_id": "t12",
        "question": "What are the confidentiality obligations of the contractor?",
        "answer_type": "free_text",
        "expected": None,
        "expected_page": 4,
    },
    {
        "question_id": "t13",
        "question": "What is the maximum liability cap in this agreement?",
        "answer_type": "number",
        "expected": 5000000,
        "expected_page": 3,
    },
]

PAGES_CONTENT = {
    1: """SERVICE AGREEMENT

This Service Agreement (the "Agreement") is entered into on 15 March 2024 by and between:

Acme Solutions Ltd, a company incorporated under the laws of England and Wales (the "Service Provider"),
represented by John Smith, Chief Executive Officer

and

Global Enterprises Inc, a corporation organized under Delaware law (the "Client"),
represented by Mary Johnson, Chief Operating Officer.

1. TOTAL CONTRACT VALUE
The total contract value for all services rendered under this Agreement shall be USD 2,500,000
(two million five hundred thousand United States Dollars).

2. PAYMENT TERMS
Payment shall be made in quarterly installments of USD 625,000 each.
""",
    2: """3. TERM AND RENEWAL
The Agreement shall commence on 15 March 2024 and shall continue for a period of three (3) years,
expiring on 15 March 2027, unless earlier terminated.

The Agreement is subject to automatic renewal for successive one-year periods unless either party
provides written notice of non-renewal at least ninety (90) days prior to expiration.
Automatic renewal: YES.

4. SCOPE OF SERVICES
The Service Provider shall provide the following services:
- Software Development
- Quality Assurance
- Technical Support
- Project Management

5. SERVICE FEES
The annual service fee shall be USD 150,000 (one hundred fifty thousand United States Dollars),
payable on the first business day of each contract year.
""",
    3: """6. LATE PAYMENT PENALTY
In the event of late payment, the Client shall be liable for a penalty interest rate of 5%
per annum on the outstanding amount, calculated daily.

7. SUBCONTRACTING
The contractor is not permitted to subcontract work without prior written approval from the Client.
Any unauthorized subcontracting shall constitute a material breach of this Agreement.

8. LIABILITY CAP
The maximum aggregate liability of either party under this Agreement shall not exceed
USD 5,000,000 (five million United States Dollars) in any twelve-month period.

9. TERMINATION
Either party may terminate this Agreement upon 90 days written notice.
""",
    4: """10. CONFIDENTIALITY
The Contractor shall maintain strict confidentiality with respect to all proprietary information,
trade secrets, and business data disclosed by the Client. The Contractor shall not disclose,
reproduce, or use such confidential information for any purpose other than the performance of
services under this Agreement. These obligations shall survive termination for a period of five years.

11. DISPUTE RESOLUTION
Any disputes arising under this Agreement shall be resolved by binding arbitration administered by
the London Court of International Arbitration, in accordance with its rules.
The seat of arbitration shall be London, England.

12. GOVERNING LAW
This Agreement shall be governed by the laws of England and Wales.
""",
}


def create_synthetic_pdf(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = fitz.open()
    for page_num in sorted(PAGES_CONTENT.keys()):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 50), PAGES_CONTENT[page_num], fontsize=11)
    doc.save(path)
    doc.close()
    print(f"Created synthetic PDF: {path} ({len(PAGES_CONTENT)} pages)")


def _check_number(got, expected) -> bool:
    if got is None or expected is None:
        return got == expected
    try:
        return abs(float(got) - float(expected)) / max(abs(float(expected)), 1) <= 0.01
    except (TypeError, ValueError):
        return False


def _check_names(got, expected) -> bool:
    if not got or not expected:
        return False
    got_set = {str(x).lower().strip() for x in got}
    exp_set = {str(x).lower().strip() for x in expected}
    intersection = got_set & exp_set
    union = got_set | exp_set
    return len(intersection) / len(union) >= 0.5 if union else False


def _check_answer(got, expected, answer_type: str) -> bool:
    if answer_type == "number":
        return _check_number(got, expected)
    if answer_type in ("bool", "boolean"):
        return got == expected
    if answer_type == "date":
        return str(got) == str(expected)
    if answer_type == "name":
        if got is None or expected is None:
            return got == expected
        return str(expected).lower() in str(got).lower() or str(got).lower() in str(expected).lower()
    if answer_type == "names":
        return _check_names(got, expected)
    if answer_type == "free_text":
        return got is not None and len(got) > 10
    return got == expected


def main(use_llm: bool = True, use_rerank: bool = True, verbose: bool = False):
    pdf_path = os.path.join(DOCS_DIR, PDF_NAME)
    if not os.path.exists(pdf_path):
        create_synthetic_pdf(pdf_path)

    os.makedirs(CACHE_DIR, exist_ok=True)
    pages_cache = os.path.join(CACHE_DIR, "pages.json")
    index_cache = os.path.join(CACHE_DIR, "index.pkl")

    pages = ingest_corpus(DOCS_DIR, pages_cache, min_chars=10)
    bm25, pages, embeddings = get_or_build_index(pages, index_cache)

    recall_bm25 = 0
    recall_rerank = 0
    extraction_correct = 0
    extraction_total = 0
    failures = []
    ttfts = []

    type_results: dict[str, list[bool]] = {}

    for tc in TEST_CASES:
        t0 = time.perf_counter()
        retrieved = retrieve_pages(bm25, pages, tc["question"], top_k=20, add_neighbors=True, embeddings=embeddings)

        gold_in_bm25 = any(
            p["doc_id"] == PDF_NAME and p["page_number"] == tc["expected_page"]
            for p in retrieved
        )
        if gold_in_bm25:
            recall_bm25 += 1

        is_cmp = is_comparison_question(tc["question"])
        if use_rerank:
            final = rerank_pages(retrieved, tc["question"], top_k=5, max_per_doc=2 if is_cmp else 0)
        else:
            final = retrieved[:5]

        gold_in_rerank = any(
            p["doc_id"] == PDF_NAME and p["page_number"] == tc["expected_page"]
            for p in final
        )
        if gold_in_rerank:
            recall_rerank += 1

        if use_llm:
            answer, _ = answer_with_llm(tc, final)
        else:
            answer, _ = answer_question(tc, final)

        ttft_ms = int((time.perf_counter() - t0) * 1000)
        ttfts.append(ttft_ms)

        if tc["answer_type"] != "free_text":
            correct = _check_answer(answer, tc["expected"], tc["answer_type"])
            extraction_total += 1
            extraction_correct += int(correct)
            type_results.setdefault(tc["answer_type"], []).append(correct)

            if not correct or verbose:
                failures.append(
                    f"  {tc['question_id']} [{tc['answer_type']}]: "
                    f"expected={tc['expected']!r}, got={answer!r} "
                    f"(gold p{tc['expected_page']} in BM25:{gold_in_bm25} rerank:{gold_in_rerank}) "
                    f"ttft:{ttft_ms}ms"
                )
        elif verbose:
            print(f"  {tc['question_id']} [free_text]: {str(answer)[:80]}")

    n = len(TEST_CASES)
    print("\n" + "=" * 60)
    print("RETRIEVAL RECALL")
    print(f"  @top-20 BM25:    {recall_bm25}/{n}  ({100*recall_bm25/n:.1f}%)")
    rerank_label = "top-5 reranked" if use_rerank else "top-5 (no rerank)"
    print(f"  @{rerank_label}: {recall_rerank}/{n}  ({100*recall_rerank/n:.1f}%)")

    print("\nEXTRACTION ACCURACY (excl. free_text)")
    for atype, results in sorted(type_results.items()):
        c = sum(results)
        print(f"  {atype:12s}: {c}/{len(results)} ({100*c/len(results):.0f}%)")
    if extraction_total:
        print(f"  {'OVERALL':12s}: {extraction_correct}/{extraction_total} ({100*extraction_correct/extraction_total:.1f}%)")

    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    ttft_label = "+5%" if avg_ttft < 1000 else "+2%" if avg_ttft < 2000 else "0%" if avg_ttft < 3000 else "-15%"
    print(f"\nAVG TTFT: {avg_ttft:.0f}ms  →  TTFT bonus: {ttft_label}")

    if failures:
        print("\nFAILURES / DETAILS:")
        for f in failures:
            print(f)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(use_llm=not args.no_llm, use_rerank=not args.no_rerank, verbose=args.verbose)
