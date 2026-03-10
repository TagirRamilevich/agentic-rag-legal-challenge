Participant Guide
About the Challenge
Agentic RAG Legal Challenge 2026 is an international engineering competition focused on building production-grade Retrieval-Augmented Generation (RAG) systems for the legal domain. The challenge runs for two weeks and culminates in winner announcements at Machines Can See 2026 in Dubai.

This is not a hackathon — it is a benchmark-style competition with objective evaluation, private test sets, telemetry, and strict anti-gaming mechanisms.

For technical details (scoring formulas, API format, telemetry schema) refer to the starter kit: EVALUATION.md, API.md, and README.md.

Your Goal
Build a RAG system that maximizes:

Legal accuracy

Grounded retrieval quality

Low latency (TTFT)

Robust document ingestion

Faithfulness / no hallucinations

Production realism

2. What You Will Receive
Participants are provided with:

Document corpus
~300 public legal documents (regulations, case law and etc.), in varied formats.

Demo set
30 documents + 100 public questions for pipeline debugging.

Final evaluation set
900 private questions across the full corpus.

Submission API
Pull-model interface where your system fetches questions, processes them locally, and submits answers with telemetry.

Dataset structure
As defined in the dataset spec document:
deterministic answers (numbers, names, booleans, lists, date) + assistant-style free-text answers (max 280 chars).

3. What You Need to Build
A fully functional RAG pipeline. It must include:

1. Document Ingestion & Parsing
Legal documents come in heterogeneous formats. You must handle:

PDF → text extraction (including OCR for scanned documents)

Mixed formatting and structural inconsistencies

Long “stress test” documents

Clause-level segmentation and hierarchy detection

Metadata extraction (titles, sections, case numbers, dates)

Your ingestion stage is critical — evaluation heavily depends on retrieval grounding.

2. Indexing & Chunking
Chunking must respect legal structure, not fixed token windows.
Recommended components:

clause-aware or heading-aware segmentation

dense embeddings + re-ranking

hybrid search (BM25 + semantic)

3. Retrieval
Your system must return:

relevant chunks

minimal noise

high recall (missing evidence leads to penalties)

Every answer must include retrieved_chunk_pages — otherwise retrieval score becomes zero.

4. Generation
Two groups of questions:

Deterministic factual questions
Answer types: number, boolean, name, names, date. If the answer is not present in the corpus, return JSON null.

Free-text assistant questions
Up to 280 characters, legally faithful, concise, well-grounded.

5. Telemetry
Every answer must include:

ttft_ms

token usage

retrieved chunks

total runtime

Missing telemetry → −10% penalty for that answer.

4. Evaluation
Your solution is scored across four dimensions:

1. Deterministic Accuracy
Simple, strict rules:

numeric tolerance (±1%)

exact match for booleans and names

Jaccard similarity for lists

ISO 8601 exact match for dates

JSON null for absent information

(See starter kit — EVALUATION.md — for full details on scoring rules.)

2. Free-Text Quality (LLM Judge)
Each answer is scored on 5 criteria:

Correctness

Completeness

Grounding

Confidence calibration

Clarity & conciseness

Evaluation uses a cascade of multiple LLMs for consistency.

3. Retrieval / Grounding Score
Penalties for:

irrelevant pages (noise)

missing required evidence (recall)

Balanced to reward precise, minimal, faithful retrieval.

4. Latency (TTFT Modifier)
Speed is part of the score:

<1s → +5% bonus

1–2s → +2%

2–3s → no modifier

>3s → penalty up to −15%

5. Submission Workflow
Fill the registration form in the Discord #welcome-challenge channel. 

After moderator verifies your submission, you will get login and password to competition platform.

Connect your system to the pull-model API.

Fetch questions in batches or streaming mode.

Run your pipeline locally.

Submit answers with telemetry for each question.

Provide a short Architecture Summary describing your models and retrieval strategy (for transparency and post-competition publication).

6. Rules & Requirements
Mandatory
Only public APIs and public models may be used.

Telemetry required for every answer.

No manual answering or partial automation.

No leaking or sharing private questions.

Allowed
Any embedding model or search engine accessible via public API.

Model ensembles and hybrid pipelines.

Local preprocessing of documents.

Custom re-rankers.

Prohibited
Hardcoding answers.

Synthetic leakage

Manually editing logs or telemetry.

7. Recommendations for a Competitive Solution
Focus on retrieval precision
Legal RAG systems fail on irrelevant context. Use hybrid retrievers, re-ranking, clause-aware chunking

Optimize for TTFT
Consider:

fast small models for retrieval

streaming generation

caching for long documents

batching efficiently

Avoid hallucinations
For many questions the correct answer is JSON null (deterministic types) or a natural-language statement such as "There is no information on this question in the provided documents." (free_text). Return an empty retrieved_chunk_pages array in both cases.

Telemetry correctness matters
Malformed telemetry destroys your score even if the answer is good.

8. Prize Categories
Expected categories include:

1st–3rd overall places

Speed Champion (lowest TTFT)

Efficiency Expert (best score/token ratio)

Retrieval Master (highest grounding score)

Best Publication (blogpost/video)

Teams may win multiple prizes.

9. Final Advice
This challenge rewards engineering rather than brute force.
Strong teams typically:

build robust ingestion pipelines

chunk carefully

test retrieval thoroughly

optimize latency pragmatically

keep answers short, grounded, and legally faithful

verify telemetry early and often

If you treat this like a real production RAG system, you will perform well.