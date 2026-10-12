# Week 6 Homework Checklist (RAG)

Use this checklist when preparing your submission.

## Current progress (2026-10-12)

This is a live checklist. A checked item below has been verified against the
current code or an actual command. Prince embedding retrieval, reranking,
calibrated refusal, CLI integration, evidence capture, and one supported live
Gemini answer with validated citations have been exercised. The latest
evidence refresh adds sentence-aware chunking and a same-child-index comparison
of multi-vector and parent retrieval. The latest live run used Gemini for a
supported answer with validated citations and evaluated all four RAGAS cases.
Pair accountability is reserved for the students.

## A. Baseline setup

- [x] Project runs locally and tests run in `.venv` (Python 3.12.15; latest full run: 49 passed).
- [x] CLI runs with a custom question: `.venv/bin/python -m src.cli --question "What is retrieval augmented generation?"`.
- [x] Current implementation locations are understood: `src/chunking.py`, `src/embeddings.py`, `src/vector_store.py`, `src/retrieval.py`, `src/reranking.py`, `src/generation.py`, `src/pipeline.py`, and `src/cli.py`.

## B. Outcome 1: Explain chunking effect on retrieval quality

- [x] Ran the starter configuration (`140/30`) and book configuration (`800/120`) on Chapter XV.
- [x] Compared sentence-aware chunking (`800/120`) on the same Chapter XV query.
- [x] Compared retrieval for the same query: `What qualities are praised or blamed in men and princes?`
- [x] Observation: starter produced 27 short chunks and ranked `prince-chapter-xv-chunk-0` first (score 0.307692); book chunking produced 5 larger chunks and ranked the same first chunk (score 0.114865). The starter's second result was chunk 11 (0.222222); the book chunker's was chunk 1 (0.099338). The larger chunks retain more continuous prose, while the starter's second match is another small overlapping fragment. These lexical scores are normalized by chunk token count, so their magnitudes are not directly comparable across chunk sizes; no human-labeled relevance set has been evaluated yet.
- [x] Sentence-aware chunking produced 5 chunks and ranked chunk 0 first (score 0.176471), followed by chunk 2 (0.099237). The second result captures a later list of qualities; it adds context but does not outrank the opening framing. This is a qualitative single-query comparison, not a labeled benchmark.

The actual tokenizer check passed for all 265 book chunks: maximum 192 tokens
including special tokens (target 220). The integration test is marked and uses
the locally cached tokenizer.

## C. Outcome 2: Implement top-k retrieval pipeline

- [x] Prince-specific semantic retrieval returns top-k chunks sorted by relevance score.
- [x] Evidence shows one Prince query with `top_k=2` and one with `top_k=3`.
- [x] Evidence explains cosine similarity and FlashRank reranker scores.
- [x] Evidence explains why Chapters II, I, and III rank in that order for the selected top-k query.
- [x] Compared multi-vector child passages with parent-window expansion over the same child index and ranking models.

## D. Outcome 3: Return answers with explicit citations

- [x] A generated natural-language Prince answer includes explicit chunk-ID citations.
- [x] A successful generated answer with validated citations is shown in evidence.
- [x] CLI output separates the answer, citations, generation error, and retrieved evidence.

The latest supported Chapter II run produced a Gemini-generated synthesis
citing `prince-chapter-ii-chunk-0`, and the pipeline validated that the cited
ID was present in retrieved evidence. The actual answer and retrieved text are
captured in `evidence.md`.

## E. Outcome 4: Refuse unsupported questions

- [x] Prince-specific unsupported questions are refused using a threshold calibrated from labeled examples.
- [x] Refusal remains available without Gemini and the refusal message is clear.
- [x] Evidence explains why retrieval was insufficient and records the scores/threshold.

The legacy `default` and `training-rag` profiles retain their starter lexical
refusal behavior. The Prince profile uses the calibrated FlashRank threshold.

## F. Outcome 5: Show supported and unsupported evidence

- [x] Final Prince evidence includes one supported question, retrieved chunks, generated answer, and validated citations.
- [x] Final Prince evidence includes one unsupported question, retrieved chunks, and refusal output.
- [x] Both captured examples are reproducible from the evidence script and book CLI.

The latest supported example used the configured Gemini model. The latest
RAGAS run generated and scored all four labeled cases.

## G. Pair accountability

- [ ] Pair report includes contribution of both members. (Student-owned; do not auto-fill.)
- [ ] The designated demo lead for this week is recorded. (Student-owned; do not auto-fill.)

## H. Suggested extension (optional)

- [ ] Improve lexical scoring or citation formatting. (Optional; not selected as the agreed extension.)
- [ ] Add at least 3 tests for your extension. (Optional; not applicable unless an extension is selected.)

## I. Submission bundle (recommended)

- [x] `evidence.md` has Outcome 1-5 sections and is generated by `scripts/generate_evidence.py` from actual executions.
- [x] CLI commands and actual outputs for supported and unsupported Prince queries are captured.
- [x] Full test output snapshot is captured.

The original retrieval and refusal evidence is complete. Pair accountability
and optional extension fields remain student-owned or unselected. The latest
RAGAS report is `evidence/ragas-metrics.json`; all four cases were scored.
Mean scores: Faithfulness 1.0000, Answer Relevancy 0.7718, Context Precision
0.8750, and Context Recall 1.0000. These are LLM-judge estimates.

## J. RAGAS evaluation (additional requested work)

- [x] Four reference-backed metrics are implemented: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
- [x] A small human-authored supported-question/reference set is stored in `data/ragas_eval.json`.
- [x] `scripts/evaluate_ragas.py` runs the real Prince pipeline and records per-case and mean scores with retrieved evidence.
- [x] A RAGAS evaluation run and its actual score report have been captured in `evidence/ragas-metrics.json`. All four answers passed citation validation and were scored in the latest run.
- [x] RAGAS is pinned with its compatible `langchain-community` dependency; offline tests do not require invoking RAGAS or Gemini.

## Latest user-reported live checks (2026-10-12)

- [x] User reports that 5 live tests have now passed. Individual prompts and outputs were not supplied, so this is a user-reported result, not an independently reproduced run.
- [x] Unsupported/low-evidence questions are refused before generation by the calibrated retrieval threshold; generated citations are checked against retrieved chunk IDs. The UI hides retrieved evidence when it refuses or recognizes an insufficient-evidence answer. These controls reduce hallucination risk but do not guarantee perfect model behavior.
- [x] Historical RAGAS results and the reranker comparison are recorded in `docs/progress-log.md`; detailed latest per-case metrics remain in `evidence/ragas-metrics.json`.
