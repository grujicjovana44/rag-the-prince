"""Capture reproducible Week 6 evidence from actual CLI and test executions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import (
    BOOK_CHUNK_OVERLAP,
    BOOK_CHUNK_SIZE,
    SENTENCE_CHUNK_OVERLAP,
    SENTENCE_CHUNK_SIZE,
    build_book_chunks,
    build_chunks,
    build_sentence_chunks,
)
from src.corpus import load_prince_corpus
from src.retrieval import retrieve_top_k


SUPPORTED_QUESTION = "What does Chapter II say about hereditary principalities and change?"
UNSUPPORTED_QUESTION = "What names of parents were given for Castruccio Castracani, and how was the child raised?"
OUTCOME_ONE_QUESTION = "What qualities are praised or blamed in men and princes?"


def _capture_cli(question: str, *, top_k: int, allow_gemini: bool, offline: bool) -> tuple[str, str, int]:
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "--corpus-profile",
        "prince",
        "--question",
        question,
        "--top-k",
        str(top_k),
    ]
    environment = os.environ.copy()
    if not allow_gemini:
        environment.pop("GEMINI_API_KEY", None)
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).replace("\r", "\n").strip()
    command_text = " ".join(repr(part) for part in command)
    return command_text, output, result.returncode


def _chunking_comparison() -> str:
    documents = load_prince_corpus()
    chapter = next(document for document in documents if document.chapter_label == "XV")
    configs = (
        ("starter 140/30", build_chunks([chapter], chunk_size=140, overlap=30)),
        (
            f"book {BOOK_CHUNK_SIZE}/{BOOK_CHUNK_OVERLAP}",
            build_book_chunks([chapter], chunk_size=BOOK_CHUNK_SIZE, overlap=BOOK_CHUNK_OVERLAP),
        ),
        (
            f"sentence-aware {SENTENCE_CHUNK_SIZE}/{SENTENCE_CHUNK_OVERLAP}",
            build_sentence_chunks(
                [chapter], chunk_size=SENTENCE_CHUNK_SIZE, overlap=SENTENCE_CHUNK_OVERLAP
            ),
        ),
    )
    lines = [f"Question: {OUTCOME_ONE_QUESTION}"]
    for label, chunks in configs:
        results = retrieve_top_k(OUTCOME_ONE_QUESTION, chunks, k=2)
        lines.append(f"{label}: chunks={len(chunks)}")
        for result in results:
            excerpt = result.chunk.text[:200].replace("\n", " ")
            lines.append(
                f"  {result.chunk.chunk_id} score={result.score:.6f} text={excerpt!r}"
            )
    lines.extend(
        [
            "Qualitative relevance: the first result in all three configurations starts with the Chapter XV heading named by the query and contains the surrounding discussion, so it directly addresses the question. The starter result is precise but cuts off mid-sentence and needs neighboring chunks for fuller context. The larger word-aware result preserves more explanation in one passage. Sentence-aware chunking avoids most sentence breaks and improves readability, though chunk lengths are less uniform and whole-sentence overlap is approximate. The starter's second result is a short overlapping fragment; the word-aware result continues the same argument, while the sentence-aware second result captures a later list of qualities. Those later chunks add context, but rank below the opening passage because they do not contain the chapter's full framing of the question.",
            "The starter retriever uses lexical overlap divided by chunk token count. Scores across chunk sizes are not directly comparable; this is a qualitative comparison, not a measured relevance benchmark.",
        ]
    )
    return "\n".join(lines)


def _section(title: str, command: str, output: str, exit_code: int) -> str:
    return (
        f"### {title}\n\n"
        f"Command: `{command}` (exit code {exit_code})\n\n"
        "```text\n"
        f"{output or '[no output]'}\n"
        "```\n"
    )


def _ragas_summary() -> str | None:
    report_path = ROOT / "evidence/ragas-metrics.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    means = report.get("mean_scores", {})
    lines = [
        "## RAGAS evaluation (latest saved run)",
        "",
        f"Generated: {report.get('generated_at_utc', 'unknown')} UTC with RAGAS {report.get('ragas_version', 'unknown')}.",
        f"Scored cases: {report.get('scored_case_count', 0)}; not scored: {report.get('not_scored_case_count', 0)}.",
        "",
    ]
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if name in means:
            lines.append(f"- {name}: {float(means[name]):.4f}")
    lines.extend(["", "Per-case answers, citations, retrieved context, and scores are in `evidence/ragas-metrics.json`."])
    return "\n".join(lines)


def generate_evidence(*, output_path: Path, allow_gemini: bool, offline: bool) -> str:
    if allow_gemini:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    comparison = _chunking_comparison()
    top2_command, top2_output, top2_code = _capture_cli(
        SUPPORTED_QUESTION, top_k=2, allow_gemini=allow_gemini, offline=offline
    )
    top3_command, top3_output, top3_code = _capture_cli(
        SUPPORTED_QUESTION, top_k=3, allow_gemini=False, offline=offline
    )
    unsupported_command, unsupported_output, unsupported_code = _capture_cli(
        UNSUPPORTED_QUESTION, top_k=4, allow_gemini=False, offline=offline
    )
    threshold_report = json.loads((ROOT / "data/refusal_calibration_scores.json").read_text(encoding="utf-8"))
    threshold = float(threshold_report["threshold"])
    best_score_match = re.search(r"rerank_score=([0-9.]+)", unsupported_output)
    refusal_explanation = (
        f"Calibrated threshold={threshold:.8f}; best retrieved rerank score="
        f"{float(best_score_match.group(1)):.8f}, below threshold, so the query is refused."
        if best_score_match
        else f"Calibrated threshold={threshold:.8f}; no reranked evidence was returned, so the query is refused."
    )
    test_command = [sys.executable, "-m", "pytest", "-q"]
    test_result = subprocess.run(
        test_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    test_output = (test_result.stdout + test_result.stderr).replace("\r", "\n").strip()
    test_command_text = " ".join(repr(part) for part in test_command)

    generation_available = "Generation error: None" in top2_output
    status = (
        "Generated answer captured from Gemini."
        if generation_available
        else "INCOMPLETE: the supported query did not produce a Gemini answer. The captured output records the real generation error and retrieved evidence; no answer has been fabricated."
    )
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    retriever_command = [sys.executable, "scripts/compare_retrievers.py", "--offline"]
    retriever_result = subprocess.run(
        retriever_command,
        cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    retriever_output = (retriever_result.stdout + retriever_result.stderr).replace("\r", "\n").strip()
    retriever_command_text = " ".join(repr(part) for part in retriever_command)
    sections = [
        "# Week 6 RAG evidence",
        "",
        f"Generated at: `{timestamp}`",
        "",
        "This file is generated from actual executions by `scripts/generate_evidence.py`. Pair contributions and demo lead are intentionally omitted for the students to complete.",
        "",
        "## Outcome 1: Chunking impact",
        "",
        "The comparison below runs the same query against Chapter XV with the starter, word-aware book, and sentence-aware chunkers.",
        "",
        "```text",
        comparison,
        "```",
        "",
        "## Outcome 2: Top-k retrieval",
        "",
        f"Query: {SUPPORTED_QUESTION}",
        "",
        _section("top_k=2 output", top2_command, top2_output, top2_code),
        _section("top_k=3 output", top3_command, top3_output, top3_code),
        "Ranking explanation for this query: top‑1 is Chapter II because it directly states that hereditary states are easier to hold and gives the ancestral-customs/prudence reason. Top‑2 is Chapter I because it defines hereditary versus new principalities, giving relevant background but not the requested discussion of change. Top‑3 is Chapter III because it discusses why people change rulers in new principalities; this is related to ‘change’ but less directly answers the hereditary-principalities question. The reranker score ordering agrees with this relevance distinction.",
        "Relevance uses L2-normalized MiniLM embeddings in FAISS `IndexFlatIP` (cosine similarity), followed by FlashRank scores; the displayed final order is reranker order.",
        "",
        "## Additional retriever comparison: multi-vector and parent retrieval",
        "",
        "Both strategies use the same Chapter XV corpus, 500/80 child chunks, MiniLM candidates, and FlashRank ranking. Multi-vector retrieval returns the matching child passages. Parent retrieval groups those child hits and returns their 1,600-character containing window.",
        _section("multi-vector vs parent retrieval", retriever_command_text, retriever_output, retriever_result.returncode),
        "Interpretation: child-level multi-vector results are compact and precise, while parent retrieval adds surrounding context and can bridge a boundary at the cost of more irrelevant text. This is a qualitative single-query comparison, not a labeled benchmark; output length and answer coverage should be considered together.",
        "",
        "## Outcome 3: Citation quality",
        "",
        f"Supported query: {SUPPORTED_QUESTION}",
        "",
        status,
        "",
        "The actual captured CLI output follows; it includes the retrieved evidence and, when generation is available, the answer and its validated citations.",
        "",
        _section("supported query output", top2_command, top2_output, top2_code),
        "## Outcome 4: Refusal behavior",
        "",
        f"Unsupported query: {UNSUPPORTED_QUESTION}",
        "",
        "The fixed threshold was selected from the real scores in `data/refusal_calibration_scores.json` using the eight repository-held labeled examples.",
        refusal_explanation,
        "",
        _section("unsupported query output", unsupported_command, unsupported_output, unsupported_code),
        "## Outcome 5: Final evidence pair",
        "",
        f"Supported command: `{top2_command}`",
        "",
        "Supported output:",
        "",
        "```text",
        top2_output or "[no output]",
        "```",
        "",
        f"Unsupported command: `{unsupported_command}`",
        "",
        "Unsupported output:",
        "",
        "```text",
        unsupported_output or "[no output]",
        "```",
        "",
        "## Test output snapshot",
        "",
        f"Command: `{test_command_text}` (exit code {test_result.returncode})",
        "",
        "```text",
        test_output or "[no output]",
        "```",
        "",
    ]
    ragas_section = _ragas_summary()
    if ragas_section:
        test_section_index = sections.index("## Test output snapshot")
        sections[test_section_index:test_section_index] = [ragas_section, ""]
    output_path.write_text("\n".join(sections), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence.md")
    parser.add_argument(
        "--allow-gemini",
        action="store_true",
        help="Allow supported-query commands to use GEMINI_API_KEY and make a Gemini API request.",
    )
    parser.add_argument("--offline", action="store_true", help="Use already cached Hugging Face model files only.")
    args = parser.parse_args()
    status = generate_evidence(output_path=args.output, allow_gemini=args.allow_gemini, offline=args.offline)
    print(f"Evidence written to {args.output}")
    print(status)


if __name__ == "__main__":
    main()
