"""Run the four reference-backed RAGAS metrics on real Prince RAG executions."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.cli import _create_prince_pipeline
from src.evaluation import EvaluationSample, RAGAS_METRIC_NAMES, build_ragas_metrics, score_samples
from src.generation import DEFAULT_GEMINI_MODEL

DEFAULT_DATASET = ROOT / "data/ragas_eval.json"
DEFAULT_OUTPUT = ROOT / "evidence/ragas-metrics.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_dataset(path: Path, limit: int | None) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    records = json.loads(raw.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON list: {path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0 when supplied.")
        records = records[:limit]

    cases: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Evaluation row {index} must be a JSON object.")
        case = {key: str(row.get(key, "")).strip() for key in ("case_id", "question", "reference")}
        if any(not value for value in case.values()):
            raise ValueError(f"Evaluation row {index} needs case_id, question, and reference.")
        if case["case_id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation case_id: {case['case_id']!r}.")
        seen_ids.add(case["case_id"])
        cases.append(case)
    return cases, hashlib.sha256(raw).hexdigest()


def _make_evaluators(model_name: str) -> tuple[object, object]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for RAGAS LLM judging.")
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS judging requires openai and the dependencies in requirements.txt."
        ) from exc

    # RAGAS 0.4.3 currently expects instructor.from_genai(), which is missing
    # from some compatible instructor releases. Google's OpenAI-compatible
    # Gemini endpoint keeps judging on Gemini without adding another adapter.
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    evaluator_llm = llm_factory(model_name, provider="openai", client=client)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        device="cpu",
        normalize_embeddings=True,
        model_kwargs={"local_files_only": True},
    )
    return evaluator_llm, evaluator_embeddings


async def _run(args: argparse.Namespace) -> dict[str, object]:
    cases, dataset_hash = _load_dataset(args.dataset, args.limit)
    model_name = os.getenv("RAGAS_EVAL_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    evaluator_llm, evaluator_embeddings = _make_evaluators(model_name)
    metrics = build_ragas_metrics(
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
    )

    pipeline_args = argparse.Namespace(
        book_chunk_size=args.chunk_size,
        book_overlap=args.overlap,
        index_dir=str(args.index_dir),
        refusal_threshold=None,
        reranker_cache_dir=str(args.reranker_cache_dir),
    )
    pipeline = _create_prince_pipeline(pipeline_args)
    scored_samples: list[EvaluationSample] = []
    evidence_rows: list[dict[str, object]] = []
    scoreable_rows: list[dict[str, object]] = []

    for case in cases:
        response = pipeline.answer(
            case["question"],
            candidate_n=args.candidate_n,
            top_k=args.top_k,
        )
        contexts = tuple(item.chunk.text for item in response.evidence)
        row: dict[str, object] = {
            "case_id": case["case_id"],
            "question": case["question"],
            "reference": case["reference"],
            "response": response.answer,
            "citations": list(response.citations),
            "refused": response.refused,
            "generation_error": response.generation_error,
            "retrieved_contexts": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "vector_score": item.vector_score,
                    "rerank_score": item.rerank_score,
                    "text": item.chunk.text,
                }
                for item in response.evidence
            ],
        }
        evidence_rows.append(row)
        if response.refused or response.generation_error or response.answer is None or not contexts:
            row["evaluation_status"] = "not_scored"
            reason = "refused" if response.refused else response.generation_error or "empty answer or context"
            print(f"Not scored {case['case_id']}: {reason}")
            continue

        row["evaluation_status"] = "scored"
        scored_samples.append(
            EvaluationSample(
                case_id=case["case_id"],
                question=case["question"],
                response=response.answer,
                retrieved_contexts=contexts,
                reference=case["reference"],
            )
        )
        scoreable_rows.append(row)
        print(f"Generated RAG answer for {case['case_id']}.")

    if not scored_samples:
        raise RuntimeError("No evaluation cases produced a valid, cited answer; no metric scores were written.")
    score_rows = await score_samples(scored_samples, metrics)
    means = {
        name: sum(row[name] for row in score_rows) / len(score_rows)
        for name in RAGAS_METRIC_NAMES
    }
    for row, scores in zip(scoreable_rows, score_rows, strict=True):
        row["scores"] = scores

    try:
        ragas_version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("RAGAS package metadata is unavailable.") from exc
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ragas_version": ragas_version,
        "generation_model": os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        "evaluator_model": model_name,
        "embedding_model": EMBEDDING_MODEL,
        "dataset": str(args.dataset.relative_to(ROOT)) if args.dataset.is_relative_to(ROOT) else str(args.dataset),
        "dataset_sha256": dataset_hash,
        "candidate_n": args.candidate_n,
        "top_k": args.top_k,
        "scored_case_count": len(score_rows),
        "not_scored_case_count": len(evidence_rows) - len(score_rows),
        "mean_scores": means,
        "results": evidence_rows,
        "note": "RAGAS LLM-judge scores are estimates; inspect per-case evidence and references before drawing conclusions.",
    }


def main() -> None:
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env", override=False)
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N labeled examples.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--candidate-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--index-dir", type=Path, default=ROOT / ".cache/prince-faiss")
    parser.add_argument("--reranker-cache-dir", type=Path, default=ROOT / ".cache/flashrank")
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RAGAS {report['ragas_version']} evaluated {len(report['results'])} real answers.")
    for metric_name, score in report["mean_scores"].items():
        print(f"{metric_name}: {score:.4f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
