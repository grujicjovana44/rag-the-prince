"""Candidate reranking and refusal-threshold calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

from .chunking import Chunk
from .embeddings import Embedder
from .vector_store import FaissVectorStore, VectorSearchResult


DEFAULT_RERANKER_MODEL = "ms-marco-TinyBERT-L-2-v2"
DEFAULT_RERANKER_CACHE = Path(".cache/flashrank")
DEFAULT_CALIBRATION_SET = Path("data/refusal_calibration.json")


@dataclass(frozen=True)
class RerankedResult:
    chunk: Chunk
    vector_score: float
    rerank_score: float


class Reranker(Protocol):
    def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[tuple[str, float]]: ...


class FlashRankReranker:
    """Thin adapter around FlashRank; model initialization is deferred."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        cache_dir: str | Path = DEFAULT_RERANKER_CACHE,
        *,
        ranker: Any | None = None,
        rerank_request_type: Any | None = None,
    ) -> None:
        if ranker is None or rerank_request_type is None:
            try:
                from flashrank import Ranker, RerankRequest
            except ImportError as exc:
                raise RuntimeError("FlashRankReranker requires the flashrank package.") from exc
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            ranker = Ranker(model_name=model_name, cache_dir=str(cache_dir))
            rerank_request_type = RerankRequest
        self.model_name = model_name
        self._ranker = ranker
        self._rerank_request_type = rerank_request_type

    def rerank(self, query: str, chunks: Sequence[Chunk], top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not chunks:
            return []
        passages = [
            {"id": chunk.chunk_id, "text": chunk.text, "meta": {"doc_id": chunk.doc_id}}
            for chunk in chunks
        ]
        ranked = self._ranker.rerank(self._rerank_request_type(query=query, passages=passages))
        scores: list[tuple[str, float]] = []
        seen: set[str] = set()
        known_ids = {chunk.chunk_id for chunk in chunks}
        for item in ranked:
            chunk_id = str(item["id"])
            score = float(item["score"])
            if chunk_id not in known_ids:
                raise ValueError(f"FlashRank returned unknown chunk ID {chunk_id!r}.")
            if chunk_id in seen:
                raise ValueError(f"FlashRank returned duplicate chunk ID {chunk_id!r}.")
            if not math.isfinite(score):
                raise ValueError(f"FlashRank returned a non-finite score for {chunk_id!r}.")
            seen.add(chunk_id)
            scores.append((chunk_id, score))
        scores.sort(key=lambda row: row[1], reverse=True)
        return scores[:top_k]


def retrieve_and_rerank(
    query: str,
    store: FaissVectorStore,
    embedder: Embedder,
    reranker: Reranker,
    *,
    candidate_n: int = 20,
    top_k: int = 4,
) -> list[RerankedResult]:
    if candidate_n <= 0:
        raise ValueError("candidate_n must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    candidates: list[VectorSearchResult] = store.search(query, embedder, top_k=candidate_n)
    by_id = {item.chunk.chunk_id: item for item in candidates}
    ranked = reranker.rerank(query, [item.chunk for item in candidates], top_k=top_k)
    results = []
    for chunk_id, score in ranked:
        candidate = by_id.get(chunk_id)
        if candidate is None:
            raise ValueError(f"Reranker returned a chunk outside the candidate set: {chunk_id!r}.")
        results.append(
            RerankedResult(
                chunk=candidate.chunk,
                vector_score=candidate.score,
                rerank_score=score,
            )
        )
    return results


def should_refuse(results: Sequence[RerankedResult], threshold: float) -> bool:
    """Refuse when there is no reranked evidence or the best score misses threshold."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return not results or results[0].rerank_score < threshold


@dataclass(frozen=True)
class LabeledQuestion:
    question_id: str
    question: str
    supported: bool
    rationale: str
    expected_chapters: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationExample:
    question_id: str
    supported: bool
    score: float


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    balanced_accuracy: float


def load_labeled_questions(path: str | Path = DEFAULT_CALIBRATION_SET) -> list[LabeledQuestion]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [
        LabeledQuestion(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            supported=bool(row["supported"]),
            rationale=str(row["rationale"]),
            expected_chapters=tuple(str(item) for item in row["expected_chapters"]),
        )
        for row in records
    ]
    if not questions:
        raise ValueError("The refusal calibration set must not be empty.")
    if len({row.question_id for row in questions}) != len(questions):
        raise ValueError("Calibration question IDs must be unique.")
    if not any(row.supported for row in questions) or not any(not row.supported for row in questions):
        raise ValueError("Calibration set must include supported and unsupported questions.")
    return questions


def calibrate_refusal_threshold(examples: Sequence[CalibrationExample]) -> ThresholdCalibration:
    """Choose the threshold maximizing balanced accuracy; ties favor refusal."""
    if not examples:
        raise ValueError("At least one measured calibration score is required.")
    if not any(example.supported for example in examples) or not any(
        not example.supported for example in examples
    ):
        raise ValueError("Calibration scores must include supported and unsupported examples.")
    if any(not math.isfinite(example.score) for example in examples):
        raise ValueError("Calibration scores must be finite.")

    unique_scores = sorted({example.score for example in examples})
    candidates = [unique_scores[0]]
    candidates.extend((left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:]))
    candidates.append(math.nextafter(unique_scores[-1], math.inf))

    evaluations = []
    for threshold in candidates:
        tp = sum(example.supported and example.score >= threshold for example in examples)
        tn = sum((not example.supported) and example.score < threshold for example in examples)
        fp = sum((not example.supported) and example.score >= threshold for example in examples)
        fn = sum(example.supported and example.score < threshold for example in examples)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        evaluations.append(
            ThresholdCalibration(
                threshold=threshold,
                true_positive=tp,
                true_negative=tn,
                false_positive=fp,
                false_negative=fn,
                balanced_accuracy=(sensitivity + specificity) / 2,
            )
        )
    return max(evaluations, key=lambda result: (result.balanced_accuracy, result.threshold))
