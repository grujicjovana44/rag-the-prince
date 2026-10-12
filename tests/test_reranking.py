from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from src.chunking import Chunk
from src.reranking import (
    CalibrationExample,
    FlashRankReranker,
    calibrate_refusal_threshold,
    load_labeled_questions,
    should_refuse,
    RerankedResult,
    retrieve_and_rerank,
)
from src.vector_store import VectorSearchResult


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc", title="title", text=text)


@dataclass
class FakeRequest:
    query: str
    passages: list[dict[str, object]]


class FakeFlashRank:
    def rerank(self, request: FakeRequest) -> list[dict[str, object]]:
        return [
            {"id": "b", "score": 0.2},
            {"id": "a", "score": 0.9},
        ]


def test_flashrank_adapter_sorts_scores_and_preserves_known_chunk_ids() -> None:
    reranker = FlashRankReranker(
        ranker=FakeFlashRank(), rerank_request_type=FakeRequest
    )

    results = reranker.rerank("question", [chunk("a", "first"), chunk("b", "second")], top_k=1)

    assert results == [("a", 0.9)]


def test_flashrank_adapter_rejects_unknown_chunk_ids() -> None:
    class UnknownFlashRank:
        def rerank(self, request: FakeRequest) -> list[dict[str, object]]:
            return [{"id": "outside", "score": 0.4}]

    reranker = FlashRankReranker(ranker=UnknownFlashRank(), rerank_request_type=FakeRequest)

    with pytest.raises(ValueError, match="unknown chunk ID"):
        reranker.rerank("question", [chunk("a", "first")], top_k=1)


def test_calibration_prefers_refusal_on_balanced_accuracy_tie() -> None:
    examples = [
        CalibrationExample("supported_a", True, 0.8),
        CalibrationExample("supported_b", True, 0.4),
        CalibrationExample("unsupported_a", False, 0.4),
        CalibrationExample("unsupported_b", False, 0.1),
    ]

    result = calibrate_refusal_threshold(examples)

    assert result.threshold == pytest.approx(0.6)
    assert result.false_positive == 0
    assert result.false_negative == 1
    assert result.balanced_accuracy == pytest.approx(0.75)


def test_calibration_requires_both_labels_and_finite_scores() -> None:
    with pytest.raises(ValueError, match="supported and unsupported"):
        calibrate_refusal_threshold([CalibrationExample("one", True, 0.5)])
    with pytest.raises(ValueError, match="finite"):
        calibrate_refusal_threshold(
            [CalibrationExample("yes", True, float("nan")), CalibrationExample("no", False, 0.1)]
        )


def test_refusal_threshold_rejects_missing_or_low_evidence() -> None:
    item = RerankedResult(chunk("a", "evidence"), vector_score=0.8, rerank_score=0.4)

    assert should_refuse([], threshold=0.5)
    assert should_refuse([item], threshold=0.5)
    assert not should_refuse([item], threshold=0.4)


def test_retrieve_and_rerank_keeps_faiss_scores_and_uses_rerank_order() -> None:
    candidates = [
        VectorSearchResult(chunk("a", "first"), 0.7),
        VectorSearchResult(chunk("b", "second"), 0.6),
    ]

    class Store:
        def search(self, query: str, embedder: object, top_k: int) -> list[VectorSearchResult]:
            assert query == "question"
            assert top_k == 20
            return candidates

    class ReversingReranker:
        def rerank(self, query: str, chunks: object, top_k: int) -> list[tuple[str, float]]:
            assert [item.chunk_id for item in chunks] == ["a", "b"]
            assert top_k == 1
            return [("b", 0.93)]

    results = retrieve_and_rerank(
        "question", Store(), object(), ReversingReranker(), candidate_n=20, top_k=1  # type: ignore[arg-type]
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "b"
    assert results[0].vector_score == pytest.approx(0.6)
    assert results[0].rerank_score == pytest.approx(0.93)


def test_repository_calibration_set_has_supported_and_near_unsupported_examples() -> None:
    questions = load_labeled_questions()

    assert len(questions) == 8
    assert sum(question.supported for question in questions) == 4
    assert all(question.expected_chapters for question in questions if question.supported)
    assert all(not question.expected_chapters for question in questions if not question.supported)


def test_recorded_real_scores_reproduce_the_saved_threshold() -> None:
    report = json.loads(Path("data/refusal_calibration_scores.json").read_text(encoding="utf-8"))
    examples = [
        CalibrationExample(row["question_id"], row["supported"], row["score"])
        for row in report["examples"]
    ]

    calibration = calibrate_refusal_threshold(examples)

    assert calibration.threshold == pytest.approx(report["threshold"])
    assert calibration.true_positive == report["true_positive"]
    assert calibration.true_negative == report["true_negative"]
    assert calibration.false_positive == report["false_positive"]
    assert calibration.false_negative == report["false_negative"]
