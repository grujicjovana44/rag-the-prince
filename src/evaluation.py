"""RAGAS scoring helpers for grounded, reference-backed Prince answers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class EvaluationSample:
    case_id: str
    question: str
    response: str
    retrieved_contexts: tuple[str, ...]
    reference: str


def build_ragas_metrics(*, evaluator_llm: Any, evaluator_embeddings: Any) -> dict[str, Any]:
    """Create the four current collections-API metrics, with injected evaluators."""
    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires the pinned ragas dependency from requirements.txt."
        ) from exc

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }


async def score_sample(
    sample: EvaluationSample,
    metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Score one real pipeline answer; metric calls are intentionally explicit."""
    metric_inputs = {
        "user_input": sample.question,
        "response": sample.response,
        "retrieved_contexts": list(sample.retrieved_contexts),
        "reference": sample.reference,
    }
    required_inputs = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "context_precision": ("user_input", "reference", "retrieved_contexts"),
        "context_recall": ("user_input", "reference", "retrieved_contexts"),
    }
    if set(metrics) != set(RAGAS_METRIC_NAMES):
        raise ValueError(f"Expected exactly these RAGAS metrics: {', '.join(RAGAS_METRIC_NAMES)}.")

    scores: dict[str, float] = {}
    for metric_name in RAGAS_METRIC_NAMES:
        payload = {key: metric_inputs[key] for key in required_inputs[metric_name]}
        result = await metrics[metric_name].ascore(**payload)
        value = float(result.value)
        if not math.isfinite(value):
            raise ValueError(f"RAGAS metric {metric_name!r} returned a non-finite score.")
        scores[metric_name] = value
    return scores


async def score_samples(
    samples: Sequence[EvaluationSample],
    metrics: Mapping[str, Any],
) -> list[dict[str, float]]:
    """Score samples sequentially to avoid unexpectedly high evaluator concurrency."""
    if not samples:
        raise ValueError("At least one evaluation sample is required.")
    return [await score_sample(sample, metrics) for sample in samples]
