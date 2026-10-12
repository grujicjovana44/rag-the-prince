"""Book RAG orchestration with threshold refusal and citation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .embeddings import Embedder
from .generation import GenerationError, LLM, REFUSAL_MESSAGE, validate_citations
from .reranking import RerankedResult, Reranker, retrieve_and_rerank, should_refuse
from .vector_store import FaissVectorStore


@dataclass(frozen=True)
class RagResponse:
    question: str
    answer: str | None
    citations: tuple[str, ...]
    refused: bool
    generation_error: str | None
    evidence: tuple[RerankedResult, ...]


class PrinceRagPipeline:
    def __init__(
        self,
        store: FaissVectorStore,
        embedder: Embedder,
        reranker: Reranker,
        *,
        refusal_threshold: float,
        llm: LLM | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.refusal_threshold = refusal_threshold
        self.llm = llm

    def answer(self, question: str, *, candidate_n: int = 20, top_k: int = 4) -> RagResponse:
        evidence = retrieve_and_rerank(
            question,
            self.store,
            self.embedder,
            self.reranker,
            candidate_n=candidate_n,
            top_k=top_k,
        )
        evidence_tuple = tuple(evidence)
        if should_refuse(evidence, self.refusal_threshold):
            return RagResponse(
                question=question,
                answer=REFUSAL_MESSAGE,
                citations=(),
                refused=True,
                generation_error=None,
                evidence=evidence_tuple,
            )
        if self.llm is None:
            return self._generation_failed(
                question, evidence_tuple, "LLM generation is not configured."
            )
        try:
            answer = self.llm.generate(question, evidence)
            validation = validate_citations(answer, {item.chunk.chunk_id for item in evidence})
            if validation.invalid_ids:
                bad_ids = ", ".join(validation.invalid_ids)
                raise GenerationError(f"Generated citations reference chunks outside retrieved evidence: {bad_ids}.")
            if validation.missing_citation:
                raise GenerationError("Generated answer contains no chunk-ID citations.")
        except Exception as exc:
            return self._generation_failed(question, evidence_tuple, str(exc))
        return RagResponse(
            question=question,
            answer=answer,
            citations=validation.citations,
            refused=False,
            generation_error=None,
            evidence=evidence_tuple,
        )

    @staticmethod
    def _generation_failed(
        question: str, evidence: Sequence[RerankedResult], error: str
    ) -> RagResponse:
        return RagResponse(
            question=question,
            answer=None,
            citations=(),
            refused=False,
            generation_error=error,
            evidence=tuple(evidence),
        )
