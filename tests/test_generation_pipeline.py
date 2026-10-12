from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.chunking import Chunk
from src.generation import (
    DEFAULT_GEMINI_MODEL,
    GenerationError,
    GeminiLLM,
    build_grounded_prompt,
    validate_citations,
)
from src.pipeline import PrinceRagPipeline
from src.reranking import RerankedResult
from src.vector_store import VectorSearchResult


def result(chunk_id: str, score: float = 0.9) -> RerankedResult:
    item = Chunk(
        chunk_id=chunk_id,
        doc_id="prince-chapter-xvii",
        title="The Prince — Chapter XVII",
        text="A prince should avoid hatred by not taking the property of his subjects.",
        chapter_label="XVII",
        chapter_title="OF CRUELTY AND CLEMENCY",
    )
    return RerankedResult(item, vector_score=0.8, rerank_score=score)


class FakeStore:
    def __init__(self, rows: list[VectorSearchResult]) -> None:
        self.rows = rows

    def search(self, query: str, embedder: object, top_k: int) -> list[VectorSearchResult]:
        return self.rows[:top_k]


class FakeReranker:
    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[str, float]]:
        return [(chunk.chunk_id, self.score) for chunk in chunks[:top_k]]


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.called = False

    def generate(self, question: str, evidence: list[RerankedResult]) -> str:
        self.called = True
        return self.text


def make_pipeline(llm: object | None, *, score: float = 0.9) -> PrinceRagPipeline:
    evidence = result("prince-chapter-xvii-chunk-4", score=score)
    store = FakeStore([VectorSearchResult(evidence.chunk, 0.8)])
    return PrinceRagPipeline(store, object(), FakeReranker(score), refusal_threshold=0.5, llm=llm)  # type: ignore[arg-type]


def test_grounded_prompt_asks_for_synthesis_and_only_retrieved_citations() -> None:
    prompt = build_grounded_prompt("How can a prince avoid hatred?", [result("prince-chapter-xvii-chunk-4")])

    assert "natural-language synthesis" in prompt
    assert "[prince-chapter-xvii-chunk-4]" in prompt
    assert "Use only these citation IDs" in prompt


def test_citation_validation_rejects_unknown_and_missing_references() -> None:
    allowed = {"prince-chapter-xvii-chunk-4"}

    assert validate_citations("Grounded answer [prince-chapter-xvii-chunk-4].", allowed).is_valid
    unknown = validate_citations("Claim [prince-chapter-xix-chunk-1].", allowed)
    assert unknown.invalid_ids == ("prince-chapter-xix-chunk-1",)
    assert not validate_citations("Claim without a citation.", allowed).is_valid


def test_pipeline_returns_synthesis_and_validated_citations() -> None:
    llm = FakeLLM("A prince can reduce hatred by respecting subjects' property [prince-chapter-xvii-chunk-4].")

    response = make_pipeline(llm).answer("How can a prince avoid hatred?")

    assert response.answer == llm.text
    assert response.citations == ("prince-chapter-xvii-chunk-4",)
    assert response.generation_error is None
    assert not response.refused


def test_pipeline_refuses_before_generation_when_score_is_low() -> None:
    llm = FakeLLM("This should never be generated.")

    response = make_pipeline(llm, score=0.2).answer("Unsupported question")

    assert response.refused
    assert response.answer is not None
    assert not llm.called
    assert response.evidence


def test_pipeline_reports_invalid_citations_and_keeps_retrieved_evidence() -> None:
    llm = FakeLLM("Unsupported claim [prince-chapter-xxvi-chunk-99].")

    response = make_pipeline(llm).answer("Question")

    assert response.answer is None
    assert "outside retrieved evidence" in (response.generation_error or "")
    assert response.evidence


def test_pipeline_reports_missing_key_or_generation_failure_without_fabrication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = GeminiLLM()
    with pytest.raises(GenerationError, match="GEMINI_API_KEY"):
        llm.generate("question", [result("prince-chapter-xvii-chunk-4")])

    missing_key_response = make_pipeline(llm).answer("Question")
    assert missing_key_response.answer is None
    assert "GEMINI_API_KEY" in (missing_key_response.generation_error or "")
    assert missing_key_response.evidence

    class FailedLLM:
        def generate(self, question: str, evidence: list[RerankedResult]) -> str:
            raise RuntimeError("simulated offline failure")

    response = make_pipeline(FailedLLM()).answer("Question")
    assert response.answer is None
    assert response.generation_error == "simulated offline failure"
    assert response.evidence


def test_gemini_adapter_uses_environment_model_and_official_client_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    class Models:
        def generate_content(self, *, model: str, contents: str, config: dict[str, object]) -> SimpleNamespace:
            assert model == "test-model"
            assert "Retrieved evidence" in contents
            assert config["response_mime_type"] == "application/json"
            schema = config["response_json_schema"]
            citation_schema = schema["properties"]["claims"]["items"]["properties"]["citation_ids"]
            assert citation_schema["items"]["enum"] == ["prince-chapter-xvii-chunk-4"]
            return SimpleNamespace(
                text=(
                    '{"claims":[{"text":"Answer",'
                    '"citation_ids":["prince-chapter-xvii-chunk-4"]}]}'
                )
            )

    llm = GeminiLLM(client=SimpleNamespace(models=Models()))
    response = llm.generate("question", [result("prince-chapter-xvii-chunk-4")])

    assert llm.model_name == "test-model"
    assert response.startswith("Answer")
    assert DEFAULT_GEMINI_MODEL == "gemini-3.1-flash-lite"


def test_gemini_adapter_rejects_citation_outside_schema_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class Models:
        def generate_content(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                text=(
                    '{"claims":[{"text":"Answer",'
                    '"citation_ids":["prince-chapter-xix-chunk-0"]}]}'
                )
            )

    llm = GeminiLLM(client=SimpleNamespace(models=Models()))
    with pytest.raises(GenerationError, match="outside the retrieved evidence"):
        llm.generate("question", [result("prince-chapter-xvii-chunk-4")])
