from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src import cli
from src.chunking import Chunk
from src.pipeline import RagResponse
from src.reranking import RerankedResult


def test_cli_keeps_starter_defaults_and_adds_prince_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["src.cli", "--question", "custom question"])

    args = cli.parse_args()

    assert args.corpus_profile == "default"
    assert args.chunk_size == 140
    assert args.overlap == 30
    assert args.top_k == 2
    assert args.book_chunk_size == 800
    assert args.book_overlap == 120
    monkeypatch.setattr(sys, "argv", ["src.cli", "--corpus-profile", "prince", "--question", "book query"])
    assert cli.parse_args().corpus_profile == "prince"


def test_default_cli_output_is_preserved(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["src.cli", "--question", "What is retrieval augmented generation?"])

    cli.main()

    output = capsys.readouterr().out
    assert "Corpus profile: default" in output
    assert "Refused: False" in output
    assert "Citations: doc-1-chunk-0, doc-2-chunk-0" in output


def test_training_rag_cli_profile_still_runs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["src.cli", "--corpus-profile", "training-rag", "--question", "Why are citations important in RAG?"],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Corpus profile: training-rag" in output
    assert "Refused: False" in output


def test_prince_cli_prints_scores_evidence_and_generation_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chunk = Chunk(
        chunk_id="prince-chapter-xvii-chunk-4",
        doc_id="prince-chapter-xvii",
        title="The Prince Chapter XVII",
        text="A prince should avoid hatred.",
        chapter_label="XVII",
    )
    evidence = RerankedResult(chunk, vector_score=0.81, rerank_score=0.97)

    class FakePipeline:
        def answer(self, question: str, *, candidate_n: int, top_k: int) -> RagResponse:
            assert question == "How can a prince avoid hatred?"
            assert candidate_n == 20
            assert top_k == 3
            return RagResponse(
                question=question,
                answer=None,
                citations=(),
                refused=False,
                generation_error="GEMINI_API_KEY is not set.",
                evidence=(evidence,),
            )

    monkeypatch.setattr(cli, "_create_prince_pipeline", lambda args: FakePipeline())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src.cli",
            "--corpus-profile",
            "prince",
            "--question",
            "How can a prince avoid hatred?",
            "--top-k",
            "3",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert "Corpus profile: prince" in output
    assert "Generation error: GEMINI_API_KEY is not set." in output
    assert "[prince-chapter-xvii-chunk-4] chapter=XVII" in output
    assert "vector_score=0.810000 rerank_score=0.97000000" in output
    assert "A prince should avoid hatred." in output
