from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from src.chunking import Chunk
from src.vector_store import FaissVectorStore, StaleIndexError


class FakeEmbedder:
    model_name = "fake-embedder-v1"

    vectors = {
        "alpha": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "diagonal": [1.0, 1.0],
        "query": [1.0, 0.0],
    }

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.concatenate((self.vectors, vectors), axis=0)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, axis=1), order.astype(np.int64)


def make_fake_faiss() -> SimpleNamespace:
    def write_index(index: FakeIndexFlatIP, path: str) -> None:
        with open(path, "wb") as output:
            pickle.dump((index.d, index.vectors), output)

    def read_index(path: str) -> FakeIndexFlatIP:
        with open(path, "rb") as source:
            dimension, vectors = pickle.load(source)
        index = FakeIndexFlatIP(dimension)
        index.vectors = vectors
        return index

    return SimpleNamespace(IndexFlatIP=FakeIndexFlatIP, write_index=write_index, read_index=read_index)


def sample_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="prince-chapter-i-chunk-0", doc_id="prince-chapter-i", title="I", text="alpha", chapter_label="I"),
        Chunk(chunk_id="prince-chapter-i-chunk-1", doc_id="prince-chapter-i", title="I", text="beta", chapter_label="I"),
        Chunk(chunk_id="prince-chapter-i-chunk-2", doc_id="prince-chapter-i", title="I", text="diagonal", chapter_label="I"),
    ]


def test_vector_store_normalizes_and_searches_by_cosine_score() -> None:
    store = FaissVectorStore.build(
        sample_chunks(), FakeEmbedder(), chunk_size=800, overlap=120, faiss_module=make_fake_faiss()
    )

    results = store.search("query", FakeEmbedder(), top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "prince-chapter-i-chunk-0",
        "prince-chapter-i-chunk-2",
        "prince-chapter-i-chunk-1",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)
    assert results[2].score == pytest.approx(0.0)


def test_vector_store_persists_index_mapping_and_manifest(tmp_path) -> None:
    chunks = sample_chunks()
    store = FaissVectorStore.build(
        chunks, FakeEmbedder(), chunk_size=800, overlap=120, faiss_module=make_fake_faiss()
    )
    store.save(tmp_path)

    loaded = FaissVectorStore.load(
        tmp_path,
        expected_text_hash=store.manifest.text_hash,
        chunk_size=800,
        overlap=120,
        model_name=FakeEmbedder.model_name,
        faiss_module=make_fake_faiss(),
    )

    assert [chunk.chunk_id for chunk in loaded.chunks] == [chunk.chunk_id for chunk in chunks]
    assert loaded.manifest == store.manifest
    assert loaded.search("query", FakeEmbedder(), top_k=1)[0].chunk.text == "alpha"


def test_vector_store_detects_stale_manifest_parameters(tmp_path) -> None:
    store = FaissVectorStore.build(
        sample_chunks(), FakeEmbedder(), chunk_size=800, overlap=120, faiss_module=make_fake_faiss()
    )
    store.save(tmp_path)

    with pytest.raises(StaleIndexError, match="stale"):
        FaissVectorStore.load(tmp_path, chunk_size=700, faiss_module=make_fake_faiss())


def test_vector_store_rejects_a_different_query_model() -> None:
    store = FaissVectorStore.build(
        sample_chunks(), FakeEmbedder(), chunk_size=800, overlap=120, faiss_module=make_fake_faiss()
    )
    other = FakeEmbedder()
    other.model_name = "other-model"

    with pytest.raises(StaleIndexError, match="does not match"):
        store.search("query", other)
