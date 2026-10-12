"""Persistent FAISS cosine-similarity store for chapter-aware chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


DEFAULT_CACHE_DIR = Path(".cache/prince-faiss")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MANIFEST_VERSION = 1


class StaleIndexError(ValueError):
    """Raised when a saved vector index does not match requested inputs."""


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IndexManifest:
    manifest_version: int
    text_hash: str
    chunk_size: int
    overlap: int
    model_name: str
    dimension: int
    chunk_count: int


def hash_chunk_text(chunks: list[Chunk]) -> str:
    """Hash ordered IDs and chunk text so edits or re-chunking invalidate cache."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_faiss(faiss_module: Any | None) -> Any:
    if faiss_module is not None:
        return faiss_module
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS vector storage requires the faiss-cpu package.") from exc
    return faiss


def _normalized_matrix(vectors: Any, expected_count: int | None = None) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix.")
    if expected_count is not None and matrix.shape[0] != expected_count:
        raise ValueError(
            f"Embedder returned {matrix.shape[0]} vectors for {expected_count} texts."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embedding matrix must not be empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _chunk_from_dict(value: dict[str, Any]) -> Chunk:
    return Chunk(**value)


class FaissVectorStore:
    """FAISS IndexFlatIP over L2-normalized vectors, equivalent to cosine score."""

    def __init__(self, index: Any, chunks: list[Chunk], manifest: IndexManifest, faiss_module: Any) -> None:
        self._index = index
        self._chunks = list(chunks)
        self.manifest = manifest
        self._faiss = faiss_module

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        chunk_size: int,
        overlap: int,
        faiss_module: Any | None = None,
    ) -> "FaissVectorStore":
        if not chunks:
            raise ValueError("Cannot build a vector store without chunks.")
        faiss = _load_faiss(faiss_module)
        vectors = _normalized_matrix(embedder.encode([chunk.text for chunk in chunks]), len(chunks))
        index = faiss.IndexFlatIP(int(vectors.shape[1]))
        index.add(vectors)
        manifest = IndexManifest(
            manifest_version=MANIFEST_VERSION,
            text_hash=hash_chunk_text(chunks),
            chunk_size=chunk_size,
            overlap=overlap,
            model_name=embedder.model_name,
            dimension=int(vectors.shape[1]),
            chunk_count=len(chunks),
        )
        return cls(index, chunks, manifest, faiss)

    @classmethod
    def load(
        cls,
        directory: str | Path = DEFAULT_CACHE_DIR,
        *,
        expected_text_hash: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        model_name: str | None = None,
        faiss_module: Any | None = None,
    ) -> "FaissVectorStore":
        directory = Path(directory)
        faiss = _load_faiss(faiss_module)
        manifest_path = directory / "manifest.json"
        chunks_path = directory / "chunks.json"
        index_path = directory / "index.faiss"
        if not (manifest_path.is_file() and chunks_path.is_file() and index_path.is_file()):
            raise FileNotFoundError(f"Vector index is incomplete under {directory}.")

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = IndexManifest(**manifest_data)
        if manifest.manifest_version != MANIFEST_VERSION:
            raise StaleIndexError("Vector index manifest version is unsupported.")
        requested = {
            "text_hash": expected_text_hash,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model_name": model_name,
        }
        mismatches = [
            f"{name}: saved={getattr(manifest, name)!r}, requested={value!r}"
            for name, value in requested.items()
            if value is not None and getattr(manifest, name) != value
        ]
        if mismatches:
            raise StaleIndexError("Saved vector index is stale (" + "; ".join(mismatches) + ").")

        chunk_data = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [_chunk_from_dict(value) for value in chunk_data]
        if len(chunks) != manifest.chunk_count or hash_chunk_text(chunks) != manifest.text_hash:
            raise StaleIndexError("Saved chunk mapping does not match its manifest.")
        index = faiss.read_index(str(index_path))
        if int(index.ntotal) != manifest.chunk_count or int(index.d) != manifest.dimension:
            raise StaleIndexError("Saved FAISS index dimensions do not match its manifest.")
        return cls(index, chunks, manifest, faiss)

    def search(self, query: str, embedder: Embedder, top_k: int = 20) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if embedder.model_name != self.manifest.model_name:
            raise StaleIndexError(
                f"Query embedder {embedder.model_name!r} does not match indexed model "
                f"{self.manifest.model_name!r}."
            )
        query_vector = _normalized_matrix(embedder.encode([query]), 1)
        if int(query_vector.shape[1]) != self.manifest.dimension:
            raise ValueError("Query embedding dimension does not match the index.")
        scores, indices = self._index.search(query_vector, min(top_k, len(self._chunks)))
        results = [
            VectorSearchResult(chunk=self._chunks[int(position)], score=float(score))
            for score, position in zip(scores[0], indices[0])
            if 0 <= int(position) < len(self._chunks)
        ]
        return results

    def save(self, directory: str | Path = DEFAULT_CACHE_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(asdict(self.manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)
