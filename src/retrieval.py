from dataclasses import dataclass
import re

from .chunking import Chunk


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


def lexical_score(query: str, chunk_text: str) -> float:
    q_tokens = tokenize(query)
    c_tokens = tokenize(chunk_text)
    if not q_tokens or not c_tokens:
        return 0.0

    q_set = set(q_tokens)
    overlap = sum(1 for token in c_tokens if token in q_set)
    return overlap / len(c_tokens)


def retrieve_top_k(query: str, chunks: list[Chunk], k: int = 3, min_score: float = 0.01) -> list[RetrievalResult]:
    if k <= 0:
        raise ValueError("k must be > 0")

    scored = [
        RetrievalResult(chunk=chunk, score=lexical_score(query, chunk.text))
        for chunk in chunks
    ]
    filtered = [item for item in scored if item.score >= min_score]
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[:k]
