from dataclasses import dataclass

from .chunking import Chunk
from .retrieval import retrieve_top_k


REFUSAL_MESSAGE = "I do not have enough grounded evidence in the provided corpus to answer that question."


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[str]
    refused: bool


def build_grounded_answer(question: str, chunks: list[Chunk], top_k: int = 2) -> RagAnswer:
    results = retrieve_top_k(question, chunks, k=top_k)
    if not results:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

    best = results[0]
    if best.score < 0.02:
        return RagAnswer(answer=REFUSAL_MESSAGE, citations=[], refused=True)

    supporting_points = []
    citations = []
    for item in results:
        citations.append(item.chunk.chunk_id)
        supporting_points.append(f"[{item.chunk.chunk_id}] {item.chunk.text}")

    answer = "Grounded response based on retrieved context:\n" + "\n".join(supporting_points)
    return RagAnswer(answer=answer, citations=citations, refused=False)
