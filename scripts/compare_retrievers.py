"""Run a local, same-query comparison of multi-vector and parent retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import build_book_chunks, split_text_with_overlap_words
from src.corpus import Document, load_prince_corpus
from src.embeddings import SentenceTransformerEmbedder
from src.reranking import FlashRankReranker, retrieve_and_rerank
from src.vector_store import DEFAULT_MODEL_NAME, FaissVectorStore


QUESTION = "What qualities are praised or blamed in men and princes?"


def compare(*, offline: bool = False) -> str:
    if offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = next(doc for doc in load_prince_corpus() if doc.chapter_label == "XV")
    # Parent units are 1,600-character windows; each is indexed through several
    # 500/80 child vectors. Both retrievers use these same vectors and ranking.
    parent_texts = split_text_with_overlap_words(source.text, chunk_size=1600, overlap=200)
    parents = [
        Document(
            doc_id=f"{source.doc_id}-parent-{i}",
            title=source.title,
            text=text,
            chapter_label=source.chapter_label,
            chapter_title=source.chapter_title,
            part_number=source.part_number,
        )
        for i, text in enumerate(parent_texts)
    ]
    children = build_book_chunks(parents, chunk_size=500, overlap=80)
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL_NAME)
    store = FaissVectorStore.build(children, embedder, chunk_size=500, overlap=80)
    reranker = FlashRankReranker()
    ranked = retrieve_and_rerank(QUESTION, store, embedder, reranker, candidate_n=20, top_k=6)

    # Multi-vector returns the matching child passages. Parent retrieval groups
    # those same child matches and returns their larger parent context.
    best_per_parent = {}
    for item in ranked:
        prior = best_per_parent.get(item.chunk.doc_id)
        if prior is None or item.rerank_score > prior.rerank_score:
            best_per_parent[item.chunk.doc_id] = item
    parent_ranked = sorted(best_per_parent.values(), key=lambda item: item.rerank_score, reverse=True)[:3]
    parent_by_id = {doc.doc_id: doc for doc in parents}

    lines = [
        f"Query: {QUESTION}",
        "Shared setup: Chapter XV; MiniLM cosine candidates; FlashRank reranking; same 500/80 child vectors.",
        "Multi-vector retrieval returns the highest-ranked child passages; parent retrieval groups those hits and returns the containing 1,600-character parent window.",
        "",
        "Multi-vector (child passages):",
    ]
    for rank, item in enumerate(ranked[:3], 1):
        preview = item.chunk.text[:210].replace("\n", " ")
        lines.append(
            f"{rank}. {item.chunk.chunk_id} parent={item.chunk.doc_id} vector={item.vector_score:.6f} rerank={item.rerank_score:.6f}; {preview}"
        )
    lines.extend(["", "Parent retrieval (expanded context):"])
    for rank, item in enumerate(parent_ranked, 1):
        parent = parent_by_id[item.chunk.doc_id]
        preview = parent.text[:210].replace("\n", " ")
        lines.append(
            f"{rank}. {parent.doc_id} matched_child={item.chunk.chunk_id} rerank={item.rerank_score:.6f}; parent_chars={len(parent.text)}; {preview}"
        )
    lines.extend(
        [
            "",
            "Interpretation: multi-vector gives precise, compact evidence and can surface multiple passages from one chapter; parent retrieval restores surrounding context and may reduce boundary loss, but returns more text and can include irrelevant material. These are retrieval examples, not a labeled benchmark; assess answer coverage and context size together.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use cached embedding and reranker models only.")
    args = parser.parse_args()
    print(compare(offline=args.offline))


if __name__ == "__main__":
    main()
