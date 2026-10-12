import argparse
import json
from pathlib import Path

from .chunking import BOOK_CHUNK_OVERLAP, BOOK_CHUNK_SIZE, build_book_chunks, build_chunks
from .corpus import list_corpus_profiles, load_corpus, load_prince_corpus
from .embeddings import SentenceTransformerEmbedder
from .generation import GeminiLLM
from .pipeline import PrinceRagPipeline
from .rag import build_grounded_answer
from .reranking import DEFAULT_RERANKER_CACHE, FlashRankReranker
from .vector_store import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    FaissVectorStore,
    StaleIndexError,
    hash_chunk_text,
)


DEFAULT_CALIBRATION_REPORT = Path("data/refusal_calibration_scores.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 RAG homework starter")
    parser.add_argument("--question", required=True, help="Question to ask over the local corpus")
    parser.add_argument("--chunk-size", type=int, default=140, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=30, help="Chunk overlap in characters")
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved chunks")
    parser.add_argument(
        "--corpus-profile",
        default="default",
        choices=[*list_corpus_profiles(), "prince"],
        help="Corpus profile: default, training-rag, or prince",
    )
    parser.add_argument("--book-chunk-size", type=int, default=BOOK_CHUNK_SIZE, help="Prince chunk size in characters")
    parser.add_argument("--book-overlap", type=int, default=BOOK_CHUNK_OVERLAP, help="Prince chunk overlap in characters")
    parser.add_argument("--candidate-n", type=int, default=20, help="Number of FAISS candidates before reranking")
    parser.add_argument("--refusal-threshold", type=float, default=None, help="Override the calibrated reranker threshold")
    parser.add_argument("--index-dir", default=str(DEFAULT_CACHE_DIR), help="Prince FAISS index directory")
    parser.add_argument("--reranker-cache-dir", default=str(DEFAULT_RERANKER_CACHE), help="FlashRank model cache directory")
    return parser.parse_args()


def _create_prince_pipeline(args: argparse.Namespace) -> PrinceRagPipeline:
    documents = load_prince_corpus()
    chunks = build_book_chunks(
        documents,
        chunk_size=args.book_chunk_size,
        overlap=args.book_overlap,
    )
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL_NAME)
    model_name = embedder.model_name
    index_dir = Path(args.index_dir)
    try:
        store = FaissVectorStore.load(
            index_dir,
            expected_text_hash=hash_chunk_text(chunks),
            chunk_size=args.book_chunk_size,
            overlap=args.book_overlap,
            model_name=model_name,
        )
    except (FileNotFoundError, StaleIndexError):
        store = FaissVectorStore.build(
            chunks,
            embedder,
            chunk_size=args.book_chunk_size,
            overlap=args.book_overlap,
        )
        store.save(index_dir)

    if args.refusal_threshold is None:
        report = json.loads(DEFAULT_CALIBRATION_REPORT.read_text(encoding="utf-8"))
        threshold = float(report["threshold"])
    else:
        threshold = args.refusal_threshold
    reranker = FlashRankReranker(cache_dir=args.reranker_cache_dir)
    return PrinceRagPipeline(
        store,
        embedder,
        reranker,
        refusal_threshold=threshold,
        llm=GeminiLLM(),
    )


def _print_prince_response(question: str, response: object) -> None:
    print("Question:")
    print(question)
    print("\nAnswer:")
    answer = getattr(response, "answer", None)
    print(answer if answer is not None else "Not generated.")
    print("\nCorpus profile: prince")
    print("\nRefused:", getattr(response, "refused"))
    print("Generation error:", getattr(response, "generation_error") or "None")
    citations = getattr(response, "citations")
    print("Citations:", ", ".join(citations) if citations else "None")
    print("\nRetrieved evidence:")
    evidence = getattr(response, "evidence")
    if not evidence:
        print("None")
    for item in evidence:
        chunk = item.chunk
        print(
            f"[{chunk.chunk_id}] chapter={chunk.chapter_label} "
            f"vector_score={item.vector_score:.6f} rerank_score={item.rerank_score:.8f}"
        )
        print(chunk.text)


def main() -> None:
    args = parse_args()
    if args.corpus_profile == "prince":
        pipeline = _create_prince_pipeline(args)
        response = pipeline.answer(
            args.question,
            candidate_n=args.candidate_n,
            top_k=args.top_k,
        )
        _print_prince_response(args.question, response)
        return

    corpus = load_corpus(args.corpus_profile)
    chunks = build_chunks(corpus, chunk_size=args.chunk_size, overlap=args.overlap)
    result = build_grounded_answer(args.question, chunks, top_k=args.top_k)

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(result.answer)
    print("\nCorpus profile:", args.corpus_profile)
    print("\nRefused:", result.refused)
    print("Citations:", ", ".join(result.citations) if result.citations else "None")


if __name__ == "__main__":
    main()
