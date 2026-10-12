import pytest

from src.chunking import (
    BOOK_CHUNK_OVERLAP,
    BOOK_CHUNK_SIZE,
    BOOK_MAX_CHUNK_CHARS,
    build_book_chunks,
    build_chunks,
    find_chunks_over_token_limit,
    split_text_with_overlap,
    split_text_with_overlap_words,
)
from src.corpus import Document, load_prince_corpus
from src.retrieval import retrieve_top_k


def test_split_text_with_overlap_returns_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = split_text_with_overlap(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 3


def test_split_text_with_overlap_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text_with_overlap("abc", chunk_size=10, overlap=10)


def test_word_aware_chunking_keeps_words_and_overlaps_adjacent_chunks() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = split_text_with_overlap_words(text, chunk_size=20, overlap=9)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in chunks)
    assert any(set(left.split()) & set(right.split()) for left, right in zip(chunks, chunks[1:]))
    assert all(word in " ".join(chunks) for word in text.split())


def test_word_aware_chunking_rejects_a_word_larger_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="single word exceeds chunk_size"):
        split_text_with_overlap_words("extraordinary", chunk_size=5, overlap=1)


def test_book_chunks_preserve_chapter_metadata_and_stable_ids() -> None:
    chapter = next(document for document in load_prince_corpus() if document.chapter_label == "XVIII")
    chunks = build_book_chunks([chapter])
    repeated = build_book_chunks([chapter])

    assert chunks
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in repeated]
    assert all(chunk.doc_id == chapter.doc_id for chunk in chunks)
    assert all(chunk.chapter_label == "XVIII" for chunk in chunks)
    assert all(chunk.chapter_title == chapter.chapter_title for chunk in chunks)
    assert all(chunk.part_number == chapter.part_number for chunk in chunks)
    assert [chunk.chunk_number for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_id.startswith("prince-chapter-xviii-chunk-") for chunk in chunks)


def test_book_chunking_keeps_documents_as_chapter_boundaries_and_enforces_char_guard() -> None:
    documents = load_prince_corpus()
    chunks = build_book_chunks(documents)

    assert all(len(chunk.text) <= BOOK_MAX_CHUNK_CHARS for chunk in chunks)
    assert all(chunk.chapter_label is not None for chunk in chunks)
    with pytest.raises(ValueError, match="offline safety limit"):
        build_book_chunks(documents[:1], chunk_size=BOOK_MAX_CHUNK_CHARS + 1, overlap=BOOK_CHUNK_OVERLAP)
    with pytest.raises(ValueError, match="no chapter metadata"):
        build_book_chunks([Document(doc_id="unknown", title="Unknown", text="text")])


def test_token_counter_reports_chunks_that_exceed_the_limit() -> None:
    chunks = build_book_chunks(load_prince_corpus()[:1], chunk_size=80, overlap=10)
    oversized = find_chunks_over_token_limit(chunks, token_counter=lambda text: len(text.split()), max_tokens=10)

    assert oversized
    assert all(token_count > 10 for _, token_count in oversized)


def test_outcome_one_compares_starter_and_book_chunking_retrieval() -> None:
    chapter = next(document for document in load_prince_corpus() if document.chapter_label == "XV")
    starter_chunks = build_chunks([chapter], chunk_size=140, overlap=30)
    book_chunks = build_book_chunks([chapter], chunk_size=BOOK_CHUNK_SIZE, overlap=BOOK_CHUNK_OVERLAP)
    query = "What qualities are praised or blamed in men and princes?"

    starter_results = retrieve_top_k(query, starter_chunks, k=2)
    book_results = retrieve_top_k(query, book_chunks, k=2)

    assert len(starter_chunks) > len(book_chunks)
    assert starter_results and book_results
    assert starter_results[0].chunk.text != book_results[0].chunk.text
