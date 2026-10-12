from dataclasses import dataclass
import re
from typing import Callable

from .corpus import Document

STARTER_CHUNK_SIZE = 140
STARTER_CHUNK_OVERLAP = 30
BOOK_CHUNK_SIZE = 800
BOOK_CHUNK_OVERLAP = 120
BOOK_MAX_CHUNK_CHARS = 880
BOOK_TARGET_MAX_TOKENS = 220
SENTENCE_CHUNK_SIZE = 800
SENTENCE_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    chapter_label: str | None = None
    chapter_title: str | None = None
    part_number: int | None = None
    chunk_number: int | None = None


def split_text_with_overlap(
    text: str,
    chunk_size: int = STARTER_CHUNK_SIZE,
    overlap: int = STARTER_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start += step

    return chunks


def split_text_with_overlap_words(
    text: str,
    chunk_size: int = BOOK_CHUNK_SIZE,
    overlap: int = BOOK_CHUNK_OVERLAP,
) -> list[str]:
    """Split on whitespace word boundaries, with a best-effort character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []
    too_long = next((word for word in words if len(word) > chunk_size), None)
    if too_long is not None:
        raise ValueError(f"A single word exceeds chunk_size ({len(too_long)} > {chunk_size}).")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        size = 0
        while end < len(words):
            next_size = len(words[end]) + (1 if end > start else 0)
            if end > start and size + next_size > chunk_size:
                break
            size += next_size
            end += 1

        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break

        next_start = end
        overlap_size = 0
        cursor = end - 1
        while cursor >= start:
            next_overlap_size = len(words[cursor]) + (1 if overlap_size else 0)
            if overlap_size + next_overlap_size > overlap:
                break
            overlap_size += next_overlap_size
            next_start = cursor
            cursor -= 1
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def build_chunks(
    documents: list[Document],
    chunk_size: int = STARTER_CHUNK_SIZE,
    overlap: int = STARTER_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        parts = split_text_with_overlap(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    chapter_label=doc.chapter_label,
                    chapter_title=doc.chapter_title,
                    part_number=doc.part_number,
                    chunk_number=i,
                )
            )
    return chunks


def build_book_chunks(
    documents: list[Document],
    chunk_size: int = BOOK_CHUNK_SIZE,
    overlap: int = BOOK_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Build word-aware chunks; each input document is kept as its own boundary."""
    if chunk_size > BOOK_MAX_CHUNK_CHARS:
        raise ValueError(
            f"Book chunk_size must be <= {BOOK_MAX_CHUNK_CHARS} characters as an offline safety limit."
        )

    chunks: list[Chunk] = []
    for doc in documents:
        if doc.chapter_label is None:
            raise ValueError(f"Book document {doc.doc_id!r} has no chapter metadata.")
        parts = split_text_with_overlap_words(doc.text, chunk_size=chunk_size, overlap=overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-{index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    chapter_label=doc.chapter_label,
                    chapter_title=doc.chapter_title,
                    part_number=doc.part_number,
                    chunk_number=index,
                )
            )
    oversized = [chunk for chunk in chunks if len(chunk.text) > BOOK_MAX_CHUNK_CHARS]
    if oversized:
        detail = ", ".join(f"{chunk.chunk_id} ({len(chunk.text)} chars)" for chunk in oversized)
        raise ValueError(f"Book chunks exceed the approximate character safety limit: {detail}.")
    return chunks


def split_text_by_sentences_with_overlap(
    text: str,
    chunk_size: int = SENTENCE_CHUNK_SIZE,
    overlap: int = SENTENCE_CHUNK_OVERLAP,
) -> list[str]:
    """Pack whole sentences into bounded chunks, repeating whole sentences at boundaries.

    A sentence longer than ``chunk_size`` is split at word boundaries so that
    malformed or unusually long source sentences still respect the size cap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= chunk_size:
            units.append(sentence)
        else:
            units.extend(split_text_with_overlap_words(sentence, chunk_size=chunk_size, overlap=0))

    chunks: list[str] = []
    start = 0
    while start < len(units):
        end = start
        size = 0
        while end < len(units):
            addition = len(units[end]) + (1 if end > start else 0)
            if end > start and size + addition > chunk_size:
                break
            size += addition
            end += 1
        chunks.append(" ".join(units[start:end]))
        if end == len(units):
            break
        next_start = end
        repeated_chars = 0
        cursor = end - 1
        while cursor >= start:
            addition = len(units[cursor]) + (1 if repeated_chars else 0)
            if repeated_chars + addition > overlap:
                break
            repeated_chars += addition
            next_start = cursor
            cursor -= 1
        start = next_start if next_start > start else start + 1
    return chunks


def build_sentence_chunks(
    documents: list[Document],
    chunk_size: int = SENTENCE_CHUNK_SIZE,
    overlap: int = SENTENCE_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Build chapter-aware chunks that prefer complete sentence boundaries."""
    chunks: list[Chunk] = []
    for doc in documents:
        parts = split_text_by_sentences_with_overlap(doc.text, chunk_size, overlap)
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-sentence-chunk-{index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=part,
                    chapter_label=doc.chapter_label,
                    chapter_title=doc.chapter_title,
                    part_number=doc.part_number,
                    chunk_number=index,
                )
            )
    return chunks


def find_chunks_over_token_limit(
    chunks: list[Chunk],
    token_counter: Callable[[str], int],
    max_tokens: int = BOOK_TARGET_MAX_TOKENS,
) -> list[tuple[str, int]]:
    """Return chunk IDs and token counts that exceed a supplied tokenizer limit."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    oversized = []
    for chunk in chunks:
        token_count = token_counter(chunk.text)
        if token_count > max_tokens:
            oversized.append((chunk.chunk_id, token_count))
    return oversized
