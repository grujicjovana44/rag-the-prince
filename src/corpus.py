from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    chapter_label: str | None = None
    chapter_title: str | None = None
    part_number: int | None = None
    footnote_definitions_removed: int = 0
    inline_markers_removed: int = 0


class CorpusFormatError(ValueError):
    """Raised when the source book does not match its expected structure."""


GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK THE PRINCE ***"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK THE PRINCE ***"
APPENDIX_START = "DESCRIPTION OF THE METHODS ADOPTED BY THE DUKE VALENTINO"
EXPECTED_CHAPTERS = tuple(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII XXIII XXIV XXV XXVI".split()
)
CHAPTER_HEADING_RE = re.compile(r"^\s*CHAPTER\s+([IVXLCDM]+)\.?(?:\[(\d+)\])?\s*$", re.IGNORECASE)
FOOTNOTE_START_RE = re.compile(r"^\s*\[\d+\]")
INLINE_NOTE_RE = re.compile(r"\[\d+\]")


def _require_one_marker(text: str, marker: str) -> int:
    occurrences = [match.start() for match in re.finditer(re.escape(marker), text)]
    if len(occurrences) != 1:
        raise CorpusFormatError(
            f"Expected exactly one Gutenberg marker {marker!r}; found {len(occurrences)}."
        )
    return occurrences[0]


def _clean_paragraphs(text: str) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        cleaned = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if cleaned:
            paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def _remove_footnotes(text: str) -> tuple[str, int, int]:
    """Remove footnote paragraphs and inline references, preserving prose paragraphs."""
    kept_lines = []
    definitions_removed = 0
    in_definition = False
    for line in text.splitlines():
        if not line.strip():
            in_definition = False
            kept_lines.append("")
            continue
        if in_definition:
            continue
        if FOOTNOTE_START_RE.match(line):
            definitions_removed += 1
            in_definition = True
            continue
        kept_lines.append(line)

    text_without_definitions = "\n".join(kept_lines)
    inline_markers = INLINE_NOTE_RE.findall(text_without_definitions)
    text_without_markers = INLINE_NOTE_RE.sub("", text_without_definitions)
    return _clean_paragraphs(text_without_markers), definitions_removed, len(inline_markers)


def _find_body_title(lines: list[str]) -> int:
    """Find the body title from the standalone Chapter I heading that follows it."""
    matches = []
    for index, line in enumerate(lines):
        if line.strip().casefold() != "the prince":
            continue
        next_nonempty = next((item.strip() for item in lines[index + 1:] if item.strip()), "")
        chapter_match = CHAPTER_HEADING_RE.match(next_nonempty)
        if chapter_match and chapter_match.group(1).upper() == "I":
            matches.append(index)
    if len(matches) != 1:
        raise CorpusFormatError(
            "Expected exactly one body title 'THE PRINCE' followed by standalone Chapter I; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_prince_corpus(path: str | Path = "data/book.txt") -> list[Document]:
    """Load only the dedication and Chapters I–XXVI from the Gutenberg text."""
    source_path = Path(path)
    try:
        raw_text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusFormatError(f"Could not read The Prince source file {source_path}: {exc}") from exc
    except UnicodeError as exc:
        raise CorpusFormatError(f"The Prince source file is not valid UTF-8: {source_path}") from exc

    start = _require_one_marker(raw_text, GUTENBERG_START)
    end = _require_one_marker(raw_text, GUTENBERG_END)
    if start >= end:
        raise CorpusFormatError("Gutenberg END marker must occur after the START marker.")
    book_text = raw_text[start + len(GUTENBERG_START):end]
    lines = book_text.splitlines()

    chapter_matches: list[tuple[int, str, str | None]] = []
    for line_number, line in enumerate(lines):
        match = CHAPTER_HEADING_RE.match(line)
        if match:
            chapter_matches.append((line_number, match.group(1).upper(), match.group(2)))
    labels = [label for _, label, _ in chapter_matches]
    if tuple(labels) != EXPECTED_CHAPTERS:
        raise CorpusFormatError(
            "Expected exactly Chapters I–XXVI in order in the book body; "
            f"found {len(labels)} chapter headings: {', '.join(labels) or 'none'}."
        )

    prince_title = _find_body_title(lines)
    body_chapters = [
        (line_number, label, heading_marker)
        for line_number, label, heading_marker in chapter_matches
        if line_number > prince_title
    ]
    if tuple(label for _, label, _ in body_chapters) != EXPECTED_CHAPTERS:
        raise CorpusFormatError("Expected exactly Chapters I–XXVI after the body title 'THE PRINCE'.")

    dedication_candidates = [
        index for index, line in enumerate(lines[:prince_title])
        if line.strip().casefold() == "dedication"
    ]
    populated_dedications = []
    for index in dedication_candidates:
        section_lines = lines[index + 1:prince_title]
        first_content_line = next((line.strip() for line in section_lines if line.strip()), "")
        # In the contents, DEDICATION is followed directly by the THE PRINCE entry.
        # In the body, it is followed by the dedication's recipient/address.
        if first_content_line.casefold() == "the prince" or CHAPTER_HEADING_RE.match(first_content_line):
            continue
        dedication_text = _clean_paragraphs("\n".join(section_lines))
        if dedication_text:
            populated_dedications.append(dedication_text)
    if len(populated_dedications) != 1:
        raise CorpusFormatError(
            "Expected exactly one populated dedication section before the body title; "
            f"found {len(populated_dedications)}."
        )
    dedication_text = populated_dedications[0]

    appendix_matches = [
        i for i, line in enumerate(lines)
        if i > body_chapters[-1][0] and line.strip().upper().startswith(APPENDIX_START)
    ]
    if len(appendix_matches) != 1:
        raise CorpusFormatError(
            f"Expected exactly one post-Chapter-XXVI appendix marker; found {len(appendix_matches)}."
        )
    appendix_line = appendix_matches[0]
    if appendix_line <= body_chapters[-1][0]:
        raise CorpusFormatError("The Duke Valentino appendix marker must follow Chapter XXVI.")

    documents = [
        Document(
            doc_id="prince-dedication",
            title="The Prince — Dedication",
            text=dedication_text,
            chapter_label="DEDICATION",
            chapter_title="Dedication to Lorenzo de' Medici",
            part_number=0,
        )
    ]

    for index, (line_number, label, heading_marker) in enumerate(body_chapters):
        section_end = body_chapters[index + 1][0] if index + 1 < len(body_chapters) else appendix_line
        heading_lines = lines[line_number + 1:section_end]
        title_lines = []
        for line in heading_lines:
            if not line.strip():
                if title_lines:
                    break
                continue
            title_lines.append(line.strip())
        chapter_title = " ".join(title_lines).strip()
        if not chapter_title:
            raise CorpusFormatError(f"Chapter {label} has no title in the book body.")
        raw_chapter_text = "\n".join([chapter_title, *heading_lines[len(title_lines) + 1:]])
        chapter_text, definitions_removed, inline_markers_removed = _remove_footnotes(raw_chapter_text)
        # Chapter XVIII's [1] is attached to its heading, so it is not in raw_chapter_text.
        inline_markers_removed += int(heading_marker is not None)
        if not chapter_text:
            raise CorpusFormatError(f"Chapter {label} has no body text.")
        if definitions_removed != inline_markers_removed:
            raise CorpusFormatError(
                f"Chapter {label} has unbalanced footnote markers: "
                f"{definitions_removed} definitions and {inline_markers_removed} inline markers removed."
            )
        documents.append(
            Document(
                doc_id=f"prince-chapter-{label.lower()}",
                title=f"The Prince — Chapter {label}: {chapter_title}",
                text=chapter_text,
                chapter_label=label,
                chapter_title=chapter_title,
                part_number=index + 1,
                footnote_definitions_removed=definitions_removed,
                inline_markers_removed=inline_markers_removed,
            )
        )

    return documents


def load_default_corpus() -> list[Document]:
    """Return a small default corpus for local RAG practice.

    Students can replace or extend this corpus with course-approved sources.
    """
    return [
        Document(
            doc_id="doc-1",
            title="RAG Fundamentals",
            text=(
                "Retrieval augmented generation combines retrieval with text generation. "
                "The model should answer using retrieved evidence rather than unsupported assumptions."
            ),
        ),
        Document(
            doc_id="doc-2",
            title="Chunking and Metadata",
            text=(
                "Chunking splits documents into smaller parts. Good chunk size and overlap improve retrieval quality. "
                "Metadata helps filtering and source traceability."
            ),
        ),
        Document(
            doc_id="doc-3",
            title="Refusal Behavior",
            text=(
                "If retrieved context does not support a claim, the system should refuse and ask for better evidence. "
                "This reduces hallucinations and increases trust."
            ),
        ),
    ]


def load_training_rag_corpus() -> list[Document]:
    """Return a corpus profile inspired by training_rag notebook examples.

    This profile emphasizes citations, source metadata, and evaluation framing.
    """
    return [
        Document(
            doc_id="tr-doc-1",
            title="RAG with Citations",
            text=(
                "RAG with citations should retrieve relevant documents with source metadata and include references in the answer. "
                "Citations improve transparency and trust because users can verify evidence."
            ),
        ),
        Document(
            doc_id="tr-doc-2",
            title="Structured Citation Output",
            text=(
                "A cited answer can include answer text and list of source titles. "
                "A common format is an answer body followed by a References section."
            ),
        ),
        Document(
            doc_id="tr-doc-3",
            title="LLM-as-a-Judge Evaluation",
            text=(
                "LLM-as-a-judge can evaluate answer quality, citation correctness, and grounding behavior at scale. "
                "It is useful but introduces bias, so metrics should be interpreted carefully."
            ),
        ),
        Document(
            doc_id="tr-doc-4",
            title="Refusal and Grounding",
            text=(
                "When retrieved evidence is missing or irrelevant, the system should refuse unsupported claims. "
                "This behavior reduces hallucinations and aligns with safe RAG practice."
            ),
        ),
    ]


def list_corpus_profiles() -> list[str]:
    return ["default", "training-rag"]


def load_corpus(profile: str = "default") -> list[Document]:
    if profile == "default":
        return load_default_corpus()
    if profile == "training-rag":
        return load_training_rag_corpus()
    raise ValueError(f"Unknown corpus profile: {profile}")
