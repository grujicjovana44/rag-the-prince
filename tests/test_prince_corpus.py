from pathlib import Path
import re

import pytest

from src.corpus import CorpusFormatError, load_prince_corpus


BOOK_PATH = Path(__file__).parents[1] / "data" / "book.txt"


def test_prince_corpus_contains_only_dedication_and_26_chapters() -> None:
    documents = load_prince_corpus(BOOK_PATH)

    assert len(documents) == 27
    assert documents[0].chapter_label == "DEDICATION"
    assert documents[0].chapter_title == "Dedication to Lorenzo de' Medici"
    assert [document.chapter_label for document in documents[1:]] == [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
        "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII",
        "XXIII", "XXIV", "XXV", "XXVI",
    ]
    assert [document.part_number for document in documents] == list(range(27))
    assert all(document.chapter_title for document in documents)
    assert all(document.text for document in documents)


def test_prince_corpus_excludes_front_matter_gutenberg_frame_and_appendices() -> None:
    documents = load_prince_corpus(BOOK_PATH)
    joined = "\n".join(document.text for document in documents).casefold()

    assert "project gutenberg" not in joined
    assert "table of contents" not in joined
    assert "he was the second son of bernardo di nicolo machiavelli" not in joined
    assert "the duke valentino had returned from lombardy" not in joined
    assert "the life of castruccio castracani" not in joined
    assert "to the magnificent lorenzo di piero de’ medici" in joined
    assert "all states, all powers" in joined
    assert "virtue against fury shall advance the fight" in joined


def test_prince_loader_rejects_missing_gutenberg_marker(tmp_path: Path) -> None:
    source = BOOK_PATH.read_text(encoding="utf-8")
    malformed = source.replace("*** START OF THE PROJECT GUTENBERG EBOOK THE PRINCE ***", "", 1)
    path = tmp_path / "missing-marker.txt"
    path.write_text(malformed, encoding="utf-8")

    with pytest.raises(CorpusFormatError, match="Gutenberg marker"):
        load_prince_corpus(path)


def test_prince_loader_rejects_unexpected_chapter_count(tmp_path: Path) -> None:
    source = BOOK_PATH.read_text(encoding="utf-8")
    lines = source.splitlines()
    chapter_heading_indexes = [
        index for index, line in enumerate(lines) if line.strip() == "CHAPTER XXVI."
    ]
    assert len(chapter_heading_indexes) == 1
    lines[chapter_heading_indexes[0]] = "CHAPTER XXVII."
    path = tmp_path / "missing-chapter.txt"
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(CorpusFormatError, match="Chapters I–XXVI"):
        load_prince_corpus(path)


def test_prince_corpus_contains_no_footnote_markers() -> None:
    documents = load_prince_corpus(BOOK_PATH)

    assert all(re.search(r"\[\d+\]", document.text) is None for document in documents)


def test_each_chapter_removes_balanced_footnote_definitions_and_references() -> None:
    documents = load_prince_corpus(BOOK_PATH)

    for document in documents:
        if document.chapter_label != "DEDICATION":
            assert document.footnote_definitions_removed == document.inline_markers_removed, document.chapter_label


def test_prince_corpus_removes_known_footnotes_but_preserves_book_text() -> None:
    documents = load_prince_corpus(BOOK_PATH)
    by_label = {document.chapter_label: document for document in documents}

    assert "Hiero II, born about 307 B.C." not in by_label["VI"].text
    assert "Duke Lodovico to raise insurrections" in by_label["III"].text
    assert "passed into Italy, this country was" in by_label["XI"].text
    assert "Lodovico[1]" not in by_label["III"].text
    assert "Italy,[1]" not in by_label["XI"].text
    assert re.search(r"(?m)1\. Some princes", by_label["XX"].text)
    assert by_label["XXVI"].text.endswith("Edward Dacre, 1640.")


def test_chapter_xviii_starts_with_chapter_text_not_translator_note() -> None:
    documents = load_prince_corpus(BOOK_PATH)
    chapter = next(document for document in documents if document.chapter_label == "XVIII")
    title, separator, body = chapter.text.partition("\n\n")

    assert separator
    assert title == chapter.chapter_title
    assert body.startswith("Every one admits how praiseworthy it is")
    assert "Burd" not in body


def test_chapters_ending_in_footnotes_end_with_machiavelli_prose() -> None:
    documents = load_prince_corpus(BOOK_PATH)
    by_label = {document.chapter_label: document for document in documents}
    expected_endings = {
        "VI": "he had but little in keeping.",
        "VII": "it was the cause of his ultimate ruin.",
        "XI": "his goodness and infinite other virtues.",
        "XII": "thus they have brought Italy to slavery and contempt.",
        "XX": "cares little about being hated by the people.",
        "XXI": "for this he must never consent to abate in anything.",
    }

    for label, ending in expected_endings.items():
        assert by_label[label].text.endswith(ending), label
