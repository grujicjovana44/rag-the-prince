import pytest

from src.corpus import list_corpus_profiles, load_corpus


def test_list_corpus_profiles_contains_training_rag() -> None:
    profiles = list_corpus_profiles()
    assert "default" in profiles
    assert "training-rag" in profiles


def test_load_training_rag_corpus_has_expected_topic() -> None:
    docs = load_corpus("training-rag")
    joined = " ".join(doc.text for doc in docs).lower()
    assert "citation" in joined
    assert "judge" in joined


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        load_corpus("unknown-profile")
