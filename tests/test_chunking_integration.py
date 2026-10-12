import pytest

from src.chunking import build_book_chunks, find_chunks_over_token_limit
from src.corpus import load_prince_corpus


@pytest.mark.integration
def test_book_chunks_fit_the_real_minilm_tokenizer_limit() -> None:
    transformers = pytest.importorskip("transformers", reason="optional tokenizer dependency is not installed")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"MiniLM tokenizer is not available in the local cache: {exc}")

    chunks = build_book_chunks(load_prince_corpus())
    too_long = find_chunks_over_token_limit(
        chunks,
        token_counter=lambda text: len(tokenizer(text, add_special_tokens=True)["input_ids"]),
    )

    assert not too_long, f"Chunks exceed the 220-token target: {too_long}"
