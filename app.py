"""Simple baby-pink Gradio interface for the Prince RAG pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.cli import _create_prince_pipeline


EXAMPLES = [
    "What does Chapter II say about hereditary principalities and change?",
    "Why does Machiavelli call mercenary arms useless and dangerous?",
    "What names of parents were given for Castruccio Castracani?",
]

CUSTOM_CSS = """
.gradio-container {
    max-width: 940px !important;
    margin: 0 auto !important;
    background: linear-gradient(145deg, #fff8fb 0%, #ffedf4 100%);
}
#app-header { text-align: center; padding: 16px 0 8px; }
#app-header h1 { color: #a74768; }
#app-header p { color: #805968; }
#answer-card, #evidence-card {
    border: 1px solid #f2c9d8 !important;
    border-radius: 18px !important;
    background: #fffafd !important;
}
button.primary {
    background: #e98cad !important;
    border-color: #e98cad !important;
    color: white !important;
}
"""


def _pipeline_args() -> argparse.Namespace:
    return argparse.Namespace(
        book_chunk_size=800,
        book_overlap=120,
        index_dir=".cache/prince-faiss",
        refusal_threshold=None,
        reranker_cache_dir=".cache/flashrank",
    )


def _format_evidence(response: object) -> str:
    evidence = getattr(response, "evidence", ())
    if not evidence:
        return "Nema pronađenih dokaza."

    sections = []
    for item in evidence:
        chunk = item.chunk
        sections.append(
            f"### `{chunk.chunk_id}` · poglavlje {chunk.chapter_label}\n"
            f"*Rerank skor: {item.rerank_score:.4f}*\n\n"
            f"> {chunk.text.replace(chr(10), ' ')}"
        )
    return "\n\n---\n\n".join(sections)


def _is_refusal(response: object) -> bool:
    if getattr(response, "refused", False):
        return True
    answer = getattr(response, "answer", None)
    if not isinstance(answer, str):
        return False
    normalized = answer.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "not enough grounded evidence",
            "evidence is insufficient",
            "insufficient evidence to answer",
            "nemam dovoljno dokaza",
            "nema dovoljno informacija",
        )
    )


def create_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio nije instaliran. Instaliraj zavisnosti iz requirements.txt."
        ) from exc

    load_dotenv(ROOT / ".env", override=False)
    os.chdir(ROOT)
    pipeline = _create_prince_pipeline(_pipeline_args())

    def answer_question(question: str) -> tuple[str, str]:
        if not question.strip():
            return "Unesi pitanje o knjizi.", "Nema pronađenih dokaza."

        response = pipeline.answer(question.strip(), candidate_n=20, top_k=4)
        if response.answer is not None:
            answer = response.answer
        elif response.generation_error:
            answer = f"Generisanje nije uspelo: {response.generation_error}"
        else:
            answer = "Odgovor nije generisan."
        # Refused questions should not look like they have supporting citations.
        evidence = "" if _is_refusal(response) else _format_evidence(response)
        return answer, evidence

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(
            label="Pitanje",
            placeholder="Na primer: What does Chapter II say about hereditary principalities?",
            lines=2,
        ),
        outputs=[
            gr.Markdown(label="Odgovor", elem_id="answer-card"),
            gr.Markdown(label="Pronađeni dokazi", elem_id="evidence-card"),
        ],
        title="The Prince · RAG",
        description="Pitaj o Machiavellijevoj knjizi. Za odgovore se prikazuju pronađeni citati i dokazi.",
        examples=EXAMPLES,
        submit_btn="Pitaj knjigu",
        clear_btn="Obriši",
    )
    # Do not impose an application-side cap on how many questions can wait.
    return interface.queue(max_size=None)


if __name__ == "__main__":
    import gradio as gr

    load_dotenv(ROOT / ".env", override=False)
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        share=False,
        theme=gr.themes.Soft(primary_hue="pink", secondary_hue="rose"),
        css=CUSTOM_CSS,
    )
