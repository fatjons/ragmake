from __future__ import annotations

from .models import SourceDocument


def build_synthetic_corpus(corpus_id: str, document_count: int, words_per_document: int) -> list[SourceDocument]:
    return [
        SourceDocument(
            corpus_id=corpus_id,
            document_id=f"doc-{index:04d}.txt",
            title=f"Doc {index:04d}",
            text=_build_document_text(index, words_per_document),
            metadata={"domain": "synthetic", "rank": index},
        )
        for index in range(document_count)
    ]


def mutate_corpus(documents: list[SourceDocument], change_fraction: float) -> list[SourceDocument]:
    changed_count = int(round(len(documents) * change_fraction))
    return [
        SourceDocument(
            corpus_id=document.corpus_id,
            document_id=document.document_id,
            title=document.title,
            text=document.text + _change_suffix(index) if index < changed_count else document.text,
            metadata=dict(document.metadata),
        )
        for index, document in enumerate(documents)
    ]


def _change_suffix(index: int) -> str:
    return f" Update note {index}: drying guidance now depends on ambient humidity, primer stability, and calibration cadence."


def _build_document_text(index: int, words_per_document: int) -> str:
    theme = ["calibration", "primer", "fixation", "drying", "coupling", "thermal", "registration", "inspection"]
    words = []
    for offset in range(words_per_document):
        words.append(theme[(index + offset) % len(theme)])
        if offset % 17 == 0:
            words.append(f"doc{index}")
    return f"Document {index} explains operational guidance for print workflows. {' '.join(words)} Recommended practice is to validate calibration, observe primer behavior, and document drift before changing the process."
