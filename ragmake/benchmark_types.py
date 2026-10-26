from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkPhase:
    name: str
    ingest_ms: float
    query_ms: float
    processed_documents: int
    skipped_documents: int
    chunks_upserted: int
    synopsis_rebuilt: int
    ingest_embedder_calls: int
    ingest_embedded_texts: int
    query_embedder_calls: int
    query_embedded_texts: int
    retrieved_chunks: int


@dataclass(slots=True)
class ChangedDocumentSavings:
    changed_documents: int
    unchanged_documents: int
    documents_reused: int
    document_reuse_rate: float
    stateful_ingest_ms: float
    stateless_ingest_ms: float
    ingest_ms_saved: float
    ingest_speedup_vs_stateless: float
    stateful_ingest_embedded_texts: int
    stateless_ingest_embedded_texts: int
    embedded_texts_avoided: int
    embedding_reuse_rate: float


@dataclass(slots=True)
class BenchmarkReport:
    backend: str
    document_count: int
    words_per_document: int
    change_fraction: float
    warm_skip_rate: float
    phases: list[BenchmarkPhase]
    changed_document_savings: ChangedDocumentSavings
