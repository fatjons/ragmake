from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter

from .adapters.sqlite_backend import SQLiteArtifactStore, SQLiteVectorStore
from .benchmark_corpus import build_synthetic_corpus, mutate_corpus
from .benchmark_types import BenchmarkPhase, BenchmarkReport, ChangedDocumentSavings
from .chunking import WordChunker
from .compiler import StatefulRAGCompiler
from .defaults import HashingEmbedder, HeuristicSynopsisCompiler
from .runtime import StatefulRAGRuntime
from .storage import InMemoryArtifactStore, InMemoryVectorStore


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dimensions: int = 64):
        super().__init__(dimensions=dimensions)
        self.calls = 0
        self.texts = 0

    def embed_texts(self, texts):
        text_list = list(texts)
        self.calls += len(text_list) > 0
        self.texts += len(text_list)
        return super().embed_texts(text_list)

    def reset_counters(self) -> None:
        self.calls = 0
        self.texts = 0


class CountingSynopsisCompiler(HeuristicSynopsisCompiler):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def compile_synopsis(self, corpus_id, chunks, previous_synopsis=None):
        self.calls += 1
        return super().compile_synopsis(corpus_id, chunks, previous_synopsis=previous_synopsis)

    def reset_counters(self) -> None:
        self.calls = 0


def run_benchmark(
    document_count: int = 100,
    words_per_document: int = 240,
    change_fraction: float = 0.1,
    backend: str = "sqlite",
    top_k: int = 5,
) -> BenchmarkReport:
    if not 0.0 <= change_fraction <= 1.0:
        raise ValueError("change_fraction must be between 0.0 and 1.0")
    corpus_id = "benchmark"
    base_documents = build_synthetic_corpus(corpus_id, document_count, words_per_document)
    changed_documents = mutate_corpus(base_documents, change_fraction)
    stateful_stack = _build_stack(backend)
    phases = [
        _run_phase("stateful_cold", base_documents, "How should calibration and primer coupling be handled?", corpus_id, top_k, stateful_stack),
        _run_phase("stateful_warm", base_documents, "How should calibration and primer coupling be handled?", corpus_id, top_k, stateful_stack),
        _run_phase("stateful_changed", changed_documents, "What changed in the drying guidance?", corpus_id, top_k, stateful_stack),
        _run_phase("stateless_base", base_documents, "How should calibration and primer coupling be handled?", corpus_id, top_k, _build_stack(backend)),
        _run_phase("stateless_changed", changed_documents, "What changed in the drying guidance?", corpus_id, top_k, _build_stack(backend)),
    ]
    warm_phase = _phase_by_name(phases, "stateful_warm")
    changed_phase = _phase_by_name(phases, "stateful_changed")
    stateless_phase = _phase_by_name(phases, "stateless_changed")
    return BenchmarkReport(
        backend=backend,
        document_count=document_count,
        words_per_document=words_per_document,
        change_fraction=round(change_fraction, 4),
        warm_skip_rate=_rate(warm_phase.skipped_documents, document_count),
        phases=phases,
        changed_document_savings=_build_changed_document_savings(document_count, changed_phase, stateless_phase),
    )


def _build_stack(backend: str) -> dict:
    if backend == "memory":
        artifact_store = InMemoryArtifactStore()
        vector_store = InMemoryVectorStore()
    elif backend == "sqlite":
        tempdir = tempfile.TemporaryDirectory(prefix="stateful-rag-bench-")
        db_path = Path(tempdir.name) / "stateful_rag.sqlite"
        artifact_store = SQLiteArtifactStore(db_path)
        vector_store = SQLiteVectorStore(db_path)
        artifact_store._tempdir = tempdir
        vector_store._tempdir = tempdir
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    embedder = CountingEmbedder()
    synopsis_compiler = CountingSynopsisCompiler()
    compiler = StatefulRAGCompiler(
        artifact_store=artifact_store,
        vector_store=vector_store,
        embedder=embedder,
        synopsis_compiler=synopsis_compiler,
        chunker=WordChunker(max_words=80, overlap_words=10),
    )
    return {
        "compiler": compiler,
        "runtime": StatefulRAGRuntime(compiler, vector_store, embedder),
        "embedder": embedder,
        "synopsis_compiler": synopsis_compiler,
    }


def _run_phase(name: str, documents: list, query: str, corpus_id: str, top_k: int, stack: dict) -> BenchmarkPhase:
    embedder = stack["embedder"]
    synopsis_compiler = stack["synopsis_compiler"]
    embedder.reset_counters()
    synopsis_compiler.reset_counters()
    ingest_start = perf_counter()
    report = stack["compiler"].ingest_documents(documents)
    ingest_ms = (perf_counter() - ingest_start) * 1000.0
    ingest_calls, ingest_texts = embedder.calls, embedder.texts
    query_start = perf_counter()
    context = stack["runtime"].build_context(query=query, corpus_id=corpus_id, top_k=top_k)
    query_ms = (perf_counter() - query_start) * 1000.0
    return BenchmarkPhase(
        name=name,
        ingest_ms=round(ingest_ms, 3),
        query_ms=round(query_ms, 3),
        processed_documents=report.processed_documents,
        skipped_documents=report.skipped_documents,
        chunks_upserted=report.chunks_upserted,
        synopsis_rebuilt=len(report.synopses_rebuilt),
        ingest_embedder_calls=ingest_calls,
        ingest_embedded_texts=ingest_texts,
        query_embedder_calls=embedder.calls - ingest_calls,
        query_embedded_texts=embedder.texts - ingest_texts,
        retrieved_chunks=len(context.chunks),
    )


def _build_changed_document_savings(document_count: int, changed_phase: BenchmarkPhase, stateless_phase: BenchmarkPhase) -> ChangedDocumentSavings:
    avoided = max(0, stateless_phase.ingest_embedded_texts - changed_phase.ingest_embedded_texts)
    unchanged_documents = changed_phase.skipped_documents
    return ChangedDocumentSavings(
        changed_documents=changed_phase.processed_documents,
        unchanged_documents=unchanged_documents,
        documents_reused=unchanged_documents,
        document_reuse_rate=_rate(unchanged_documents, document_count),
        stateful_ingest_ms=changed_phase.ingest_ms,
        stateless_ingest_ms=stateless_phase.ingest_ms,
        ingest_ms_saved=round(max(0.0, stateless_phase.ingest_ms - changed_phase.ingest_ms), 3),
        ingest_speedup_vs_stateless=round(stateless_phase.ingest_ms / max(0.001, changed_phase.ingest_ms), 4),
        stateful_ingest_embedded_texts=changed_phase.ingest_embedded_texts,
        stateless_ingest_embedded_texts=stateless_phase.ingest_embedded_texts,
        embedded_texts_avoided=avoided,
        embedding_reuse_rate=_rate(avoided, stateless_phase.ingest_embedded_texts),
    )


def _phase_by_name(phases: list[BenchmarkPhase], name: str) -> BenchmarkPhase:
    return next(phase for phase in phases if phase.name == name)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 4)
