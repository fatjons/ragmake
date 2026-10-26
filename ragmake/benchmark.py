from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .benchmark_core import run_benchmark
from .benchmark_types import BenchmarkPhase, BenchmarkReport


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark changed-document savings versus stateless recompilation")
    parser.add_argument("--documents", type=int, default=100, help="Number of synthetic documents")
    parser.add_argument("--words-per-document", type=int, default=240, help="Approximate document size")
    parser.add_argument("--change-fraction", type=float, default=0.1, help="Fraction of documents mutated between runs")
    parser.add_argument("--backend", choices=["memory", "sqlite"], default="sqlite", help="Storage backend for benchmark")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved chunks per query")
    parser.add_argument("--format", choices=["summary", "json"], default="summary", help="Output format")
    args = parser.parse_args()
    report = run_benchmark(args.documents, args.words_per_document, args.change_fraction, args.backend, args.top_k)
    print(render_summary(report) if args.format == "summary" else json.dumps(asdict(report), indent=2))


def render_summary(report: BenchmarkReport) -> str:
    phase_map = {phase.name: phase for phase in report.phases}
    savings = report.changed_document_savings
    lines = [
        "Changed-document savings",
        f"backend: {report.backend}",
        f"corpus: {report.document_count} docs, {savings.changed_documents} changed, {savings.unchanged_documents} unchanged",
        f"document reuse: {savings.documents_reused}/{report.document_count} ({_pct(savings.document_reuse_rate)})",
        f"embedding reuse: avoided {savings.embedded_texts_avoided}/{savings.stateless_ingest_embedded_texts} chunk embeddings ({_pct(savings.embedding_reuse_rate)})",
        f"ingest speedup vs stateless: {savings.ingest_speedup_vs_stateless:.2f}x ({savings.ingest_ms_saved:.3f} ms saved)",
        f"warm-cache skip rate: {_pct(report.warm_skip_rate)}",
        "",
        "Changed-path detail",
        _phase_line("stateful", phase_map["stateful_changed"]),
        _phase_line("stateless", phase_map["stateless_changed"]),
    ]
    return "\n".join(lines)


def _phase_line(label: str, phase: BenchmarkPhase) -> str:
    return f"{label}: processed={phase.processed_documents} skipped={phase.skipped_documents} chunks={phase.chunks_upserted} ingest_embeddings={phase.ingest_embedded_texts} ingest_ms={phase.ingest_ms:.3f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


__all__ = ["BenchmarkReport", "render_summary", "run_benchmark"]


if __name__ == "__main__":
    main()
