"""Stateful vs stateless ingest: what a persistent build state saves when little changes."""
from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from contextlib import ExitStack, closing
from pathlib import Path

from .adapters.sqlite import SQLiteArtifactStore, SQLiteVectorStore
from .compiler import Compiler
from .defaults import HashingEmbedder, HeuristicNectar
from .models import Document
from .storage import MemoryArtifactStore, MemoryVectorStore

_COLUMNS = ("processed", "skipped", "chunks_upserted", "embedded", "digests", "reduces", "ingest_ms")


class _CountingNectar(HeuristicNectar):
    digests = reduces = 0

    def digest(self, document):
        self.digests += 1
        return super().digest(document)

    def reduce(self, corpus_id, digests, previous):
        self.reduces += 1
        return super().reduce(corpus_id, digests, previous)


def _paragraph(rng: random.Random, topic: list[str], words: int) -> str:
    return " ".join(rng.choice(topic) if rng.random() < 0.7 else f"w{rng.randrange(5000)}" for _ in range(words))


def _documents(count: int, words: int, changed: int, edited: bool) -> list[Document]:
    docs = []
    for i in range(count):
        rng = random.Random(i)
        topic = [f"t{i}x{j}" for j in range(30)]
        paragraphs = [_paragraph(rng, topic, words // (3 + i % 3)) for _ in range(3 + i % 3)]
        if edited and i < changed:
            paragraphs[-1] = _paragraph(random.Random(-i - 1), topic, words // len(paragraphs))
        docs.append(Document("bench", f"doc-{i}", "\n\n".join(paragraphs), title=f"Document {i}"))
    return docs


def run_benchmark(documents: int = 100, words_per_document: int = 600, change_fraction: float = 0.1,
                  backend: str = "memory") -> dict:
    if not 0 <= change_fraction <= 1:
        raise ValueError("change_fraction must be in [0, 1]")
    if backend not in ("memory", "sqlite"):
        raise ValueError(f"unknown backend: {backend!r}")
    changed = round(documents * change_fraction)
    base = _documents(documents, words_per_document, changed, False)
    mutated = _documents(documents, words_per_document, changed, True)

    with tempfile.TemporaryDirectory() as tmp, ExitStack() as cleanup:
        def stack(name: str) -> tuple[Compiler, _CountingNectar]:
            nectar = _CountingNectar()
            if backend == "memory":
                stores = MemoryArtifactStore(), MemoryVectorStore()
            else:
                db = str(Path(tmp) / f"{name}.db")
                stores = [cleanup.enter_context(closing(s(db))) for s in (SQLiteArtifactStore, SQLiteVectorStore)]
            return Compiler(*stores, HashingEmbedder(), nectar), nectar

        def phase(compiler: Compiler, nectar: _CountingNectar, docs: list[Document]) -> dict:
            digests, reduces, start = nectar.digests, nectar.reduces, time.perf_counter()
            r = compiler.ingest(docs)
            return {"processed": r.processed, "skipped": r.skipped, "chunks_upserted": r.chunks_upserted,
                    "embedded": r.embedded, "digests": nectar.digests - digests, "reduces": nectar.reduces - reduces,
                    "ingest_ms": round((time.perf_counter() - start) * 1000, 1)}

        compiler, nectar = stack("stateful")
        phases = {"cold": phase(compiler, nectar, base), "warm": phase(compiler, nectar, base),
                  "changed": phase(compiler, nectar, mutated), "stateless": phase(*stack("stateless"), mutated)}
    stateless = phases["stateless"]["embedded"]
    return {"backend": backend, "documents": documents, "changed": changed, "phases": phases,
            "embedding_reuse": 1 - phases["changed"]["embedded"] / stateless if stateless else 0.0}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark stateful ingest against a stateless rebuild")
    p.add_argument("--documents", type=int, default=100)
    p.add_argument("--words-per-document", type=int, default=600)
    p.add_argument("--change-fraction", type=float, default=0.1)
    p.add_argument("--backend", choices=["memory", "sqlite"], default="sqlite")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        result = run_benchmark(args.documents, args.words_per_document, args.change_fraction, args.backend)
    except ValueError as e:
        p.error(str(e))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{result['backend']}: {result['documents']} documents, {result['changed']} changed")
    print(f"{'phase':<10}" + "".join(f"{c:>17}" for c in _COLUMNS))
    for name, row in result["phases"].items():
        print(f"{name:<10}" + "".join(f"{row[c]:>17}" for c in _COLUMNS))
    print(f"embedding reuse in changed phase: {result['embedding_reuse']:.1%} of stateless")


if __name__ == "__main__":
    main()
