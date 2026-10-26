from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import quote

from .chunking import Chunker
from .interfaces import ArtifactStore, Embedder, NectarCompiler, VectorStore
from .models import Chunk, Document, IngestReport, LedgerEntry, Nectar


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _q(x: str) -> str:
    return quote(x, safe="")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Compiler:
    def __init__(self, artifacts: ArtifactStore, vectors: VectorStore, embedder: Embedder,
                 nectar: NectarCompiler, chunker: Chunker | None = None, namespace: str = "ragmake"):
        self.artifacts, self.vectors, self.embedder, self.nectar = artifacts, vectors, embedder, nectar
        self.chunker = chunker or Chunker()
        self.ns = namespace
        # The toolchain is a build input: changing any of it rebuilds every document.
        self.recipe = [embedder.name, self.chunker.name, nectar.name]

    def _ledger_path(self, corpus_id: str, document_id: str) -> str:
        return f"{self.ns}/ledger/{_q(corpus_id)}/{_q(document_id)}.json"

    def _entries(self, corpus_id: str) -> list[LedgerEntry]:
        prefix = f"{self.ns}/ledger/{_q(corpus_id)}/"  # trailing "/" so corpus "abc" misses "abcd"
        return sorted((LedgerEntry(**d) for _, d in self.artifacts.iter_json(prefix)), key=lambda e: e.document_id)

    def ledger(self, corpus_id: str, document_id: str) -> LedgerEntry | None:
        data = self.artifacts.read_json(self._ledger_path(corpus_id, document_id))
        return LedgerEntry(**data) if data else None

    def load_nectar(self, corpus_id: str) -> Nectar | None:
        data = self.artifacts.read_json(f"{self.ns}/nectar/{_q(corpus_id)}.json")
        return Nectar(**data) if data else None

    def ingest(self, documents: Iterable[Document], *, force: bool = False, prune: bool = False) -> IngestReport:
        report = IngestReport()
        seen: dict[str, set[str]] = {}
        touched: set[str] = set()
        for doc in documents:
            seen.setdefault(doc.corpus_id, set()).add(doc.document_id)
            if self._process(doc, force, report):
                touched.add(doc.corpus_id)
        if prune:
            for corpus_id, ids in seen.items():
                gone = [e.document_id for e in self._entries(corpus_id) if e.document_id not in ids]
                if self._remove(corpus_id, gone, report):
                    touched.add(corpus_id)
        for corpus_id in sorted(touched):
            if self._refresh(corpus_id, False)[1]:
                report.nectars_rebuilt.append(corpus_id)
        return report

    def remove(self, corpus_id: str, document_ids: Iterable[str]) -> int:
        report = IngestReport()
        if self._remove(corpus_id, document_ids, report):
            self._refresh(corpus_id, False)
        return report.removed

    def refresh_nectar(self, corpus_id: str, force: bool = False) -> Nectar | None:
        return self._refresh(corpus_id, force)[0]

    def _process(self, doc: Document, force: bool, report: IngestReport) -> bool:
        document_hash = _sha(json.dumps([doc.title, doc.text, doc.metadata, self.recipe], sort_keys=True))
        old = self.ledger(doc.corpus_id, doc.document_id)
        if old and old.document_hash == document_hash and not force:
            report.skipped += 1
            return False
        chunks = self.chunker.chunk(doc)
        stale = sorted(set(old.chunk_keys if old else ()) - {c.key for c in chunks})
        vectors = self._embed(chunks, report)
        if stale:
            self.vectors.delete_chunks(stale)
        self.vectors.upsert_chunks(chunks, vectors)
        entry = LedgerEntry(doc.corpus_id, doc.document_id, document_hash, doc.title,
                            self.nectar.digest(doc), [c.key for c in chunks], _utcnow())
        self.artifacts.write_json(self._ledger_path(doc.corpus_id, doc.document_id), asdict(entry))
        report.processed += 1
        report.chunks_upserted += len(chunks)
        report.chunks_deleted += len(stale)
        return True

    def _embed(self, chunks: list[Chunk], report: IngestReport) -> list[list[float]]:
        def path(content_hash: str) -> str:
            return f"{self.ns}/embeddings/{_q(self.embedder.name)}/{content_hash}.json"

        vecs = {c.content_hash: self.artifacts.read_json(path(c.content_hash)) for c in chunks}
        todo = {c.content_hash: c.text for c in chunks if vecs[c.content_hash] is None}
        if todo:
            fresh = self.embedder.embed_texts(list(todo.values()))
            if len(fresh) != len(todo):
                raise ValueError(f"embedder returned {len(fresh)} vectors for {len(todo)} texts")
            for content_hash, vector in zip(todo, fresh):
                vecs[content_hash] = list(vector)
                self.artifacts.write_json(path(content_hash), vecs[content_hash])
            report.embedded += len(todo)
        return [vecs[c.content_hash] for c in chunks]

    def _remove(self, corpus_id: str, document_ids: Iterable[str], report: IngestReport) -> int:
        removed = 0
        for document_id in document_ids:
            entry = self.ledger(corpus_id, document_id)
            if entry is None:
                continue
            self.vectors.delete_chunks(entry.chunk_keys)
            self.artifacts.delete(self._ledger_path(corpus_id, document_id))
            report.chunks_deleted += len(entry.chunk_keys)
            removed += 1
        report.removed += removed
        return removed

    def _refresh(self, corpus_id: str, force: bool) -> tuple[Nectar | None, bool]:
        """Return (nectar, rebuilt)."""
        entries = self._entries(corpus_id)
        path = f"{self.ns}/nectar/{_q(corpus_id)}.json"
        if not entries:
            self.artifacts.delete(path)
            return None, False
        signature = _sha(json.dumps([[e.document_id, e.document_hash] for e in entries]))
        stored = self.load_nectar(corpus_id)
        if stored and stored.signature == signature and not force:
            return stored, False
        text = self.nectar.reduce(corpus_id, [(e.document_id, e.digest) for e in entries],
                                  stored.text if stored else None)
        catalog = [{"document_id": e.document_id, "title": e.title, "digest": e.digest} for e in entries]
        nectar = Nectar(corpus_id, signature, len(entries), sum(len(e.chunk_keys) for e in entries),
                        text, catalog, _utcnow())
        self.artifacts.write_json(path, asdict(nectar))
        return nectar, True
