from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone

from .chunking import WordChunker
from .interfaces import ArtifactStore, Embedder, SynopsisCompiler, VectorStore
from .models import CorpusSynopsis, IngestionReport, LedgerEntry, SourceDocument


class StatefulRAGCompiler:
    """Compile raw documents into reusable RAG state, including corpus nectar.

    The compiler tracks document and chunk hashes so unchanged content is never
    reprocessed. On each ingest it recomputes the corpus content signature and
    rebuilds the corpus synopsis (nectar) only when that signature changes.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        vector_store: VectorStore,
        embedder: Embedder,
        synopsis_compiler: SynopsisCompiler,
        chunker: WordChunker | None = None,
        namespace: str = ".stateful_rag",
    ):
        self.artifact_store = artifact_store
        self.vector_store = vector_store
        self.embedder = embedder
        self.synopsis_compiler = synopsis_compiler
        self.chunker = chunker or WordChunker()
        self.namespace = namespace.rstrip("/")

    def ingest_documents(
        self,
        documents: list[SourceDocument],
        force: bool = False,
        rebuild_synopses: bool = True,
    ) -> IngestionReport:
        report = IngestionReport()
        touched_corpora: set[str] = set()

        for document in documents:
            document_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            ledger_path = self._ledger_path(document.corpus_id, document.document_id)
            existing_data = self.artifact_store.read_json(ledger_path)
            existing_entry = LedgerEntry.from_dict(existing_data) if existing_data else None

            if existing_entry and existing_entry.document_hash == document_hash and not force:
                report.skipped_documents += 1
                continue

            if existing_entry and existing_entry.chunk_ids:
                self.vector_store.delete_chunk_ids(existing_entry.chunk_ids)

            chunks = self.chunker.chunk(document)
            vectors = self._embed_chunks(chunks)
            self.vector_store.upsert_chunks(chunks, vectors)

            now = _utcnow()
            entry = LedgerEntry(
                corpus_id=document.corpus_id,
                document_id=document.document_id,
                document_hash=document_hash,
                chunk_count=len(chunks),
                chunk_ids=[chunk.chunk_key for chunk in chunks],
                chunk_hashes=[chunk.content_hash for chunk in chunks],
                metadata=dict(document.metadata),
                created_at=existing_entry.created_at if existing_entry else now,
                updated_at=now,
            )
            self.artifact_store.write_json(ledger_path, asdict(entry))

            touched_corpora.add(document.corpus_id)
            report.processed_documents += 1
            report.chunks_upserted += len(chunks)

        if rebuild_synopses:
            for corpus_id in sorted(touched_corpora):
                refreshed = self.refresh_synopsis(corpus_id)
                if refreshed is not None:
                    report.synopses_rebuilt.append(corpus_id)

        return report

    def refresh_synopsis(
        self,
        corpus_id: str,
        max_chunks: int = 64,
    ) -> CorpusSynopsis | None:
        content_signature, document_count, chunk_count = self._compute_content_signature(corpus_id)
        if document_count == 0:
            return None

        synopsis_path = self._synopsis_path(corpus_id)
        existing_data = self.artifact_store.read_json(synopsis_path)
        existing = CorpusSynopsis.from_dict(existing_data) if existing_data else None

        if existing and existing.content_signature == content_signature:
            return existing

        chunks = self.vector_store.list_chunks(corpus_id, limit=max_chunks)
        synopsis_text = self.synopsis_compiler.compile_synopsis(
            corpus_id=corpus_id,
            chunks=chunks,
            previous_synopsis=existing.synopsis_text if existing else None,
        )
        now = _utcnow()
        synopsis = CorpusSynopsis(
            corpus_id=corpus_id,
            content_signature=content_signature,
            document_count=document_count,
            chunk_count=chunk_count,
            synopsis_text=synopsis_text,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self.artifact_store.write_json(synopsis_path, asdict(synopsis))
        return synopsis

    def load_synopsis(self, corpus_id: str) -> CorpusSynopsis | None:
        data = self.artifact_store.read_json(self._synopsis_path(corpus_id))
        return CorpusSynopsis.from_dict(data) if data else None

    def load_ledger_entry(self, corpus_id: str, document_id: str) -> LedgerEntry | None:
        data = self.artifact_store.read_json(self._ledger_path(corpus_id, document_id))
        return LedgerEntry.from_dict(data) if data else None

    def _embed_chunks(self, chunks) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(chunks)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, chunk in enumerate(chunks):
            cache_path = self._embedding_path(chunk.content_hash)
            cached = self.artifact_store.read_json(cache_path)
            if isinstance(cached, list):
                vectors[index] = [float(value) for value in cached]
                continue
            missing_indices.append(index)
            missing_texts.append(chunk.text)

        if missing_texts:
            embedded = self.embedder.embed_texts(missing_texts)
            for vector, index in zip(embedded, missing_indices):
                vectors[index] = [float(value) for value in vector]
                self.artifact_store.write_json(
                    self._embedding_path(chunks[index].content_hash),
                    vectors[index],
                )

        return [vector if vector is not None else [] for vector in vectors]

    def _compute_content_signature(self, corpus_id: str) -> tuple[str, int, int]:
        records = []
        chunk_count = 0
        for _, data in self.artifact_store.iter_json(self._ledger_prefix(corpus_id)):
            entry = LedgerEntry.from_dict(data)
            chunk_count += entry.chunk_count
            records.append(f"{entry.document_id}|{entry.document_hash}")
        records.sort()
        hasher = hashlib.sha256()
        for record in records:
            hasher.update(record.encode("utf-8"))
        return hasher.hexdigest(), len(records), chunk_count

    def _ledger_prefix(self, corpus_id: str) -> str:
        return f"{self.namespace}/ledger/{corpus_id}"

    def _ledger_path(self, corpus_id: str, document_id: str) -> str:
        return f"{self._ledger_prefix(corpus_id)}/{document_id}.json"

    def _embedding_path(self, content_hash: str) -> str:
        return f"{self.namespace}/cache/embeddings/{content_hash}.json"

    def _synopsis_path(self, corpus_id: str) -> str:
        return f"{self.namespace}/synopses/{corpus_id}.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
