from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

from .models import ChunkRecord, RetrievedChunk


class ArtifactStore(Protocol):
    def read_json(self, path: str) -> Any | None:
        ...

    def write_json(self, path: str, data: Any) -> None:
        ...

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        ...

    def delete(self, path: str) -> None:
        ...


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SynopsisCompiler(Protocol):
    def compile_synopsis(
        self,
        corpus_id: str,
        chunks: Sequence[ChunkRecord],
        previous_synopsis: str | None = None,
    ) -> str:
        ...


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        ...

    def delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        ...

    def list_chunks(self, corpus_id: str, limit: int | None = None) -> list[ChunkRecord]:
        ...

    def search(
        self,
        corpus_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...
