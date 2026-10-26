from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence

from .models import Chunk, Document, Hit


class ArtifactStore(Protocol):
    """Flat JSON key/value store. Paths are '/'-separated and already escaped."""

    def read_json(self, path: str) -> Any | None: ...
    def write_json(self, path: str, data: Any) -> None: ...
    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]: ...  # literal prefix match
    def delete(self, path: str) -> None: ...


class Embedder(Protocol):
    name: str  # embedding cache namespace: must change whenever vectors would

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def delete_chunks(self, keys: Sequence[str]) -> None: ...
    def search(self, corpus_id: str, vector: Sequence[float], top_k: int) -> list[Hit]: ...  # cosine, best first


class NectarCompiler(Protocol):
    def digest(self, document: Document) -> str: ...
    def reduce(self, corpus_id: str, digests: Sequence[tuple[str, str]], previous: str | None) -> str: ...
