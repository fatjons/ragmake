from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    corpus_id: str
    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    corpus_id: str
    document_id: str
    chunk_id: int  # ordinal within the document
    key: str  # unique across the store; changes when position or content changes
    text: str
    char_start: int
    char_end: int
    content_hash: str  # sha256 of text; keys the embedding cache
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class LedgerEntry:
    """What the compiler last built for one document."""

    corpus_id: str
    document_id: str
    document_hash: str  # covers title, text, metadata and the build recipe
    title: str | None
    digest: str  # per-document nectar
    chunk_keys: list[str]
    updated_at: str


@dataclass(slots=True)
class Nectar:
    """Corpus-level memory: a summary reduced from every digest, plus the catalog."""

    corpus_id: str
    signature: str  # hash of sorted (document_id, document_hash)
    document_count: int
    chunk_count: int
    text: str
    catalog: list[dict[str, Any]]  # [{document_id, title, digest}] sorted by id
    updated_at: str


@dataclass(slots=True)
class IngestReport:
    processed: int = 0
    skipped: int = 0
    removed: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    embedded: int = 0  # texts actually sent to the embedder (cache misses)
    nectars_rebuilt: list[str] = field(default_factory=list)
