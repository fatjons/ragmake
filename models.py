from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import SessionState


@dataclass(slots=True)
class SourceDocument:
    corpus_id: str
    document_id: str
    text: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    corpus_id: str
    document_id: str
    chunk_id: int
    chunk_key: str
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LedgerEntry:
    corpus_id: str
    document_id: str
    document_hash: str
    chunk_count: int
    chunk_ids: list[str]
    chunk_hashes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    last_error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        return cls(
            corpus_id=data["corpus_id"],
            document_id=data["document_id"],
            document_hash=data["document_hash"],
            chunk_count=int(data["chunk_count"]),
            chunk_ids=list(data.get("chunk_ids") or []),
            chunk_hashes=list(data.get("chunk_hashes") or []),
            metadata=dict(data.get("metadata") or {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            last_error=data.get("last_error"),
        )


@dataclass(slots=True)
class CorpusSynopsis:
    corpus_id: str
    content_signature: str
    document_count: int
    chunk_count: int
    synopsis_text: str
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusSynopsis:
        return cls(
            corpus_id=data["corpus_id"],
            content_signature=data["content_signature"],
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
            synopsis_text=data["synopsis_text"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float


@dataclass(slots=True)
class RuntimeContext:
    query: str
    corpus_id: str
    synopsis_text: str | None
    content_signature: str | None
    chunks: list[RetrievedChunk]
    session: SessionState | None = None


@dataclass(slots=True)
class IngestionReport:
    processed_documents: int = 0
    skipped_documents: int = 0
    chunks_upserted: int = 0
    synopses_rebuilt: list[str] = field(default_factory=list)
