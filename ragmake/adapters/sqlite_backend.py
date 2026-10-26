from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import ChunkRecord, RetrievedChunk


class SQLiteArtifactStore:
    """SQLite-backed artifact store for local and edge deployments."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._ensure_tables()

    def read_json(self, path: str) -> Any | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM rag_artifacts WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def write_json(self, path: str, data: Any) -> None:
        payload_json = json.dumps(data, ensure_ascii=True, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_artifacts(path, payload_json)
                VALUES(?, ?)
                ON CONFLICT(path) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (path, payload_json),
            )

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT path, payload_json
                FROM rag_artifacts
                WHERE path LIKE ?
                ORDER BY path
                """,
                (f"{prefix}%",),
            ).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

    def delete(self, path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rag_artifacts WHERE path = ?", (path,))

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_artifacts (
                    path TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


class SQLiteVectorStore:
    """SQLite vector store with Python-side cosine search.

    This is a practical non-cloud backend for small and medium corpora, local
    demos, reproducible experiments, and edge deployments.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._ensure_tables()

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        rows = []
        for chunk, vector in zip(chunks, vectors):
            rows.append(
                (
                    chunk.chunk_key,
                    chunk.corpus_id,
                    chunk.document_id,
                    chunk.chunk_id,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.token_start,
                    chunk.token_end,
                    chunk.content_hash,
                    json.dumps(chunk.metadata, ensure_ascii=True, sort_keys=True),
                    json.dumps([float(value) for value in vector]),
                )
            )
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO rag_chunks(
                    chunk_key,
                    corpus_id,
                    document_id,
                    chunk_id,
                    text,
                    char_start,
                    char_end,
                    token_start,
                    token_end,
                    content_hash,
                    metadata_json,
                    vector_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_key) DO UPDATE SET
                    corpus_id = excluded.corpus_id,
                    document_id = excluded.document_id,
                    chunk_id = excluded.chunk_id,
                    text = excluded.text,
                    char_start = excluded.char_start,
                    char_end = excluded.char_end,
                    token_start = excluded.token_start,
                    token_end = excluded.token_end,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    vector_json = excluded.vector_json
                """,
                rows,
            )

    def delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM rag_chunks WHERE chunk_key = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

    def list_chunks(self, corpus_id: str, limit: int | None = None) -> list[ChunkRecord]:
        query = """
            SELECT
                corpus_id,
                document_id,
                chunk_id,
                chunk_key,
                text,
                char_start,
                char_end,
                token_start,
                token_end,
                content_hash,
                metadata_json
            FROM rag_chunks
            WHERE corpus_id = ?
            ORDER BY document_id, chunk_id
        """
        params: list[Any] = [corpus_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def search(
        self,
        corpus_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    corpus_id,
                    document_id,
                    chunk_id,
                    chunk_key,
                    text,
                    char_start,
                    char_end,
                    token_start,
                    token_end,
                    content_hash,
                    metadata_json,
                    vector_json
                FROM rag_chunks
                WHERE corpus_id = ?
                """,
                (corpus_id,),
            ).fetchall()

        results: list[RetrievedChunk] = []
        for row in rows:
            chunk = self._row_to_chunk(row[:-1])
            vector = json.loads(row[-1])
            score = _cosine_similarity(query_vector, vector)
            results.append(RetrievedChunk(chunk=ChunkRecord(**asdict(chunk)), score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_key TEXT PRIMARY KEY,
                    corpus_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    token_start INTEGER NOT NULL,
                    token_end INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_corpus ON rag_chunks(corpus_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(corpus_id, document_id)"
            )

    def _row_to_chunk(self, row: Sequence[Any]) -> ChunkRecord:
        return ChunkRecord(
            corpus_id=row[0],
            document_id=row[1],
            chunk_id=int(row[2]),
            chunk_key=row[3],
            text=row[4],
            char_start=int(row[5]),
            char_end=int(row[6]),
            token_start=int(row[7]),
            token_end=int(row[8]),
            content_hash=row[9],
            metadata=json.loads(row[10]),
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
