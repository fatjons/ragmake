from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Callable, Sequence

from ..models import ChunkRecord, RetrievedChunk


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresArtifactStore:
    """Postgres JSONB artifact store for ledgers, synopses, caches, and sessions."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str = "public",
        table_name: str = "stateful_rag_artifacts",
        connect_fn: Callable[[], Any] | None = None,
    ):
        self.dsn = dsn
        self.schema = _safe_identifier(schema)
        self.table_name = _safe_identifier(table_name)
        self.connect_fn = connect_fn
        self._ensure_schema()

    def read_json(self, path: str) -> Any | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT payload_json FROM {self._qualified_table()} WHERE path = %s",
                (path,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return self._decode_payload(row[0])

    def write_json(self, path: str, data: Any) -> None:
        payload_json = json.dumps(data, ensure_ascii=True, sort_keys=True)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self._qualified_table()}(path, payload_json)
                VALUES(%s, %s::jsonb)
                ON CONFLICT(path) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (path, payload_json),
            )

    def iter_json(self, prefix: str):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT path, payload_json
                FROM {self._qualified_table()}
                WHERE path LIKE %s
                ORDER BY path
                """,
                (f"{prefix}%",),
            )
            rows = cur.fetchall()
        return [(row[0], self._decode_payload(row[1])) for row in rows]

    def delete(self, path: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._qualified_table()} WHERE path = %s",
                (path,),
            )

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qualified_table()} (
                    path TEXT PRIMARY KEY,
                    payload_json JSONB NOT NULL
                )
                """
            )

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.table_name}"

    def _connect(self):
        if self.connect_fn is not None:
            return self.connect_fn()

        import psycopg

        conn = psycopg.connect(self.dsn, autocommit=True)
        return conn

    @staticmethod
    def _decode_payload(value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value


class PgVectorStore:
    """Postgres + pgvector backend for chunk storage and semantic retrieval."""

    def __init__(
        self,
        dimensions: int,
        dsn: str | None = None,
        *,
        schema: str = "public",
        table_name: str = "stateful_rag_chunks",
        connect_fn: Callable[[], Any] | None = None,
        index_method: str = "hnsw",
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.dimensions = dimensions
        self.dsn = dsn
        self.schema = _safe_identifier(schema)
        self.table_name = _safe_identifier(table_name)
        self.connect_fn = connect_fn
        self.index_method = index_method
        self._ensure_schema()

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
                    [float(value) for value in vector],
                )
            )
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._qualified_table()}(
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
                    embedding
                )
                VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
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
                    embedding = excluded.embedding
                """,
                rows,
            )

    def delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"DELETE FROM {self._qualified_table()} WHERE chunk_key = %s",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

    def list_chunks(self, corpus_id: str, limit: int | None = None) -> list[ChunkRecord]:
        query = f"""
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
            FROM {self._qualified_table()}
            WHERE corpus_id = %s
            ORDER BY document_id, chunk_id
        """
        params: list[Any] = [corpus_id]
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def search(
        self,
        corpus_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
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
                    1 - (embedding <=> %s) AS score
                FROM {self._qualified_table()}
                WHERE corpus_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                ([float(value) for value in query_vector], corpus_id, [float(value) for value in query_vector], top_k),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(chunk=ChunkRecord(**asdict(self._row_to_chunk(row[:-1]))), score=float(row[-1]))
            for row in rows
        ]

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qualified_table()} (
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
                    metadata_json JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_corpus_idx ON {self._qualified_table()}(corpus_id)"
            )
            if self.index_method == "hnsw":
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw
                    ON {self._qualified_table()}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            elif self.index_method == "ivfflat":
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_ivfflat
                    ON {self._qualified_table()}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.table_name}"

    def _connect(self):
        if self.connect_fn is not None:
            return self.connect_fn()

        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(conn)
        return conn

    @staticmethod
    def _row_to_chunk(row: Sequence[Any]) -> ChunkRecord:
        metadata = row[10]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
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
            metadata=dict(metadata),
        )


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return value
