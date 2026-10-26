from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Sequence

from ragmake.models import Chunk, Hit

_COLS = ("corpus_id", "document_id", "chunk_id", "key", "text", "char_start", "char_end", "content_hash", "metadata")


def _ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


class _Postgres:
    """One lazily opened autocommit connection; subclasses create their tables in _setup."""

    def __init__(self, dsn: str | None, schema: str, table: str, connect: Callable[[], Any] | None):
        self.dsn, self.connect = dsn, connect
        self.table = f"{_ident(schema)}.{_ident(table)}"
        self._conn: Any = None

    @property
    def conn(self) -> Any:
        if self._conn is None:
            if self.connect:
                conn = self.connect()
            else:
                import psycopg

                conn = psycopg.connect(self.dsn, autocommit=True)
            self._setup(conn)
            self._conn = conn
        return self._conn

    def _setup(self, conn: Any) -> None:
        raise NotImplementedError

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class PostgresArtifactStore(_Postgres):
    def __init__(self, dsn: str | None = None, *, schema: str = "public", table: str = "ragmake_artifacts",
                 connect: Callable[[], Any] | None = None):
        super().__init__(dsn, schema, table, connect)
        self.schema = schema

    def _setup(self, conn: Any) -> None:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (path TEXT PRIMARY KEY, data JSONB NOT NULL)")

    def read_json(self, path: str) -> Any | None:
        row = self.conn.execute(f"SELECT data FROM {self.table} WHERE path = %s", (path,)).fetchone()
        return None if row is None else row[0]

    def write_json(self, path: str, data: Any) -> None:
        self.conn.execute(
            f"INSERT INTO {self.table} (path, data) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (path) DO UPDATE SET data = excluded.data",
            (path, json.dumps(data)),
        )

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        sql = f"SELECT path, data FROM {self.table} WHERE left(path, length(%s)) = %s ORDER BY path"
        return [(path, data) for path, data in self.conn.execute(sql, (prefix, prefix)).fetchall()]

    def delete(self, path: str) -> None:
        self.conn.execute(f"DELETE FROM {self.table} WHERE path = %s", (path,))


class PgVectorStore(_Postgres):
    def __init__(self, dimensions: int, dsn: str | None = None, *, schema: str = "public", table: str = "ragmake_chunks",
                 connect: Callable[[], Any] | None = None, index: str | None = "hnsw"):
        super().__init__(dsn, schema, table, connect)
        self.schema, self.dimensions, self.index = schema, int(dimensions), _ident(index) if index else None

    def _setup(self, conn: Any) -> None:
        from pgvector.psycopg import register_vector

        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {self.table} (
                corpus_id TEXT NOT NULL, document_id TEXT NOT NULL, chunk_id INTEGER NOT NULL,
                key TEXT PRIMARY KEY, text TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
                content_hash TEXT NOT NULL, metadata JSONB NOT NULL, embedding vector({self.dimensions}) NOT NULL)"""
        )
        name = self.table.replace(".", "_")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_corpus ON {self.table} (corpus_id)")
        if self.index:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name}_embedding ON {self.table} USING {self.index} (embedding vector_cosine_ops)")

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        from pgvector import Vector

        update = ", ".join(f"{c} = excluded.{c}" for c in _COLS[:3] + _COLS[4:]) + ", embedding = excluded.embedding"
        sql = (f"INSERT INTO {self.table} ({', '.join(_COLS)}, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s) "
               f"ON CONFLICT (key) DO UPDATE SET {update}")
        rows = [
            (c.corpus_id, c.document_id, c.chunk_id, c.key, c.text, c.char_start, c.char_end, c.content_hash,
             json.dumps(c.metadata), Vector(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(sql, rows)

    def delete_chunks(self, keys: Sequence[str]) -> None:
        self.conn.execute(f"DELETE FROM {self.table} WHERE key = ANY(%s)", (list(keys),))

    def search(self, corpus_id: str, vector: Sequence[float], top_k: int) -> list[Hit]:
        from pgvector import Vector

        q = Vector(vector)
        sql = (f"SELECT {', '.join(_COLS)}, 1 - (embedding <=> %s) FROM {self.table} WHERE corpus_id = %s "
               "ORDER BY embedding <=> %s LIMIT %s")
        rows = self.conn.execute(sql, (q, corpus_id, q, top_k)).fetchall()
        return [Hit(Chunk(*row[:9]), float(row[9])) for row in rows]
