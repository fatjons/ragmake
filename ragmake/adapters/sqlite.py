from __future__ import annotations

import json
import sqlite3
from array import array
from heapq import nlargest
from operator import mul
from typing import Any, Iterable, Sequence

from ragmake.models import Chunk, Hit

_COLS = "corpus_id, document_id, chunk_id, key, text, char_start, char_end, content_hash, metadata"


def _connect(path: str, schema: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    with conn:
        conn.executescript(schema)
    return conn


def _unit(vec: Sequence[float]) -> array:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return array("f", (x / norm for x in vec))


def _chunk(row: Sequence[Any]) -> Chunk:
    return Chunk(*row[:8], json.loads(row[8]))


class SQLiteArtifactStore:
    def __init__(self, db_path: str):
        self.conn = _connect(db_path, "CREATE TABLE IF NOT EXISTS rag_artifacts (path TEXT PRIMARY KEY, data TEXT NOT NULL)")

    def read_json(self, path: str) -> Any | None:
        row = self.conn.execute("SELECT data FROM rag_artifacts WHERE path = ?", (path,)).fetchone()
        return None if row is None else json.loads(row[0])

    def write_json(self, path: str, data: Any) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO rag_artifacts VALUES (?, ?)", (path, json.dumps(data)))

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        rows = self.conn.execute(
            "SELECT path, data FROM rag_artifacts WHERE substr(path, 1, ?) = ? ORDER BY path", (len(prefix), prefix)
        ).fetchall()
        return [(path, json.loads(data)) for path, data in rows]

    def delete(self, path: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM rag_artifacts WHERE path = ?", (path,))

    def close(self) -> None:
        self.conn.close()


class SQLiteVectorStore:
    """Brute-force cosine search over L2-normalized float32 blobs."""

    def __init__(self, db_path: str):
        self.conn = _connect(
            db_path,
            f"""CREATE TABLE IF NOT EXISTS rag_chunks (
                corpus_id TEXT NOT NULL, document_id TEXT NOT NULL, chunk_id INTEGER NOT NULL,
                key TEXT PRIMARY KEY, text TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
                content_hash TEXT NOT NULL, metadata TEXT NOT NULL, vector BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS rag_chunks_corpus ON rag_chunks (corpus_id);""",
        )

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        rows = [
            (c.corpus_id, c.document_id, c.chunk_id, c.key, c.text, c.char_start, c.char_end, c.content_hash,
             json.dumps(c.metadata), _unit(v).tobytes())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.conn:
            self.conn.executemany(f"INSERT OR REPLACE INTO rag_chunks ({_COLS}, vector) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    def delete_chunks(self, keys: Sequence[str]) -> None:
        with self.conn:
            self.conn.executemany("DELETE FROM rag_chunks WHERE key = ?", [(k,) for k in keys])

    def search(self, corpus_id: str, vector: Sequence[float], top_k: int) -> list[Hit]:
        q = _unit(vector)
        scored = []
        for *row, blob in self.conn.execute(f"SELECT {_COLS}, vector FROM rag_chunks WHERE corpus_id = ?", (corpus_id,)):
            v = array("f")
            v.frombytes(blob)
            scored.append((sum(map(mul, q, v)), row))
        return [Hit(_chunk(row), score) for score, row in nlargest(top_k, scored, key=lambda s: s[0])]

    def close(self) -> None:
        self.conn.close()
