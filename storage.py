from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import ChunkRecord, RetrievedChunk


class FileArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read_json(self, path: str) -> Any | None:
        file_path = self.root / path
        if not file_path.exists():
            return None
        return json.loads(file_path.read_text(encoding="utf-8"))

    def write_json(self, path: str, data: Any) -> None:
        file_path = self.root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        base = self.root / prefix
        if not base.exists():
            return []
        items: list[tuple[str, Any]] = []
        for file_path in sorted(base.rglob("*.json")):
            relative = file_path.relative_to(self.root).as_posix()
            items.append((relative, json.loads(file_path.read_text(encoding="utf-8"))))
        return items

    def delete(self, path: str) -> None:
        file_path = self.root / path
        if file_path.exists():
            file_path.unlink()


class InMemoryArtifactStore:
    def __init__(self):
        self._items: dict[str, Any] = {}

    def read_json(self, path: str) -> Any | None:
        return self._items.get(path)

    def write_json(self, path: str, data: Any) -> None:
        self._items[path] = json.loads(json.dumps(data))

    def iter_json(self, prefix: str) -> Iterable[tuple[str, Any]]:
        return [
            (path, value)
            for path, value in sorted(self._items.items())
            if path.startswith(prefix)
        ]

    def delete(self, path: str) -> None:
        self._items.pop(path, None)


class InMemoryVectorStore:
    def __init__(self):
        self._chunks: dict[str, ChunkRecord] = {}
        self._vectors: dict[str, list[float]] = {}
        self._corpora: dict[str, set[str]] = defaultdict(set)

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.chunk_key] = ChunkRecord(**asdict(chunk))
            self._vectors[chunk.chunk_key] = [float(value) for value in vector]
            self._corpora[chunk.corpus_id].add(chunk.chunk_key)

    def delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            chunk = self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
            if chunk:
                keys = self._corpora.get(chunk.corpus_id)
                if keys:
                    keys.discard(chunk_id)

    def list_chunks(self, corpus_id: str, limit: int | None = None) -> list[ChunkRecord]:
        keys = sorted(
            self._corpora.get(corpus_id, set()),
            key=lambda key: (
                self._chunks[key].document_id,
                self._chunks[key].chunk_id,
            ),
        )
        if limit is not None:
            keys = keys[:limit]
        return [ChunkRecord(**asdict(self._chunks[key])) for key in keys]

    def search(
        self,
        corpus_id: str,
        query_vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        scores: list[RetrievedChunk] = []
        for chunk_id in self._corpora.get(corpus_id, set()):
            score = _cosine_similarity(query_vector, self._vectors[chunk_id])
            scores.append(RetrievedChunk(chunk=ChunkRecord(**asdict(self._chunks[chunk_id])), score=score))
        scores.sort(key=lambda item: item.score, reverse=True)
        return scores[:top_k]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
