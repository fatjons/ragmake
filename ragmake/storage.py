from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

from .models import Chunk, Hit


def normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else [0.0] * len(vec)


class FileArtifactStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def _path(self, rel: str) -> Path:
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"unsafe artifact path: {rel!r}")
        return self.root / rel

    def read_json(self, path: str) -> Any | None:
        try:
            return json.loads(self._path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def write_json(self, path: str, data: Any) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, target)

    def iter_json(self, prefix: str) -> Iterator[tuple[str, Any]]:
        base = self._path(prefix + "x").parent  # deepest directory the prefix can match in
        for file in sorted(base.rglob("*.json")) if base.is_dir() else ():
            rel = file.relative_to(self.root).as_posix()
            if rel.startswith(prefix):
                yield rel, json.loads(file.read_text(encoding="utf-8"))

    def delete(self, path: str) -> None:
        self._path(path).unlink(missing_ok=True)


class MemoryArtifactStore:
    def __init__(self):
        self.data: dict[str, str] = {}  # serialized, so callers never share mutable state

    def read_json(self, path: str) -> Any | None:
        return json.loads(self.data[path]) if path in self.data else None

    def write_json(self, path: str, data: Any) -> None:
        self.data[path] = json.dumps(data)

    def iter_json(self, prefix: str) -> Iterator[tuple[str, Any]]:
        for path in sorted(p for p in self.data if p.startswith(prefix)):
            yield path, json.loads(self.data[path])

    def delete(self, path: str) -> None:
        self.data.pop(path, None)


class MemoryVectorStore:
    def __init__(self):
        self.rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            self.rows[chunk.key] = (chunk, normalize(vector))

    def delete_chunks(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.rows.pop(key, None)

    def search(self, corpus_id: str, vector: Sequence[float], top_k: int) -> list[Hit]:
        q = normalize(vector)
        hits = [Hit(c, sum(a * b for a, b in zip(q, v)))
                for c, v in self.rows.values() if c.corpus_id == corpus_id]
        return sorted(hits, key=lambda h: -h.score)[:top_k]
