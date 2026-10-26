from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

from .models import ChunkRecord


_WORD_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


class HashingEmbedder:
    """Deterministic fallback embedder for local demos and tests.

    Not suitable for production semantic search.
    """

    def __init__(self, dimensions: int = 64):
        if dimensions <= 0:
            raise ValueError("dimensions must be > 0")
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        words = _WORD_RE.findall(text.lower())
        if not words:
            return vector

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class HeuristicSynopsisCompiler:
    """LLM-free nectar compiler for local demos and tests.

    Produces a deterministic corpus synopsis from top terms and chunk excerpts.
    Use OpenAISynopsisCompiler for semantic, LLM-written nectar in production.
    """

    def __init__(self, max_points: int = 8, excerpt_chars: int = 180):
        self.max_points = max_points
        self.excerpt_chars = excerpt_chars

    def compile_synopsis(
        self,
        corpus_id: str,
        chunks: Sequence[ChunkRecord],
        previous_synopsis: str | None = None,
    ) -> str:
        doc_ids = sorted({chunk.document_id for chunk in chunks})
        terms = Counter()
        for chunk in chunks:
            terms.update(_WORD_RE.findall(chunk.text.lower()))

        top_terms = ", ".join(term for term, _ in terms.most_common(12))
        lines = [
            f"Corpus: {corpus_id}",
            f"Documents: {len(doc_ids)}",
            f"Chunks: {len(chunks)}",
        ]
        if top_terms:
            lines.append(f"Representative terms: {top_terms}")

        for chunk in chunks[: self.max_points]:
            excerpt = " ".join(chunk.text.split())
            excerpt = excerpt[: self.excerpt_chars].rstrip()
            lines.append(f"- {chunk.document_id}#{chunk.chunk_id}: {excerpt}")

        if previous_synopsis and previous_synopsis not in lines:
            lines.append("Synopsis refreshed from prior compiled state.")

        return "\n".join(lines)
