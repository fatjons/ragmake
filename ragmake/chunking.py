from __future__ import annotations

import hashlib
import re
import zlib

from .models import Chunk, Document

_PARAGRAPH = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\S+")


class Chunker:
    """Content-defined chunking: boundaries depend on nearby text only, so an edit
    changes nearby chunks and leaves the rest (and their embeddings) alone."""

    def __init__(self, max_words: int = 200, overlap_words: int = 20, min_words: int = 50):
        if not (max_words > 0 and 0 <= overlap_words < max_words and 0 <= min_words <= max_words):
            raise ValueError("need max_words > 0, 0 <= overlap_words < max_words, 0 <= min_words <= max_words")
        self.max_words, self.overlap_words, self.min_words = max_words, overlap_words, min_words
        self.name = f"cdc-{max_words}-{overlap_words}-{min_words}"

    def _pieces(self, text: str):
        """Yield (start, end, words, windowed) for each paragraph or oversized-paragraph window."""
        pos = 0
        for m in (*_PARAGRAPH.finditer(text), None):
            end = m.start() if m else len(text)
            words = [(pos + w.start(), pos + w.end()) for w in _WORD.finditer(text[pos:end])]
            pos = m.end() if m else end
            if len(words) <= self.max_words:
                if words:
                    yield words[0][0], words[-1][1], len(words), False
                continue
            for i in range(0, len(words), self.max_words - self.overlap_words):
                win = words[i : i + self.max_words]
                yield win[0][0], win[-1][1], len(win), True
                if i + self.max_words >= len(words):
                    break

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        spans: list[tuple[int, int]] = []
        cur: list[int] = []  # [start, end, words]

        def flush() -> None:
            if cur:
                spans.append((cur[0], cur[1]))
                cur.clear()

        for start, end, n, windowed in self._pieces(text):
            if windowed:
                flush()
                spans.append((start, end))
                continue
            if cur and cur[2] + n > self.max_words:
                flush()
            cur[:] = [cur[0], end, cur[2] + n] if cur else [start, end, n]
            if cur[2] >= self.min_words and zlib.crc32(text[start:end].encode()) % 4 == 0:
                flush()
        flush()

        metadata = {**document.metadata, **({"title": document.title} if document.title else {})}
        chunks = []
        for i, (start, end) in enumerate(spans):
            body = text[start:end]
            content_hash = hashlib.sha256(body.encode()).hexdigest()
            ident = f"{document.corpus_id}\0{document.document_id}\0{i}\0{content_hash}"
            chunks.append(Chunk(document.corpus_id, document.document_id, i,
                                hashlib.sha1(ident.encode()).hexdigest(),
                                body, start, end, content_hash, dict(metadata)))
        return chunks
