from __future__ import annotations

import hashlib
import re

from .models import ChunkRecord, SourceDocument


_WORD_RE = re.compile(r"\S+")


class WordChunker:
    def __init__(self, max_words: int = 400, overlap_words: int = 40):
        if max_words <= 0:
            raise ValueError("max_words must be > 0")
        if overlap_words < 0:
            raise ValueError("overlap_words must be >= 0")
        if overlap_words >= max_words:
            raise ValueError("overlap_words must be smaller than max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, document: SourceDocument) -> list[ChunkRecord]:
        word_matches = list(_WORD_RE.finditer(document.text))
        if not word_matches:
            return []

        chunks: list[ChunkRecord] = []
        step = self.max_words - self.overlap_words

        for chunk_id, word_start in enumerate(range(0, len(word_matches), step)):
            window = word_matches[word_start : word_start + self.max_words]
            if not window:
                break

            char_start = window[0].start()
            char_end = window[-1].end()
            text = document.text[char_start:char_end]
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_key = hashlib.sha1(
                f"{document.corpus_id}:{document.document_id}:{chunk_id}:{content_hash}".encode("utf-8")
            ).hexdigest()
            metadata = dict(document.metadata)
            if document.title:
                metadata.setdefault("title", document.title)

            chunks.append(
                ChunkRecord(
                    corpus_id=document.corpus_id,
                    document_id=document.document_id,
                    chunk_id=chunk_id,
                    chunk_key=chunk_key,
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=word_start,
                    token_end=word_start + len(window),
                    content_hash=content_hash,
                    metadata=metadata,
                )
            )

            if word_start + self.max_words >= len(word_matches):
                break

        return chunks
