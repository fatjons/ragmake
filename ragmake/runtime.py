from __future__ import annotations

from .compiler import Compiler
from .models import Hit
from .session import SessionStore


class Runtime:
    def __init__(self, compiler: Compiler, sessions: SessionStore | None = None):
        self.compiler, self.sessions = compiler, sessions

    def search(self, query: str, corpus_id: str, top_k: int = 5) -> list[Hit]:
        vector = self.compiler.embedder.embed_texts([query])[0]
        return self.compiler.vectors.search(corpus_id, vector, top_k)

    def context(self, query: str, corpus_id: str, *, top_k: int = 5, session_id: str | None = None,
                catalog_limit: int = 50) -> dict:
        nectar = self.compiler.load_nectar(corpus_id)
        payload = {
            "query": query,
            "corpus_id": corpus_id,
            "signature": nectar.signature if nectar else None,
            "nectar": nectar.text if nectar else None,
            "catalog": nectar.catalog[:catalog_limit] if nectar else [],
            "sources": [{"document_id": h.chunk.document_id, "title": h.chunk.metadata.get("title"),
                         "chunk_id": h.chunk.chunk_id, "score": h.score, "char_start": h.chunk.char_start,
                         "char_end": h.chunk.char_end, "text": h.chunk.text}
                        for h in self.search(query, corpus_id, top_k)],
        }
        if session_id and self.sessions and (summary := self.sessions.summary(session_id)):
            payload["session"] = summary
        return payload
