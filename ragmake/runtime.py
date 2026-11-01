from __future__ import annotations

from .compiler import Compiler, digest_corpus
from .models import Hit, Nectar
from .session import SessionStore


class Runtime:
    def __init__(self, compiler: Compiler, sessions: SessionStore | None = None):
        self.compiler, self.sessions = compiler, sessions

    def search(self, query: str, corpus_id: str, top_k: int = 5) -> list[Hit]:
        vector = self._query_vector(self.compiler.load_nectar(corpus_id), query, corpus_id)
        return self.compiler.vectors.search(corpus_id, vector, top_k)

    def _query_vector(self, nectar: Nectar | None, query: str, corpus_id: str) -> list[float]:
        name = self.compiler.embedder.name
        if nectar and nectar.embedder != name:
            raise ValueError(f"corpus {corpus_id!r} was built with embedder {nectar.embedder!r}, not {name!r}; "
                             "re-ingest or use the same embedder")
        return self.compiler.embedder.embed_texts([query], query=True)[0]

    def context(self, query: str, corpus_id: str, *, top_k: int = 5, session_id: str | None = None,
                catalog_limit: int = 50) -> dict:
        nectar = self.compiler.load_nectar(corpus_id)
        vector = self._query_vector(nectar, query, corpus_id)
        vectors = self.compiler.vectors
        payload = {
            "query": query,
            "corpus_id": corpus_id,
            "signature": nectar.signature if nectar else None,
            "nectar": nectar.text if nectar else None,
            "documents": nectar.document_count if nectar else 0,
            "catalog": [{"document_id": h.chunk.document_id, "title": h.chunk.metadata.get("title"),
                         "digest": h.chunk.text, "score": h.score}
                        for h in vectors.search(digest_corpus(corpus_id), vector, catalog_limit)],
            "sources": [{"document_id": h.chunk.document_id, "title": h.chunk.metadata.get("title"),
                         "chunk_id": h.chunk.chunk_id, "score": h.score, "char_start": h.chunk.char_start,
                         "char_end": h.chunk.char_end, "text": h.chunk.text}
                        for h in vectors.search(corpus_id, vector, top_k)],
        }
        if session_id and self.sessions and (summary := self.sessions.summary(session_id)):
            payload["session"] = summary
        return payload
