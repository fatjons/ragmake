from __future__ import annotations

from .compiler import StatefulRAGCompiler
from .interfaces import Embedder, VectorStore
from .models import RuntimeContext
from .session import SessionStateManager


class StatefulRAGRuntime:
    """Query-time loader that assembles nectar and evidence into a prompt payload.

    At query time the agent receives two layers: the pre-compiled corpus nectar
    (what the corpus contains, built at ingest time) and retrieved evidence (what
    is relevant to this specific query). The runtime does not call an LLM for
    answering — that is the caller's responsibility.
    """

    def __init__(
        self,
        compiler: StatefulRAGCompiler,
        vector_store: VectorStore,
        embedder: Embedder,
        session_manager: SessionStateManager | None = None,
    ):
        self.compiler = compiler
        self.vector_store = vector_store
        self.embedder = embedder
        self.session_manager = session_manager

    def build_context(
        self,
        query: str,
        corpus_id: str,
        top_k: int = 5,
        include_synopsis: bool = True,
        session_id: str | None = None,
    ) -> RuntimeContext:
        query_vector = self.embedder.embed_texts([query])[0]
        chunks = self.vector_store.search(corpus_id=corpus_id, query_vector=query_vector, top_k=top_k)
        synopsis = self.compiler.load_synopsis(corpus_id) if include_synopsis else None
        session = self.session_manager.load(session_id) if self.session_manager and session_id else None
        return RuntimeContext(
            query=query,
            corpus_id=corpus_id,
            synopsis_text=synopsis.synopsis_text if synopsis else None,
            content_signature=synopsis.content_signature if synopsis else None,
            chunks=chunks,
            session=session,
        )

    def render_prompt_payload(self, context: RuntimeContext) -> dict[str, object]:
        payload: dict[str, object] = {
            "query": context.query,
            "corpus_id": context.corpus_id,
            "content_signature": context.content_signature,
            "synopsis": context.synopsis_text,
            "sources": [
                {
                    "document_id": item.chunk.document_id,
                    "chunk_id": item.chunk.chunk_id,
                    "score": item.score,
                    "char_start": item.chunk.char_start,
                    "char_end": item.chunk.char_end,
                    "text": item.chunk.text,
                }
                for item in context.chunks
            ],
        }
        if context.session:
            payload["session"] = {
                "learned_concepts": context.session.learned_concepts,
                "discussed_entities": context.session.discussed_entities,
                "recent_intents": [i.intent for i in context.session.intents[-3:]],
                "recent_decisions": [d.decision for d in context.session.decisions[-3:]],
            }
        return payload
