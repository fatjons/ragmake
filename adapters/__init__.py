"""Optional production adapters for the generic stateful RAG kernel."""

from .azure_ai_search import AzureAISearchVectorStore
from .azure_blob import AzureBlobArtifactStore
from .openai_embedder import OpenAICompatibleEmbedder, OpenAISynopsisCompiler
from .pgvector_backend import PgVectorStore, PostgresArtifactStore
from .sqlite_backend import SQLiteArtifactStore, SQLiteVectorStore

__all__ = [
    "AzureAISearchVectorStore",
    "AzureBlobArtifactStore",
    "OpenAICompatibleEmbedder",
    "OpenAISynopsisCompiler",
    "PgVectorStore",
    "PostgresArtifactStore",
    "SQLiteArtifactStore",
    "SQLiteVectorStore",
]
