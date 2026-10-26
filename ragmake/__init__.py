from typing import TYPE_CHECKING

from .chunking import WordChunker
from .compiler import StatefulRAGCompiler
from .defaults import HashingEmbedder, HeuristicSynopsisCompiler
from .adapters import PgVectorStore, PostgresArtifactStore, SQLiteArtifactStore, SQLiteVectorStore
from .models import (
    ChunkRecord,
    CorpusSynopsis,
    IngestionReport,
    RetrievedChunk,
    RuntimeContext,
    SourceDocument,
)
from .runtime import StatefulRAGRuntime
from .session import SessionStateManager
from .storage import FileArtifactStore, InMemoryArtifactStore, InMemoryVectorStore

if TYPE_CHECKING:
    from .benchmark import BenchmarkReport

__all__ = [
    "BenchmarkReport",
    "ChunkRecord",
    "CorpusSynopsis",
    "FileArtifactStore",
    "HashingEmbedder",
    "HeuristicSynopsisCompiler",
    "InMemoryArtifactStore",
    "InMemoryVectorStore",
    "IngestionReport",
    "PgVectorStore",
    "PostgresArtifactStore",
    "RetrievedChunk",
    "RuntimeContext",
    "SessionStateManager",
    "SQLiteArtifactStore",
    "SQLiteVectorStore",
    "SourceDocument",
    "StatefulRAGCompiler",
    "StatefulRAGRuntime",
    "WordChunker",
    "run_benchmark",
]


def __getattr__(name: str):
    if name in {"BenchmarkReport", "run_benchmark"}:
        from .benchmark import BenchmarkReport, run_benchmark

        exports = {
            "BenchmarkReport": BenchmarkReport,
            "run_benchmark": run_benchmark,
        }
        return exports[name]
    raise AttributeError(name)
