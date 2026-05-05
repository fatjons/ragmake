from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adapters.sqlite_backend import SQLiteArtifactStore, SQLiteVectorStore
from .chunking import WordChunker
from .compiler import StatefulRAGCompiler
from .defaults import HashingEmbedder, HeuristicSynopsisCompiler
from .models import SourceDocument
from .runtime import StatefulRAGRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Local demo for the stateful RAG kernel")
    parser.add_argument("files", nargs="+", help="Text files to ingest for this run")
    parser.add_argument("--corpus", default="demo", help="Corpus identifier")
    parser.add_argument("--query", required=True, help="Question to run after ingestion")
    parser.add_argument(
        "--state-dir",
        default=".stateful_rag_demo",
        help="Directory containing the persistent SQLite state",
    )
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "stateful_rag.sqlite"
    artifact_store = SQLiteArtifactStore(db_path)
    vector_store = SQLiteVectorStore(db_path)
    embedder = HashingEmbedder()
    compiler = StatefulRAGCompiler(
        artifact_store=artifact_store,
        vector_store=vector_store,
        embedder=embedder,
        synopsis_compiler=HeuristicSynopsisCompiler(),
        chunker=WordChunker(max_words=160, overlap_words=20),
    )

    documents = [
        SourceDocument(
            corpus_id=args.corpus,
            document_id=Path(file_name).as_posix(),
            text=Path(file_name).read_text(encoding="utf-8"),
            title=Path(file_name).name,
        )
        for file_name in args.files
    ]

    report = compiler.ingest_documents(documents)
    runtime = StatefulRAGRuntime(compiler=compiler, vector_store=vector_store, embedder=embedder)
    payload = runtime.render_prompt_payload(runtime.build_context(args.query, corpus_id=args.corpus))

    print(json.dumps({"report": asdict(report), "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
