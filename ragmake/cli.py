from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .adapters.sqlite import SQLiteArtifactStore, SQLiteVectorStore
from .chunking import Chunker
from .compiler import Compiler
from .defaults import HashingEmbedder, HeuristicNectar
from .models import Document
from .runtime import Runtime


def _documents(paths: list[str], corpus_id: str):
    for arg in paths:
        root = Path(arg)
        files = sorted(f for f in root.rglob("*") if f.suffix in {".txt", ".md"}) if root.is_dir() else [root]
        for file in files:
            try:
                document_id = file.resolve().relative_to(Path.cwd()).as_posix()
            except ValueError:
                document_id = str(file)
            yield Document(corpus_id, document_id, file.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ragmake")
    ap.add_argument("--state", default=".ragmake")
    ap.add_argument("--corpus", default="default")
    ap.add_argument("--openai", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--prune", action="store_true")
    query = sub.add_parser("query")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=5)
    sub.add_parser("nectar")
    args = ap.parse_args(argv)

    if args.openai:
        from openai import OpenAI

        from .adapters.openai import OpenAIEmbedder, OpenAINectar

        client = OpenAI()
        embedder = OpenAIEmbedder(client, model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        nectar = OpenAINectar(client, model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    else:
        embedder, nectar = HashingEmbedder(), HeuristicNectar()
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    db = state / "ragmake.db"
    compiler = Compiler(SQLiteArtifactStore(db), SQLiteVectorStore(db), embedder, nectar, Chunker())

    if args.command == "ingest":
        out = asdict(compiler.ingest(_documents(args.paths, args.corpus), prune=args.prune))
    elif args.command == "query":
        out = Runtime(compiler).context(args.text, args.corpus, top_k=args.top_k)
    else:
        found = compiler.load_nectar(args.corpus)
        out = asdict(found) if found else None
    print(json.dumps(out, indent=2))
