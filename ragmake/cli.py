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


def _client(args):
    from openai import OpenAI

    keyless = args.base_url and not os.environ.get("OPENAI_API_KEY")
    return OpenAI(base_url=args.base_url, api_key="none" if keyless else None)


def _embedder(args):
    kind, _, model = args.embedder.partition(":")
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "openai":
        from .adapters.openai import OpenAIEmbedder

        return OpenAIEmbedder(_client(args), model, dimensions=args.dimensions,
                              query_prefix=args.query_prefix or "", document_prefix=args.document_prefix or "")
    if kind == "st":
        from .adapters.sentence_transformers import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model, dimensions=args.dimensions, query_prompt=args.query_prefix,
                                           document_prompt=args.document_prefix)
    raise SystemExit(f"unknown embedder {args.embedder!r}: use hashing, openai:MODEL or st:MODEL_ID")


def _nectar(args):
    kind, _, model = args.nectar.partition(":")
    if kind == "heuristic":
        return HeuristicNectar()
    if kind == "openai":
        from .adapters.openai import OpenAINectar

        return OpenAINectar(_client(args), model)
    raise SystemExit(f"unknown nectar {args.nectar!r}: use heuristic or openai:MODEL")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ragmake")
    ap.add_argument("--state", default=".ragmake")
    ap.add_argument("--corpus", default="default")
    ap.add_argument("--embedder", default="hashing", metavar="SPEC", help="hashing | openai:MODEL | st:MODEL_ID")
    ap.add_argument("--nectar", default="heuristic", metavar="SPEC", help="heuristic | openai:MODEL")
    ap.add_argument("--base-url", help="OpenAI-compatible server, e.g. http://localhost:11434/v1")
    ap.add_argument("--dimensions", type=int)
    ap.add_argument("--query-prefix", help="query prefix (openai:) or prompt (st:)")
    ap.add_argument("--document-prefix", help="document prefix (openai:) or prompt (st:)")
    sub = ap.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--prune", action="store_true")
    query = sub.add_parser("query")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=5)
    sub.add_parser("nectar")
    args = ap.parse_args(argv)

    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    db = state / "ragmake.db"
    compiler = Compiler(SQLiteArtifactStore(db), SQLiteVectorStore(db), _embedder(args), _nectar(args), Chunker())

    if args.command == "ingest":
        out = asdict(compiler.ingest(_documents(args.paths, args.corpus), prune=args.prune))
    elif args.command == "query":
        out = Runtime(compiler).context(args.text, args.corpus, top_k=args.top_k)
    else:
        found = compiler.load_nectar(args.corpus)
        out = asdict(found) if found else None
    print(json.dumps(out, indent=2))
