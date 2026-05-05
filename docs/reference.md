# ragmake Reference

This document is the code-backed reference for the package. It covers the
architecture, the nectar model, the storage layout, adapter surface, local
persistent usage, benchmark behavior, and operational limits.

---

## Core concept: nectar

Standard RAG is reactive. The agent sees whatever chunks retrieval returns for a
given query and has to infer the shape of the corpus from those fragments alone.
If retrieval misses something, the agent never knew it existed.

ragmake is different. Before any query runs, the compiler reads all ingested
documents and distills them into a **nectar** — a corpus-level synthesis that
captures the scope, dominant concepts, recurring terminology, and document set
of the entire corpus. Nectar is:

- **compiled at ingest time**, not query time
- **query-independent** — it exists before any question is asked
- **cached and reused** — rebuilt only when the corpus content signature changes
- **always current** — the signature is computed from a hash of all document hashes, so any change to any document triggers a refresh

At query time the agent receives two distinct layers:

| Layer | What it is | Built when |
|-------|-----------|------------|
| **Nectar** (`synopsis` in code) | What the corpus contains — scope, concepts, shape | Compile time. Persists until corpus changes. |
| **Evidence** | What's relevant to this specific query | Query time. Retrieved from the vector store. |

The agent is not reasoning from fragments alone. It has the map before it looks
at the territory.

---

## Architecture

The runtime model has two planes.

### Compile plane

1. Caller provides `SourceDocument` objects.
2. `StatefulRAGCompiler` hashes each full document and skips unchanged inputs unless `force=True`.
3. Changed documents are split by `WordChunker` into `ChunkRecord` objects with provenance.
4. Each chunk embedding is looked up by `content_hash` before calling the configured `Embedder`. Unchanged chunks within a changed document reuse cached vectors.
5. New or changed chunks are upserted into the configured `VectorStore`.
6. A `LedgerEntry` is written for each processed document.
7. The compiler recomputes a corpus content signature from all ledger entries.
8. The corpus synopsis (nectar) is rebuilt only when that signature changes.

### Serve plane

1. `StatefulRAGRuntime.build_context()` embeds the query once.
2. The configured `VectorStore` returns the top matching chunks (evidence).
3. The runtime loads the compiled corpus synopsis (nectar).
4. The runtime optionally loads a `SessionState` through `SessionStateManager`.
5. `render_prompt_payload()` produces a plain dict containing query, corpus id, nectar, retrieved evidence, and optional session summary.

The agent receives a payload where `synopsis` is the nectar and `sources` is
the evidence. Both are always present when the corpus is non-empty.

---

## Artifact model

The kernel persists four state families under the compiler namespace. The
default namespace is `.stateful_rag`.

| Family | Path pattern | Purpose |
|--------|-------------|---------|
| Ledgers | `.stateful_rag/ledger/<corpus_id>/<document_id>.json` | Per-document ingest state and chunk IDs |
| Embedding cache | `.stateful_rag/cache/embeddings/<content_hash>.json` | Vectors keyed by chunk content hash |
| Synopses (nectar) | `.stateful_rag/synopses/<corpus_id>.json` | Compiled corpus-level memory |
| Sessions | `.stateful_rag/sessions/<session_id>.json` | Per-session conversational state |

Important runtime types:

- `SourceDocument` — caller-owned input document
- `ChunkRecord` — chunk text plus `document_id`, `chunk_id`, character spans, word offsets, and `content_hash`
- `LedgerEntry` — per-document ingest state
- `CorpusSynopsis` — the compiled nectar: corpus scope, content signature, document and chunk counts, synopsis text
- `SessionState` — learned concepts, discussed entities, intents, and decisions for one session
- `RuntimeContext` — retrieved chunks plus nectar and optional session
- `IngestionReport` — processed, skipped, and rebuilt counts from one ingest

Note on chunk offsets: `WordChunker` splits on whitespace-like words.
`ChunkRecord.token_start` and `token_end` are word offsets, not model-token offsets.

---

## Public surfaces

### ArtifactStore implementations

- `FileArtifactStore` — JSON files on disk; easiest to inspect manually
- `InMemoryArtifactStore` — tests and ephemeral runs
- `SQLiteArtifactStore` — durable local state in one SQLite file
- `AzureBlobArtifactStore` — blob-backed JSON artifacts
- `PostgresArtifactStore` — JSONB-backed artifacts

### VectorStore implementations

- `InMemoryVectorStore` — tests and ephemeral runs
- `SQLiteVectorStore` — durable local vectors with Python-side cosine retrieval
- `AzureAISearchVectorStore` — Azure AI Search storage and vector retrieval
- `PgVectorStore` — Postgres with `pgvector`

### Embedder implementations

- `HashingEmbedder` — deterministic fallback for tests, demos, and benchmarks
- `OpenAICompatibleEmbedder` — OpenAI-style embeddings client

### SynopsisCompiler implementations (nectar compilers)

- `HeuristicSynopsisCompiler` — deterministic local fallback; extracts top terms and chunk excerpts; no LLM required
- `OpenAISynopsisCompiler` — uses chat completions to write grounded nectar from representative corpus chunks; produces semantic, readable summaries

The `SynopsisCompiler` protocol is the nectar interface. Implement it to plug in
any summarization strategy — local models, custom prompts, rule-based extractors.

---

## Local persistent walkthrough

For a durable local stack, pair SQLite artifacts with SQLite vectors. Use
`OpenAISynopsisCompiler` for production-quality nectar or `HeuristicSynopsisCompiler`
for offline and test setups.

```python
from dataclasses import asdict
from pathlib import Path

from stateful_rag import (
    HashingEmbedder,
    HeuristicSynopsisCompiler,
    SessionStateManager,
    SourceDocument,
    SQLiteArtifactStore,
    SQLiteVectorStore,
    StatefulRAGCompiler,
    StatefulRAGRuntime,
    WordChunker,
)

db_path = Path("demo_env/ragmake.sqlite")
artifact_store = SQLiteArtifactStore(db_path)
vector_store = SQLiteVectorStore(db_path)
embedder = HashingEmbedder(dimensions=64)
synopsis_compiler = HeuristicSynopsisCompiler()
session_manager = SessionStateManager(artifact_store)

compiler = StatefulRAGCompiler(
    artifact_store=artifact_store,
    vector_store=vector_store,
    embedder=embedder,
    synopsis_compiler=synopsis_compiler,
    chunker=WordChunker(max_words=80, overlap_words=10),
)
runtime = StatefulRAGRuntime(
    compiler=compiler,
    vector_store=vector_store,
    embedder=embedder,
    session_manager=session_manager,
)

documents = [
    SourceDocument(
        corpus_id="support-kb",
        document_id="refund-policy.txt",
        title="Refund Policy",
        text="Enterprise refunds are allowed within 30 days when the onboarding pack is unused.",
        metadata={"team": "finance"},
    ),
    SourceDocument(
        corpus_id="support-kb",
        document_id="api-access.txt",
        title="API Access",
        text="Workspace admins can rotate API keys from the admin console.",
        metadata={"team": "platform"},
    ),
]

# First ingest: processes both documents, compiles nectar
first_report = compiler.ingest_documents(documents)

# Record session state for conversational memory
session_manager.record_learning(
    "customer-42",
    summary="User is handling enterprise billing questions.",
    concepts=["enterprise refunds", "invoice workflow"],
    entities=["Acme Corp"],
)
session_manager.record_intent("customer-42", intent="collect_refund_requirements")

# Query: agent receives nectar + evidence + session
context = runtime.build_context(
    query="What information is required for an enterprise refund request?",
    corpus_id="support-kb",
    top_k=3,
    session_id="customer-42",
)
payload = runtime.render_prompt_payload(context)

# payload["synopsis"] — the nectar (corpus memory)
# payload["sources"]  — the evidence (retrieved chunks)
# payload["session"]  — conversational memory

# Second ingest: only the changed document is reprocessed
updated_documents = [
    documents[0],
    SourceDocument(
        corpus_id="support-kb",
        document_id="api-access.txt",
        title="API Access",
        text="Workspace admins can rotate API keys from the admin console every 90 days.",
        metadata={"team": "platform"},
    ),
]
second_report = compiler.ingest_documents(updated_documents)

print(asdict(first_report))
# {'processed_documents': 2, 'skipped_documents': 0, 'chunks_upserted': 2, 'synopses_rebuilt': ['support-kb']}

print(asdict(second_report))
# {'processed_documents': 1, 'skipped_documents': 1, 'chunks_upserted': 1, 'synopses_rebuilt': ['support-kb']}
# Only the changed document was reprocessed. Nectar was refreshed because the corpus signature changed.
```

What to expect:

- first ingest processes both documents and compiles nectar for `support-kb`
- session section appears only when both `session_manager` and `session_id` are provided
- second ingest only reprocesses the changed document
- unchanged document keeps its prior ledger and vectors
- nectar is rebuilt because the corpus content signature changed

---

## CLI behavior

`ragmake-demo` ingests plain UTF-8 text files and emits a JSON prompt payload:

1. reads the provided files as UTF-8 text
2. stores artifacts and vectors in `--state-dir/ragmake.sqlite`
3. ingests the files into one corpus
4. emits JSON containing the ingest report and prompt payload (including nectar)

Rerunning the command with the same `--state-dir` and unchanged files produces
warm-ingest skips. The nectar in the payload will be identical to the prior run.

---

## Benchmark behavior

`ragmake-benchmark` measures changed-document savings against a stateless rebuild.
Phases:

- `stateful_cold` — first ingest, builds full state and nectar
- `stateful_warm` — reingest of identical corpus; all documents skipped
- `stateful_changed` — reingest with a fraction of documents changed
- `stateless_base` — stateless baseline, full ingest
- `stateless_changed` — stateless reingest, no reuse

Summary output focuses on:

- document reuse during the changed run
- embedding calls avoided versus the stateless changed run
- ingest speedup versus the stateless changed run
- warm-cache skip rate as a sanity check

The benchmark does not measure nectar quality — it measures computational reuse.
Nectar quality depends on the `SynopsisCompiler` implementation.

---

## Adapter notes

**OpenAI-style adapters**

- `OpenAICompatibleEmbedder` accepts an existing client or builds one from OpenAI or Azure OpenAI credentials.
- `OpenAISynopsisCompiler` uses chat completions to compile nectar from representative corpus chunks. The prompt instructs the model to stay grounded in the provided material and produce a compact, traceable summary.

**SQLite adapters**

- `SQLiteArtifactStore` and `SQLiteVectorStore` share one SQLite file.
- `SQLiteVectorStore` stores vectors as JSON and ranks with Python-side cosine similarity.

**Azure adapters**

- `AzureBlobArtifactStore` stores JSON artifacts in a blob container.
- `AzureAISearchVectorStore` can create the search index with `ensure_index(dimensions)` when an index client is available.

**Postgres adapters**

- `PostgresArtifactStore` stores artifacts in JSONB.
- `PgVectorStore` requires `pgvector` and a fixed embedding dimension at initialization time.

---

## Operational limits

- `WordChunker` is intentionally simple and does not target a tokenizer budget.
- The deterministic local components are for tests, demos, and benchmarks.
- The compiler does not provide a first-class delete API for documents removed from a corpus.
- The package expects the caller to provide text extraction and document loading.
- The runtime assembles prompt context and nectar; it does not generate final answers or manage an agent loop.
- Schema tolerance is improved by `from_dict()` loaders, but data migrations across external systems are the caller's responsibility.

---

## Verification

```bash
python -m unittest tests.test_incremental_ingest tests.test_benchmark
ragmake-demo --query "How does primer coupling work?" sample1.txt sample2.txt
ragmake-benchmark --documents 40 --change-fraction 0.15 --backend sqlite
```
