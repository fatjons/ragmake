# ragmake

[![CI](https://github.com/fatjons/ragmake/actions/workflows/ci.yml/badge.svg)](https://github.com/fatjons/ragmake/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

**RAG without corpus memory is blind guessing.** A retriever hands the agent the chunks that
scored highest for one query, and the agent never learns what the corpus contains.

ragmake compiles documents into persistent state, like `make`: unchanged inputs are never
reprocessed. At query time the agent gets **nectar** (what the corpus is) and **evidence**
(what is relevant to this query).

## How it works

```
document --> chunks --> embeddings     cached by (embedder name, chunk content hash)
    |
    +------> digest                    rebuilt when the document hash changes
                |
all digests +--> nectar + catalog      rebuilt when the corpus signature changes
```

- The **document hash** covers title, text, metadata and the build recipe (embedder, chunker
  and nectar compiler names). Changing the toolchain rebuilds every document.
- The **corpus signature** is a hash of the sorted `(document_id, document_hash)` pairs.
- Chunking is content-defined, so an edit changes nearby chunks and leaves the rest alone.
- Deletes are explicit: `ingest(..., prune=True)` removes documents missing from the ingest,
  `compiler.remove(corpus_id, ids)` removes named ones. Their chunks go too.
- Reingesting unchanged documents calls the embedder and nectar compiler zero times.

## Install

Python 3.11+. The core is stdlib only.

```bash
pip install -e .                 # core
pip install -e '.[openai]'       # OpenAI embeddings and nectar
pip install -e '.[postgres]'     # pgvector + psycopg
pip install -e '.[azure]'        # Azure Blob + Azure AI Search
```

## CLI

```bash
ragmake ingest docs/                  # every .txt and .md under docs/, recursively
ragmake ingest docs/ --prune          # also delete documents no longer under docs/
ragmake query "What changed in the refund policy?" --top-k 5
ragmake nectar                        # print the corpus nectar and catalog
```

Global flags go before the subcommand: `--state DIR` (default `.ragmake`, holds `ragmake.db`),
`--corpus ID` (default `default`), and `--openai`, which reads `OPENAI_API_KEY`,
`OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`) and `OPENAI_CHAT_MODEL`
(default `gpt-4o-mini`).

Queries must use the same embedder as the ingest that built the state. Never mix `--openai`
and the default embedder on one state directory: each switch rebuilds everything.

## Python: local, no dependencies

```python
from ragmake import Compiler, Document, HashingEmbedder, HeuristicNectar, MemoryArtifactStore, MemoryVectorStore, Runtime

documents = [
    Document("support-kb", "refund-policy.txt",
             "Enterprise refunds are allowed within 30 days when the onboarding pack is unused."),
    Document("support-kb", "api-access.txt", "Workspace admins can rotate API keys from the admin console."),
]
compiler = Compiler(MemoryArtifactStore(), MemoryVectorStore(), HashingEmbedder(), HeuristicNectar())
compiler.ingest(documents)
context = Runtime(compiler).context("What do I need for an enterprise refund?", "support-kb")
```

## Python: persistent, with OpenAI

```python
from openai import OpenAI
from ragmake import Compiler, Runtime
from ragmake.adapters.openai import OpenAIEmbedder, OpenAINectar
from ragmake.adapters.sqlite import SQLiteArtifactStore, SQLiteVectorStore

client = OpenAI()  # reads OPENAI_API_KEY
db = "ragmake.db"
compiler = Compiler(SQLiteArtifactStore(db), SQLiteVectorStore(db),
                    OpenAIEmbedder(client, "text-embedding-3-small"), OpenAINectar(client, "gpt-4o-mini"))
compiler.ingest(documents)  # same Document list as above
context = Runtime(compiler).context("What do I need for an enterprise refund?", "support-kb")
```

Rerunning with unchanged documents makes no embedding or chat calls for ingest. Each query
still embeds its query text.

## The payload

`Runtime.context()` returns a plain dict. Output of the local example, long strings and
numbers trimmed:

```json
{
  "query": "What do I need for an enterprise refund?",
  "corpus_id": "support-kb",
  "signature": "4a04b188...",
  "nectar": "Corpus support-kb: 2 documents.\nRecurring terms: workspace, api, console...",
  "catalog": [
    {"document_id": "api-access.txt", "title": null, "digest": "Workspace admins can rotate API keys..."},
    {"document_id": "refund-policy.txt", "title": null, "digest": "Enterprise refunds are allowed..."}
  ],
  "sources": [
    {"document_id": "refund-policy.txt", "title": null, "chunk_id": 0, "score": 0.114,
     "char_start": 0, "char_end": 81, "text": "Enterprise refunds are allowed within 30 days..."},
    {"document_id": "api-access.txt", "title": null, "chunk_id": 0, "score": 0.0,
     "char_start": 0, "char_end": 60, "text": "Workspace admins can rotate API keys..."}
  ]
}
```

`nectar` and `catalog` are query-independent. `sources` is the evidence. A `session` key is
added when `session_id` is passed.

## Sessions

Per-session memory is optional and sits beside the corpus memory:

```python
from ragmake import Compiler, HashingEmbedder, HeuristicNectar, MemoryArtifactStore, MemoryVectorStore, Runtime, SessionStore

store = MemoryArtifactStore()
compiler = Compiler(store, MemoryVectorStore(), HashingEmbedder(), HeuristicNectar())
sessions = SessionStore(store)
sessions.record("customer-42", "question", "Asked how enterprise refunds work.",
                concepts=["enterprise refunds"], entities=["Acme Corp"])
context = Runtime(compiler, sessions).context("What do I need for an enterprise refund?", "support-kb",
                                              session_id="customer-42")
# context["session"] == {"concepts": [...], "entities": [...], "recent": {"question": [...]}}
```

## Backends

| Interface | Implementations |
| --- | --- |
| `ArtifactStore` (JSON key/value) | `FileArtifactStore`, `MemoryArtifactStore` (`ragmake`); `SQLiteArtifactStore` (`ragmake.adapters.sqlite`); `PostgresArtifactStore` (`ragmake.adapters.pgvector`); `AzureBlobArtifactStore` (`ragmake.adapters.azure`) |
| `VectorStore` | `MemoryVectorStore` (`ragmake`); `SQLiteVectorStore` (`ragmake.adapters.sqlite`); `PgVectorStore` (`ragmake.adapters.pgvector`); `AzureAISearchVectorStore` (`ragmake.adapters.azure`) |
| `Embedder` | `HashingEmbedder` (`ragmake`); `OpenAIEmbedder` (`ragmake.adapters.openai`) |
| `NectarCompiler` | `HeuristicNectar` (`ragmake`); `OpenAINectar` (`ragmake.adapters.openai`) |

To add a backend, implement the matching Protocol in `ragmake/interfaces.py`. `Embedder` and
`NectarCompiler` also need a `name` that changes whenever their output would.

## Benchmark

`ragmake-benchmark` compares stateful ingest with a stateless rebuild. Phases: `cold` (first
ingest), `warm` (unchanged reingest), `changed` (reingest after 10% of documents were edited),
`stateless` (rebuild of the edited corpus from nothing).

```bash
ragmake-benchmark --documents 200       # sqlite backend; also --backend memory, --json
```

Real output of `PYTHONPATH=. python -m ragmake.benchmark --documents 200`:

```
sqlite: 200 documents, 20 changed
phase             processed          skipped  chunks_upserted         embedded          digests          reduces        ingest_ms
cold                    200                0              799              799              200                1            850.3
warm                      0              200                0                0                0                0              5.1
changed                  20              180               79               20               20                1             55.5
stateless               200                0              799              799              200                1            778.0
embedding reuse in changed phase: 97.5% of stateless
```

## Limits

- `HashingEmbedder` and `HeuristicNectar` are deterministic offline stand-ins for tests and
  demos. They match words, not meaning.
- `SQLiteVectorStore` is brute force: each query scores every chunk in the corpus in Python.
- Chunk sizes count whitespace-separated words, not tokens.
- Azure scores are Azure's relevance scores, not raw cosine. The pgvector and Azure adapters
  are only import-tested.

## License

MIT. See [LICENSE](LICENSE).
