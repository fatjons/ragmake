# CLAUDE.md

## How work is done here

- The main session is the **senior architect and senior engineer**. It owns the
  design, writes the contracts (`models.py`, `interfaces.py`), splits work into
  well-specified jobs, and **reviews every line** an agent produces before it is
  committed. Nothing lands unreviewed.
- Coding is delegated to agents: **Sonnet 5.5** (`model: sonnet`) for real
  engineering (core logic, adapters, tricky tests), **Haiku 5.5**
  (`model: haiku`) for mechanical work (renames, docs sync, config, trimming).
- Every agent prompt carries the exact spec plus the quality bar below. Agents
  on disjoint files may run in parallel; agents never commit.

## Quality bar (Linus standard)

- Low LOC. No bloat. Every line earns its place. Delete before you add.
- Good taste: pick data structures that make special cases disappear.
- No speculative abstraction, no placeholder config for things that don't
  exist, no boilerplate `from_dict` when `Cls(**d)` works, no defensive code for
  impossible states.
- Correctness first: content hashes decide rebuilds, caches are keyed by
  everything that affects their value, paths are never built from unescaped
  user input.
- Comments only where the *why* isn't obvious. Names say what things are.
- Stdlib only in core. Optional deps are imported lazily inside adapters.

## Project

ragmake: stateful RAG kernel. Documents compile into persistent state, like a
build system: unchanged inputs are never reprocessed.

- **Digest**: per-document summary, rebuilt only when that document's hash changes.
- **Nectar**: corpus-level summary reduced from all digests plus a catalog of every
  document. Rebuilt only when the corpus signature (set of doc hashes) changes.
- **Evidence**: query-time retrieved chunks.

Layout: `ragmake/` (core), `ragmake/adapters/` (sqlite, pgvector, azure, openai),
`tests/`. Run tests: `python -m unittest discover -s tests`.
