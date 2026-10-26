import tempfile
import unittest

from ragmake import (Chunker, Compiler, Document, FileArtifactStore, HashingEmbedder, HeuristicNectar,
                     MemoryArtifactStore, MemoryVectorStore, Runtime, SessionStore)


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dimensions=256):
        super().__init__(dimensions)
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return super().embed_texts(texts)


def make(embedder=None, artifacts=None):
    return Compiler(artifacts or MemoryArtifactStore(), MemoryVectorStore(), embedder or CountingEmbedder(),
                    HeuristicNectar())


def docs(n, corpus="c", tag=""):
    return [Document(corpus, f"d{i}", f"Widget{i} notes {tag}. All about gadget{i} and sprocket{i}.") for i in range(n)]


class CompilerTests(unittest.TestCase):
    def test_cold_then_warm(self):
        c = make()
        r = c.ingest(docs(3))
        self.assertEqual((r.processed, r.embedded, r.chunks_upserted, r.nectars_rebuilt), (3, 3, 3, ["c"]))
        calls = c.embedder.calls
        r = c.ingest(docs(3))
        self.assertEqual((r.processed, r.skipped, r.embedded, r.nectars_rebuilt), (0, 3, 0, []))
        self.assertEqual(c.embedder.calls, calls)

    def test_changed_doc_only(self):
        c = make()
        c.ingest(docs(3))
        d = docs(3)
        d[1].text += " Now with extra words."
        r = c.ingest(d)
        self.assertEqual((r.processed, r.skipped, r.embedded, r.nectars_rebuilt), (1, 2, 1, ["c"]))

    def test_title_and_metadata_changes_count(self):
        c = make()
        c.ingest(docs(2))
        d = docs(2)
        d[0].title = "New title"
        r = c.ingest(d)
        self.assertEqual((r.processed, r.embedded, r.nectars_rebuilt), (1, 0, ["c"]))
        d[1].metadata = {"k": 1}
        self.assertEqual(c.ingest(d).processed, 1)

    def test_prune(self):
        c = make()
        c.ingest(docs(3))
        r = c.ingest(docs(2), prune=True)
        self.assertEqual((r.removed, r.chunks_deleted, r.nectars_rebuilt), (1, 1, ["c"]))
        self.assertIsNone(c.ledger("c", "d2"))
        hits = Runtime(c).search("Widget2 notes gadget2 sprocket2", "c", 10)
        self.assertEqual(sorted(h.chunk.document_id for h in hits), ["d0", "d1"])

    def test_remove(self):
        c = make()
        c.ingest(docs(3))
        self.assertEqual(c.remove("c", ["d0", "nope"]), 1)
        self.assertEqual([e["document_id"] for e in c.load_nectar("c").catalog], ["d1", "d2"])
        c.remove("c", ["d1", "d2"])
        self.assertIsNone(c.load_nectar("c"))

    def test_catalog_has_every_document(self):
        c = make()
        c.ingest(docs(100))
        nectar = c.load_nectar("c")
        self.assertEqual(sorted(e["document_id"] for e in nectar.catalog), sorted(f"d{i}" for i in range(100)))
        self.assertEqual((nectar.document_count, nectar.chunk_count), (100, 100))
        self.assertLessEqual(len(nectar.text), 8000)

    def test_recipe_change_rebuilds_and_cache_is_per_embedder(self):
        artifacts = MemoryArtifactStore()
        make(CountingEmbedder(256), artifacts).ingest(docs(3))
        r = make(CountingEmbedder(64), artifacts).ingest(docs(3))
        self.assertEqual((r.processed, r.embedded, r.nectars_rebuilt), (3, 3, ["c"]))
        r = make(CountingEmbedder(256), artifacts).ingest(docs(3))
        self.assertEqual((r.processed, r.embedded), (3, 0))

    def test_corpus_prefix_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = make(artifacts=FileArtifactStore(tmp))
            c.ingest(docs(1, "abc") + docs(2, "abcd"))
            self.assertEqual(c.load_nectar("abc").document_count, 1)
            self.assertEqual(c.load_nectar("abcd").document_count, 2)


class ChunkerTests(unittest.TestCase):
    def test_edit_in_last_paragraph_keeps_earlier_chunks(self):
        def doc(last):
            paras = [" ".join(f"p{i}w{j}" for j in range(60)) for i in range(5)] + [last]
            return Document("c", "d", "\n\n".join(paras))

        a, b = (Chunker().chunk(doc(t)) for t in ("tail one", "tail two"))
        self.assertGreaterEqual(len(a), 2)
        self.assertEqual([x.content_hash for x in a[:-1]], [x.content_hash for x in b[:-1]])
        self.assertNotEqual(a[-1].content_hash, b[-1].content_hash)

    def test_oversized_paragraph_windows(self):
        text = " ".join(f"w{i}" for i in range(500))
        chunks = Chunker().chunk(Document("c", "d", text, title="T"))
        self.assertEqual([len(c.text.split()) for c in chunks], [200, 200, 140])
        self.assertEqual(chunks[1].text, text[chunks[1].char_start:chunks[1].char_end])
        self.assertEqual(chunks[0].metadata, {"title": "T"})

    def test_invalid_args(self):
        for kw in ({"max_words": 0}, {"overlap_words": 200}, {"min_words": 201}):
            with self.assertRaises(ValueError):
                Chunker(**kw)


class StoreTests(unittest.TestCase):
    def test_file_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = FileArtifactStore(tmp)
            for bad in ("../x.json", "a/../x.json", "/x.json"):
                with self.assertRaises(ValueError):
                    s.write_json(bad, 1)
            s.write_json("a/abc/x.json", 1)
            s.write_json("a/abcd/y.json", 2)
            self.assertEqual([p for p, _ in s.iter_json("a/abc/")], ["a/abc/x.json"])
            self.assertEqual([p for p, _ in s.iter_json("a/ab")], ["a/abc/x.json", "a/abcd/y.json"])
            self.assertEqual(list(s.iter_json("zzz/")), [])
            s.delete("a/abc/x.json")
            self.assertIsNone(s.read_json("a/abc/x.json"))


class RuntimeTests(unittest.TestCase):
    def test_context_payload_and_session(self):
        c = make()
        c.ingest(docs(3))
        sessions = SessionStore(c.artifacts, max_events=2)
        rt = Runtime(c, sessions)
        sessions.record("s 1", "intent", "find widgets", concepts=["widgets"])
        sessions.record("s 1", "learning", "one", entities=["acme"])
        sessions.record("s 1", "learning", "two")
        sessions.record("s 1", "learning", "three")
        self.assertEqual(sessions.summary("s 1", recent=1),
                         {"concepts": ["widgets"], "entities": ["acme"], "recent": {"learning": ["three"]}})
        self.assertIsNone(sessions.summary("nope"))

        ctx = rt.context("Widget1 gadget1", "c", top_k=2, session_id="s 1", catalog_limit=2)
        self.assertEqual(set(ctx), {"query", "corpus_id", "signature", "nectar", "catalog", "sources", "session"})
        self.assertEqual(len(ctx["catalog"]), 2)
        self.assertEqual(ctx["sources"][0]["document_id"], "d1")
        self.assertEqual(set(ctx["sources"][0]),
                         {"document_id", "title", "chunk_id", "score", "char_start", "char_end", "text"})
        self.assertNotIn("session", rt.context("x", "c"))
        empty = rt.context("x", "missing")
        self.assertEqual((empty["nectar"], empty["catalog"], empty["sources"]), (None, [], []))


if __name__ == "__main__":
    unittest.main()
