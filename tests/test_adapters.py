import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace as NS

from ragmake.adapters.openai import OpenAIEmbedder, OpenAINectar
from ragmake.adapters.sqlite import SQLiteArtifactStore, SQLiteVectorStore
from ragmake.models import Chunk, Document


def chunk(key, corpus="c", **meta):
    return Chunk(corpus, "d", 0, key, f"text {key}", 0, 5, "h" + key, meta)


class SQLiteTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = os.path.join(tmp.name, "r.db")

    def test_artifacts(self):
        s = SQLiteArtifactStore(self.db)
        self.addCleanup(s.close)
        self.assertIsNone(s.read_json("x"))
        for path in ["a/b/1", "a/b%/2", "a/b_/3", "a/c/4", "a/bx/5"]:
            s.write_json(path, {"p": path})
        s.write_json("a/b/1", {"p": "new"})
        self.assertEqual(s.read_json("a/b/1"), {"p": "new"})
        self.assertEqual([p for p, _ in s.iter_json("a/b")], ["a/b%/2", "a/b/1", "a/b_/3", "a/bx/5"])
        self.assertEqual([p for p, _ in s.iter_json("a/b%")], ["a/b%/2"])
        self.assertEqual([p for p, _ in s.iter_json("a/b_")], ["a/b_/3"])
        s.delete("a/b/1")
        self.assertIsNone(s.read_json("a/b/1"))

    def test_vectors(self):
        s = SQLiteVectorStore(self.db)
        s.upsert_chunks([chunk("x", k=1), chunk("y"), chunk("z", corpus="other")], [[1, 0], [0.8, 0.6], [1, 0]])
        hits = s.search("c", [3, 0], 5)
        self.assertEqual([h.chunk.key for h in hits], ["x", "y"])
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        self.assertAlmostEqual(hits[1].score, 0.8, places=5)
        self.assertEqual(hits[0].chunk, chunk("x", k=1))
        self.assertEqual(len(s.search("c", [1, 0], 1)), 1)
        s.upsert_chunks([chunk("x")], [[0, 1]])
        self.assertEqual(s.search("c", [1, 0], 1)[0].chunk.key, "y")
        s.delete_chunks(["y", "missing"])
        self.assertEqual([h.chunk.key for h in s.search("c", [1, 0], 5)], ["x"])
        s.close()

    def test_persistence_sharing_and_close(self):
        a, v = SQLiteArtifactStore(self.db), SQLiteVectorStore(self.db)
        a.write_json("k", [1])
        v.upsert_chunks([chunk("x")], [[1, 0]])
        a.close()
        v.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            a.read_json("k")
        a, v = SQLiteArtifactStore(self.db), SQLiteVectorStore(self.db)
        self.assertEqual(a.read_json("k"), [1])
        self.assertEqual(v.search("c", [1, 0], 1)[0].chunk.key, "x")
        a.close()
        v.close()


class OpenAITests(unittest.TestCase):
    def test_embedder_batches(self):
        calls = []

        def create(model, input, **kw):
            calls.append((model, list(input), kw))
            return NS(data=[NS(index=i, embedding=[float(len(t))]) for i, t in reversed(list(enumerate(input)))])

        e = OpenAIEmbedder(NS(embeddings=NS(create=create)), "m", dimensions=8, batch_size=2)
        self.assertEqual(e.name, "openai:m:8")
        self.assertEqual(e.embed_texts(["a", "bb", "ccc"]), [[1.0], [2.0], [3.0]])
        self.assertEqual([len(c[1]) for c in calls], [2, 1])
        self.assertEqual(calls[0][2], {"dimensions": 8})
        self.assertEqual(OpenAIEmbedder(None, "m").name, "openai:m:native")

    def nectar(self, **kw):
        self.prompts = []

        def create(messages, **_):
            self.prompts.append(messages[1]["content"])
            return NS(choices=[NS(message=NS(content=f" out{len(self.prompts)} "))])

        return OpenAINectar(NS(chat=NS(completions=NS(create=create))), "m", **kw)

    def test_digest_truncates(self):
        n = self.nectar(doc_chars=5)
        self.assertEqual(n.digest(Document("c", "d", "0123456789", title="T")), "out1")
        self.assertEqual(self.prompts[0], "Title: T\n\n01234")

    def test_reduce_single_call(self):
        n = self.nectar()
        self.assertEqual(n.reduce("c", [("a", "x"), ("b", "y")], "old"), "out1")
        self.assertEqual(len(self.prompts), 1)
        self.assertIn("- a: x\n- b: y", self.prompts[0])
        self.assertIn("old", self.prompts[0])

    def test_reduce_recurses_over_budget(self):
        n = self.nectar(budget_chars=60)
        digests = [(f"d{i}", "x" * 20) for i in range(6)]  # 6 lines of 27 chars: 3 batches of 2
        n.reduce("c", digests, "old")
        # 3 partials, then one final reduce over 3 short "part" lines
        self.assertEqual(len(self.prompts), 4)
        self.assertNotIn("old", "".join(self.prompts[:3]))
        self.assertIn("- part 1: out1", self.prompts[3])
        self.assertIn("old", self.prompts[3])


class ImportTests(unittest.TestCase):
    def test_optional_modules_import_without_deps(self):
        import ragmake.adapters.azure, ragmake.adapters.pgvector  # noqa: F401


if __name__ == "__main__":
    unittest.main()
