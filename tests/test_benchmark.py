import contextlib
import io
import json
import unittest

from ragmake.benchmark import main, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_stateful_reuse(self):
        r = run_benchmark(documents=12, change_fraction=0.25, backend="memory")
        cold, warm, changed, stateless = (r["phases"][k] for k in ("cold", "warm", "changed", "stateless"))
        self.assertEqual((warm["embedded"], warm["digests"], warm["reduces"], warm["skipped"]), (0, 0, 0, 12))
        self.assertEqual((changed["processed"], changed["digests"], changed["reduces"]), (3, 3, 1))
        self.assertLess(changed["embedded"], changed["chunks_upserted"])
        self.assertGreater(stateless["embedded"], changed["embedded"])
        self.assertGreater(cold["chunks_upserted"], 12 * 2)
        self.assertGreater(r["embedding_reuse"], 0.5)

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["--documents", "6", "--backend", "sqlite", *argv])
        return out.getvalue()

    def test_main_formats(self):
        self.assertEqual(json.loads(self.run_main("--json"))["documents"], 6)
        text = self.run_main()
        self.assertEqual(len(text.splitlines()), 7)
        self.assertIn("embedding reuse", text)

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            run_benchmark(change_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
