import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stateful_rag.benchmark import render_summary, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_changed_document_savings_use_ingest_embeddings(self):
        report = run_benchmark(document_count=12, words_per_document=80, change_fraction=0.25, backend="memory", top_k=3)
        savings = report.changed_document_savings
        phase_map = {phase.name: phase for phase in report.phases}

        self.assertEqual(savings.changed_documents, 3)
        self.assertEqual(savings.unchanged_documents, 9)
        self.assertEqual(report.warm_skip_rate, 1.0)
        self.assertEqual(phase_map["stateful_warm"].processed_documents, 0)
        self.assertEqual(phase_map["stateful_changed"].query_embedded_texts, 1)
        self.assertEqual(phase_map["stateless_changed"].query_embedded_texts, 1)
        self.assertEqual(
            savings.embedded_texts_avoided,
            phase_map["stateless_changed"].ingest_embedded_texts - phase_map["stateful_changed"].ingest_embedded_texts,
        )

    def test_render_summary_focuses_on_changed_path(self):
        report = run_benchmark(document_count=10, words_per_document=60, change_fraction=0.2, backend="memory", top_k=2)
        summary = render_summary(report)

        self.assertIn("Changed-document savings", summary)
        self.assertIn("document reuse:", summary)
        self.assertIn("embedding reuse:", summary)
        self.assertIn("Changed-path detail", summary)


if __name__ == "__main__":
    unittest.main()
