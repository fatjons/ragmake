import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stateful_rag.adapters.sqlite_backend import SQLiteArtifactStore, SQLiteVectorStore
from stateful_rag.chunking import WordChunker
from stateful_rag.compiler import StatefulRAGCompiler
from stateful_rag.defaults import HashingEmbedder, HeuristicSynopsisCompiler
from stateful_rag.models import SourceDocument
from stateful_rag.runtime import StatefulRAGRuntime


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(dimensions=32)
        self.calls = 0
        self.texts = 0

    def embed_texts(self, texts):
        text_list = list(texts)
        if text_list:
            self.calls += 1
            self.texts += len(text_list)
        return super().embed_texts(text_list)


def build_stack(db_path: Path, embedder: CountingEmbedder):
    artifact_store = SQLiteArtifactStore(db_path)
    vector_store = SQLiteVectorStore(db_path)
    compiler = StatefulRAGCompiler(
        artifact_store=artifact_store,
        vector_store=vector_store,
        embedder=embedder,
        synopsis_compiler=HeuristicSynopsisCompiler(),
        chunker=WordChunker(max_words=200, overlap_words=20),
    )
    runtime = StatefulRAGRuntime(compiler=compiler, vector_store=vector_store, embedder=embedder)
    return compiler, runtime


def make_document(document_id: str, text: str) -> SourceDocument:
    return SourceDocument(corpus_id="manuals", document_id=document_id, text=text)


class IncrementalIngestTests(unittest.TestCase):
    def test_warm_reingest_skips_unchanged_documents_across_instances(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "stateful_rag.sqlite"
            document = make_document("runbook.txt", "Primer coupling depends on calibration cadence.")

            cold_embedder = CountingEmbedder()
            compiler, runtime = build_stack(db_path, cold_embedder)
            cold_report = compiler.ingest_documents([document])

            self.assertEqual(cold_report.processed_documents, 1)
            self.assertEqual(cold_embedder.texts, 1)

            cold_context = runtime.build_context("How does primer coupling work?", corpus_id="manuals", top_k=1)
            self.assertEqual(len(cold_context.chunks), 1)

            warm_embedder = CountingEmbedder()
            compiler, runtime = build_stack(db_path, warm_embedder)
            warm_report = compiler.ingest_documents([document])

            self.assertEqual(warm_report.processed_documents, 0)
            self.assertEqual(warm_report.skipped_documents, 1)
            self.assertEqual(warm_embedder.texts, 0)

            warm_context = runtime.build_context("How does primer coupling work?", corpus_id="manuals", top_k=1)
            self.assertEqual(len(warm_context.chunks), 1)
            self.assertEqual(warm_context.chunks[0].chunk.document_id, "runbook.txt")

    def test_changed_reingest_only_reprocesses_changed_documents(self):
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "stateful_rag.sqlite"
            base_documents = [
                make_document("alpha.txt", "Thermal balance affects fixation quality."),
                make_document("beta.txt", "Primer stability controls drying guidance."),
            ]
            changed_documents = [
                base_documents[0],
                make_document("beta.txt", "Primer stability now depends on humidity and drying cadence."),
            ]

            cold_embedder = CountingEmbedder()
            compiler, _ = build_stack(db_path, cold_embedder)
            compiler.ingest_documents(base_documents)

            changed_embedder = CountingEmbedder()
            compiler, runtime = build_stack(db_path, changed_embedder)
            report = compiler.ingest_documents(changed_documents)

            self.assertEqual(report.processed_documents, 1)
            self.assertEqual(report.skipped_documents, 1)
            self.assertEqual(changed_embedder.texts, 1)
            self.assertEqual(report.synopses_rebuilt, ["manuals"])

            context = runtime.build_context("What changed in drying guidance?", corpus_id="manuals", top_k=2)
            self.assertEqual(len(context.chunks), 2)


if __name__ == "__main__":
    unittest.main()
