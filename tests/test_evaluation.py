"""Unit tests for RAG quantitative evaluation (Recall@K, Precision@K, MRR)."""

import unittest

from app.ingestion.chunker import TextChunk
from app.ingestion.embedder import MockEmbedder
from app.rag.evaluator import EvalSample, RAGEvaluator
from app.rag.retriever import RetrievalEngine
from app.vectorstore.qdrant_client import VectorStoreClient


class TestRAGEvaluation(unittest.TestCase):
    """Test suite verifying quantitative evaluation metrics calculation."""

    def setUp(self) -> None:
        self.embedder = MockEmbedder(dimension=384)
        self.vectorstore = VectorStoreClient(
            collection_name="eval_collection",
            dimension=384,
            use_in_memory=True,
        )
        self.retriever = RetrievalEngine(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        self.evaluator = RAGEvaluator(retriever=self.retriever)

        # Seed document chunks
        c1 = TextChunk(
            chunk_id="eval_c1",
            document_id="doc_attention",
            filename="attention_paper.pdf",
            page=3,
            chunk_index=0,
            text="The Transformer model relies entirely on self-attention mechanisms without recurrence.",
            char_count=87,
            token_estimate=21,
            pipeline_version="v1",
        )
        c2 = TextChunk(
            chunk_id="eval_c2",
            document_id="doc_bert",
            filename="bert_paper.pdf",
            page=1,
            chunk_index=0,
            text="BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
            char_count=85,
            token_estimate=21,
            pipeline_version="v1",
        )
        vectors = self.embedder.embed_texts([c1.text, c2.text])
        self.vectorstore.upsert_chunks([c1, c2], vectors)

    def test_evaluator_metrics_calculation(self) -> None:
        """Verify Recall@K and MRR computations match ground-truth assertions."""
        dataset = [
            EvalSample(
                query="The Transformer model relies entirely on self-attention mechanisms without recurrence.",
                expected_answer="Transformers rely on self-attention",
                relevant_document="attention_paper.pdf",
                relevant_page=3,
            ),
            EvalSample(
                query="BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
                expected_answer="BERT pre-trains bidirectional representations",
                relevant_document="bert_paper.pdf",
                relevant_page=1,
            ),
        ]

        report = self.evaluator.evaluate_dataset(dataset, top_k=2, threshold=0.10)

        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["recall_at_k"], 1.0)
        self.assertEqual(report["mean_reciprocal_rank"], 1.0)
        self.assertTrue(report["avg_precision_at_k"] > 0.0)

        # Detail assertions
        details = report["detailed_results"]
        self.assertEqual(len(details), 2)
        self.assertTrue(details[0]["hit_at_k"])
        self.assertEqual(details[0]["reciprocal_rank"], 1.0)


if __name__ == "__main__":
    unittest.main()
