"""Unit tests for retrieval engine and similarity thresholding."""

import unittest

from app.ingestion.chunker import TextChunk
from app.ingestion.embedder import MockEmbedder
from app.rag.retriever import RetrievalEngine
from app.vectorstore.qdrant_client import VectorStoreClient


class TestRetrievalEngine(unittest.TestCase):
    """Test suite for semantic retrieval and confidence threshold behavior."""

    def setUp(self) -> None:
        self.embedder = MockEmbedder(dimension=384)
        self.vectorstore = VectorStoreClient(
            collection_name="retrieval_test",
            dimension=384,
            use_in_memory=True,
        )
        self.retriever = RetrievalEngine(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

        # Seed test chunks
        c1 = TextChunk(
            chunk_id="c1",
            document_id="doc-rag",
            filename="rag_paper.pdf",
            page=4,
            chunk_index=0,
            text="Retrieval-Augmented Generation couples external knowledge with LLMs.",
            char_count=67,
            token_estimate=16,
            pipeline_version="v1",
        )
        c2 = TextChunk(
            chunk_id="c2",
            document_id="doc-opt",
            filename="optimizer.pdf",
            page=12,
            chunk_index=0,
            text="Gradient descent updates model parameters iteratively using gradients.",
            char_count=71,
            token_estimate=17,
            pipeline_version="v1",
        )
        vectors = self.embedder.embed_texts([c1.text, c2.text])
        self.vectorstore.upsert_chunks([c1, c2], vectors)

    def test_high_confidence_retrieval(self) -> None:
        """Verify relevant query retrieves chunks above threshold."""
        trace = self.retriever.retrieve(
            query="Retrieval-Augmented Generation couples knowledge",
            top_k=5,
            threshold=0.30,
        )
        self.assertFalse(trace.is_low_confidence)
        self.assertTrue(len(trace.filtered_chunks) > 0)
        self.assertEqual(trace.filtered_chunks[0].chunk_id, "c1")
        self.assertEqual(trace.filtered_chunks[0].page, 4)
        self.assertEqual(trace.filtered_chunks[0].filename, "rag_paper.pdf")
        self.assertTrue(trace.embedding_time_ms >= 0.0)
        self.assertTrue(trace.retrieval_time_ms >= 0.0)

    def test_low_confidence_rejection(self) -> None:
        """Verify that when threshold is set strictly higher than similarity, low confidence is flagged."""
        trace = self.retriever.retrieve(
            query="completely unrelated astronomy query about black holes",
            top_k=5,
            threshold=0.99,  # Impossibly high threshold
        )
        self.assertTrue(trace.is_low_confidence)
        self.assertEqual(len(trace.filtered_chunks), 0)

    def test_empty_query_handling(self) -> None:
        """Verify empty string returns immediate low confidence without error."""
        trace = self.retriever.retrieve(query="   ", top_k=5, threshold=0.45)
        self.assertTrue(trace.is_low_confidence)
        self.assertEqual(len(trace.retrieved_chunks), 0)


if __name__ == "__main__":
    unittest.main()
