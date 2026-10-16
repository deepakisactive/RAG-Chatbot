"""Unit tests for recursive text chunking and embedding generation."""

import math
import unittest

from app.ingestion.chunker import RecursiveChunker, TextChunk
from app.ingestion.embedder import MockEmbedder, get_embedder
from app.ingestion.parser import ParsedPDF, PDFPage


class TestChunkerAndEmbedder(unittest.TestCase):
    """Test suite for chunking logic and embedding generation."""

    def setUp(self) -> None:
        self.chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        self.embedder = MockEmbedder(dimension=384)

    def test_chunk_page_preserves_provenance(self) -> None:
        """Verify that chunking a page retains the page number and generates structured chunk IDs."""
        text = (
            "Paragraph one introduces the core concepts of Retrieval-Augmented Generation.\n\n"
            "Paragraph two explains the role of dense vector embeddings in semantic search.\n\n"
            "Paragraph three covers prompt engineering and hallucination prevention strategies."
        )
        page = PDFPage.from_text(page_number=3, text=text)
        chunks = self.chunker.chunk_page(
            page=page,
            document_id="doc-xyz",
            filename="paper.pdf",
            pipeline_version="v1",
        )

        self.assertTrue(len(chunks) >= 1)
        for chunk in chunks:
            self.assertEqual(chunk.document_id, "doc-xyz")
            self.assertEqual(chunk.filename, "paper.pdf")
            self.assertEqual(chunk.page, 3)
            self.assertEqual(chunk.pipeline_version, "v1")
            self.assertTrue(chunk.chunk_id.startswith("doc-xyz_p3_c"))
            self.assertTrue(len(chunk.text) > 0)
            self.assertTrue(chunk.char_count == len(chunk.text))

    def test_chunk_document_multi_page(self) -> None:
        """Verify chunking a full multi-page document."""
        p1 = PDFPage.from_text(page_number=1, text="Introduction to local AI architectures.")
        p2 = PDFPage.from_text(page_number=2, text="Evaluation metrics: Recall@K and MRR.")
        parsed = ParsedPDF(
            document_id="doc-multi",
            filename="report.pdf",
            filepath="/tmp/report.pdf",
            file_size_bytes=2048,
            file_hash="dummyhash",
            page_count=2,
            total_characters=len(p1.text) + len(p2.text),
            pages=[p1, p2],
        )

        chunks = self.chunker.chunk_document(parsed, pipeline_version="v2")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[1].page, 2)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[1].chunk_index, 1)

    def test_chunk_overlap_validation(self) -> None:
        """Verify error when overlap is greater than or equal to chunk size."""
        with self.assertRaises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_embedder_dimension_and_normalization(self) -> None:
        """Verify embedding dimension is 384 and vectors are L2-normalized."""
        texts = ["What is vector search?", "Llama 3.2 running locally on Windows."]
        embeddings = self.embedder.embed_texts(texts)

        self.assertEqual(len(embeddings), 2)
        for emb in embeddings:
            self.assertEqual(len(emb), 384)
            # Check L2 unit norm: sum of squares == 1.0 within floating tolerance
            l2_norm = math.sqrt(sum(x * x for x in emb))
            self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_embed_query(self) -> None:
        """Verify single query embedding generation."""
        query = "Explain cosine similarity in Qdrant"
        query_vec = self.embedder.embed_query(query)
        self.assertEqual(len(query_vec), 384)
        l2_norm = math.sqrt(sum(x * x for x in query_vec))
        self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_embedder_determinism(self) -> None:
        """Verify identical text produces identical embedding vector."""
        vec1 = self.embedder.embed_query("deterministic check")
        vec2 = self.embedder.embed_query("deterministic check")
        self.assertEqual(vec1, vec2)


if __name__ == "__main__":
    unittest.main()
