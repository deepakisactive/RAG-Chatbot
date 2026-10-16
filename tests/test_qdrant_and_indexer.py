"""Unit tests for Qdrant integration, vector store operations, and incremental indexing."""

import tempfile
import unittest
from pathlib import Path

from app.database.database import DatabaseManager
from app.documents.manager import DocumentManager
from app.ingestion.chunker import TextChunk
from app.ingestion.embedder import MockEmbedder
from app.ingestion.indexer import DocumentIndexer
from app.vectorstore.qdrant_client import VectorStoreClient


def create_sample_pdf(text: str) -> bytes:
    """Generate a real valid PDF using PyMuPDF if installed, else fallback bytes."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    except Exception:
        return text.encode("utf-8")


class TestQdrantAndIndexer(unittest.TestCase):
    """Test suite for Qdrant vector store client and incremental indexer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test_rag.db"
        self.upload_dir = self.temp_path / "docs"

        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.embedder = MockEmbedder(dimension=384)
        # Use in-memory store for isolated, deterministic unit testing
        self.vectorstore = VectorStoreClient(
            collection_name="test_collection",
            dimension=384,
            use_in_memory=True,
        )
        self.doc_manager = DocumentManager(
            db_manager=self.db_manager, upload_dir=self.upload_dir
        )
        self.indexer = DocumentIndexer(
            db_manager=self.db_manager,
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_vectorstore_upsert_search_and_delete(self) -> None:
        """Verify vector store upserting, similarity search, and payload-filtered deletion."""
        chunk1 = TextChunk(
            chunk_id="doc1_p1_c0000",
            document_id="doc1",
            filename="paper1.pdf",
            page=1,
            chunk_index=0,
            text="Machine learning models require training on representative data.",
            char_count=65,
            token_estimate=16,
            pipeline_version="v1",
        )
        chunk2 = TextChunk(
            chunk_id="doc2_p1_c0000",
            document_id="doc2",
            filename="paper2.pdf",
            page=1,
            chunk_index=0,
            text="Quantum computing utilizes qubits for superposition and entanglement.",
            char_count=69,
            token_estimate=17,
            pipeline_version="v1",
        )

        vectors = self.embedder.embed_texts([chunk1.text, chunk2.text])
        count = self.vectorstore.upsert_chunks([chunk1, chunk2], vectors)
        self.assertEqual(count, 2)
        self.assertEqual(self.vectorstore.count_vectors(), 2)

        # Search for query related to chunk 1
        query_vec = self.embedder.embed_query("machine learning models")
        results = self.vectorstore.search(query_vec, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_id, "doc1_p1_c0000")
        self.assertEqual(results[0].rank, 1)
        self.assertTrue(results[0].similarity_score > results[1].similarity_score)

        # Filtered deletion by document_id
        self.vectorstore.delete_document_vectors("doc1")
        self.assertEqual(self.vectorstore.count_vectors(), 1)

        # Search again: doc1 must not be present
        post_del_results = self.vectorstore.search(query_vec, top_k=2)
        self.assertEqual(len(post_del_results), 1)
        self.assertEqual(post_del_results[0].document_id, "doc2")

    def test_incremental_indexing(self) -> None:
        """Verify that incremental indexing only processes unindexed documents."""
        # 1. Upload Doc 1 with valid PDF binary
        content1 = create_sample_pdf("Artificial intelligence and neural network systems.")
        doc1_rec, _, _ = self.doc_manager.upload_document(content1, "doc1.pdf")

        # 2. Index Doc 1
        success, count1, _ = self.indexer.index_document(doc1_rec.document_id)
        self.assertTrue(success)
        self.assertTrue(count1 > 0)

        doc1_after = self.doc_manager.get_document(doc1_rec.document_id)
        self.assertEqual(doc1_after.status, "INDEXED")

        # 3. Upload Doc 2 with valid PDF binary
        content2 = create_sample_pdf("Natural language processing and token embeddings.")
        doc2_rec, _, _ = self.doc_manager.upload_document(content2, "doc2.pdf")
        doc2_before = self.doc_manager.get_document(doc2_rec.document_id)
        self.assertEqual(doc2_before.status, "PROCESSING")

        # 4. Run index_all_unindexed
        batch_results = self.indexer.index_all_unindexed()
        self.assertEqual(batch_results["total_discovered"], 2)
        self.assertEqual(batch_results["unindexed_found"], 1)  # Only doc2!
        self.assertEqual(batch_results["processed"], 1)

        # Confirm both are indexed now
        doc2_after = self.doc_manager.get_document(doc2_rec.document_id)
        self.assertEqual(doc2_after.status, "INDEXED")

    def test_reindex_all_purges_old_vectors(self) -> None:
        """Verify reindexing re-creates chunks under a new pipeline version."""
        content = create_sample_pdf("Fast vector search using hierarchical navigable small world graphs.")
        doc_rec, _, _ = self.doc_manager.upload_document(content, "hnsw.pdf")
        self.indexer.index_document(doc_rec.document_id, pipeline_version="v1.0")

        # Re-index under v2.0
        reindex_res = self.indexer.reindex_all(pipeline_version="v2.0")
        self.assertEqual(reindex_res["processed"], 1)
        self.assertEqual(reindex_res["pipeline_version"], "v2.0")

        doc_updated = self.doc_manager.get_document(doc_rec.document_id)
        self.assertEqual(doc_updated.pipeline_version, "v2.0")


if __name__ == "__main__":
    unittest.main()
