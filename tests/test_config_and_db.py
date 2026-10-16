"""Unit tests for configuration and database repository operations."""

import os
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.database.database import DatabaseManager
from app.database.models import DocumentRecord, PipelineRecord, QueryRecord
from app.database.repository import DocumentRepository, PipelineRepository, QueryRepository


class TestConfigAndDatabase(unittest.TestCase):
    """Test suite verifying settings loading and database repositories."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_rag.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.doc_repo = DocumentRepository(db=self.db_manager)
        self.pipeline_repo = PipelineRepository(db=self.db_manager)
        self.query_repo = QueryRepository(db=self.db_manager)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_settings_initialization(self) -> None:
        """Verify settings load valid defaults and normalize paths."""
        settings = Settings()
        self.assertEqual(settings.OLLAMA_MODEL, "llama3.2:1b")
        self.assertEqual(settings.EMBEDDING_DIMENSION, 384)
        self.assertEqual(settings.DISTANCE_METRIC, "Cosine")
        self.assertTrue(settings.CHUNK_SIZE > 0)
        self.assertTrue(settings.SIMILARITY_THRESHOLD >= 0.0)

    def test_document_lifecycle_in_database(self) -> None:
        """Verify DocumentRepository CRUD operations."""
        record = DocumentRecord(
            document_id="doc-test-123",
            filename="sample_paper.pdf",
            filepath="/tmp/sample_paper.pdf",
            file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            file_size=102400,
            page_count=5,
            chunk_count=20,
            status="INDEXED",
            processing_time=125.5,
            embedding_model="BAAI/bge-small-en-v1.5",
            pipeline_version="v1.0",
        )

        inserted = self.doc_repo.insert(record)
        self.assertIsNotNone(inserted.id)

        # Retrieve by document_id
        fetched = self.doc_repo.get_by_id("doc-test-123")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.filename, "sample_paper.pdf")
        self.assertEqual(fetched.page_count, 5)
        self.assertEqual(fetched.chunk_count, 20)
        self.assertEqual(fetched.status, "INDEXED")

        # Retrieve by hash
        by_hash = self.doc_repo.get_by_hash("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        self.assertIsNotNone(by_hash)
        self.assertEqual(by_hash.document_id, "doc-test-123")

        # Update record
        fetched.chunk_count = 25
        self.doc_repo.update(fetched)
        updated = self.doc_repo.get_by_id("doc-test-123")
        self.assertEqual(updated.chunk_count, 25)

        # Check stats
        stats = self.doc_repo.get_stats()
        self.assertEqual(stats["total_documents"], 1)
        self.assertEqual(stats["total_pages"], 5)
        self.assertEqual(stats["total_chunks"], 25)

        # Delete record
        deleted = self.doc_repo.delete("doc-test-123")
        self.assertTrue(deleted)
        self.assertIsNone(self.doc_repo.get_by_id("doc-test-123"))

    def test_pipeline_repository(self) -> None:
        """Verify PipelineRepository insert and retrieve."""
        pipeline = PipelineRecord(
            pipeline_id="pipe-v1",
            chunk_size=512,
            chunk_overlap=64,
            chunking_strategy="recursive",
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dimension=384,
            distance_metric="Cosine",
            top_k=5,
            similarity_threshold=0.45,
        )
        self.pipeline_repo.insert(pipeline)

        latest = self.pipeline_repo.get_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.pipeline_id, "pipe-v1")
        self.assertEqual(latest.chunk_size, 512)

    def test_query_repository_and_stats(self) -> None:
        """Verify QueryRepository logs and calculates aggregate stats."""
        q1 = QueryRecord(
            query_id="q-1",
            query="What is RAG?",
            model="llama3.2:1b",
            top_k=5,
            threshold=0.45,
            retrieved_count=3,
            retrieval_time=45.2,
            generation_time=1200.0,
            total_time=1245.2,
            max_similarity=0.82,
            is_low_confidence=False,
        )
        q2 = QueryRecord(
            query_id="q-2",
            query="Unknown subject query",
            model="llama3.2:1b",
            top_k=5,
            threshold=0.45,
            retrieved_count=0,
            retrieval_time=30.0,
            generation_time=0.0,
            total_time=30.0,
            max_similarity=0.21,
            is_low_confidence=True,
        )

        self.query_repo.insert(q1)
        self.query_repo.insert(q2)

        recent = self.query_repo.list_recent(10)
        self.assertEqual(len(recent), 2)

        stats = self.query_repo.get_stats()
        self.assertEqual(stats["total_queries"], 2)
        self.assertEqual(stats["successful_retrievals"], 1)
        self.assertEqual(stats["low_confidence_queries"], 1)
        self.assertTrue(stats["avg_retrieval_time"] > 0)


if __name__ == "__main__":
    unittest.main()
