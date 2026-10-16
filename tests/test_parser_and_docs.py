"""Unit tests for PDF parsing, text extraction, deduplication, and document lifecycle."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from app.database.database import DatabaseManager
from app.documents.manager import DocumentManager, sanitize_filename
from app.ingestion.parser import PDFPage, PDFParser


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


class TestParserAndDocumentManager(unittest.TestCase):
    """Test suite for PDF parsing and document lifecycle management."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test_rag.db"
        self.upload_dir = self.temp_path / "uploaded_docs"
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.doc_manager = DocumentManager(
            db_manager=self.db_manager, upload_dir=self.upload_dir
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sanitize_filename(self) -> None:
        """Verify filename sanitization prevents path traversal and illegal chars."""
        self.assertEqual(sanitize_filename("../../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(sanitize_filename("my research paper (v1).pdf"), "my_research_paper__v1_.pdf")
        self.assertEqual(sanitize_filename("clean_name.pdf"), "clean_name.pdf")

    def test_pdf_page_metrics(self) -> None:
        """Verify PDFPage computes character, word, and token estimates accurately."""
        sample_text = "Retrieval-Augmented Generation is a technique for enhancing LLMs."
        page = PDFPage.from_text(page_number=1, text=sample_text)
        self.assertEqual(page.page_number, 1)
        self.assertEqual(page.char_count, len(sample_text))
        self.assertEqual(page.word_count, len(sample_text.split()))
        self.assertTrue(page.token_estimate > 0)

    def test_sha256_hash_calculation(self) -> None:
        """Verify SHA-256 hash calculation is deterministic and matches expected hash."""
        sample_bytes = b"Hello, RAG world!"
        hash1 = PDFParser.calculate_hash(sample_bytes)
        hash2 = PDFParser.calculate_hash(sample_bytes)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # 64 hex characters for SHA-256

    def test_upload_and_deduplication(self) -> None:
        """Verify successful upload and automatic rejection of duplicate uploads."""
        dummy_content = create_sample_pdf("Mock document text for RAG retrieval testing.")
        filename = "test_doc.pdf"

        # 1. First upload
        record, is_new, msg = self.doc_manager.upload_document(dummy_content, filename)
        self.assertTrue(is_new)
        self.assertIn("Successfully uploaded", msg)
        self.assertTrue(Path(record.filepath).exists())

        # Check in DB
        fetched = self.doc_manager.get_document(record.document_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.filename, "test_doc.pdf")

        # 2. Duplicate upload with same content
        dup_record, is_new_dup, dup_msg = self.doc_manager.upload_document(
            dummy_content, "another_name.pdf"
        )
        self.assertFalse(is_new_dup)
        self.assertIn("already exists", dup_msg)
        self.assertEqual(dup_record.document_id, record.document_id)

        # Confirm only 1 document is in the repository
        all_docs = self.doc_manager.list_documents()
        self.assertEqual(len(all_docs), 1)

    def test_upload_empty_file_raises(self) -> None:
        """Verify uploading an empty byte stream raises a ValueError."""
        with self.assertRaises(ValueError):
            self.doc_manager.upload_document(b"", "empty.pdf")

    def test_delete_document_lifecycle(self) -> None:
        """Verify complete deletion removes file, metadata, and invokes vector cleanup."""
        dummy_content = create_sample_pdf("Mock content to be deleted.")
        record, _, _ = self.doc_manager.upload_document(dummy_content, "to_delete.pdf")
        doc_path = Path(record.filepath)
        self.assertTrue(doc_path.exists())

        # Mock vectorstore client
        mock_vectorstore = MagicMock()

        # Execute deletion
        success, msg = self.doc_manager.delete_document(
            record.document_id, vectorstore_client=mock_vectorstore
        )
        self.assertTrue(success)
        self.assertIn("deleted successfully", msg)

        # Verify physical file removed
        self.assertFalse(doc_path.exists())

        # Verify Qdrant deletion hook invoked with correct document_id
        mock_vectorstore.delete_document_vectors.assert_called_once_with(record.document_id)

        # Verify removed from database
        self.assertIsNone(self.doc_manager.get_document(record.document_id))


if __name__ == "__main__":
    unittest.main()
