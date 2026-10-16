"""Document Manager Module.

Orchestrates document storage, deduplication, lifecycle state transitions,
and atomic deletion across filesystem, SQLite metadata, and vector store.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.database.database import DatabaseManager, get_db
from app.database.models import DocumentRecord
from app.database.repository import DocumentRepository
from app.ingestion.parser import PDFParser, ParsedPDF
from app.logging_config import logger, timed_operation


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and special character issues."""
    name = Path(filename).name
    # Keep alphanumeric, dots, dashes, and underscores
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return clean or "document.pdf"


class DocumentManager:
    """Manages document uploads, deduplication, storage, and cleanup."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        upload_dir: Optional[Path | str] = None,
    ) -> None:
        self.db = db_manager or get_db()
        self.doc_repo = DocumentRepository(db=self.db)
        self.upload_dir = Path(upload_dir) if upload_dir else settings.UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def upload_document(
        self,
        file_bytes: bytes,
        filename: str,
        pipeline_version: str = "v1",
        embedding_model: Optional[str] = None,
    ) -> Tuple[DocumentRecord, bool, str]:
        """Upload, validate, and store a PDF document with duplicate detection.

        Returns:
            Tuple of (DocumentRecord, is_new_upload, user_message)
        """
        # 1. Enforce size limit
        file_size = len(file_bytes)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            err_msg = f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed limit ({settings.MAX_UPLOAD_SIZE_MB} MB)."
            logger.warning(err_msg)
            raise ValueError(err_msg)

        if file_size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        # 2. Compute SHA-256 hash for deduplication
        file_hash = PDFParser.calculate_hash(file_bytes)
        existing_doc = self.doc_repo.get_by_hash(file_hash)
        if existing_doc:
            logger.info(
                f"Duplicate upload detected: '{filename}' matches existing document ID {existing_doc.document_id} ('{existing_doc.filename}')."
            )
            return (
                existing_doc,
                False,
                f"Document already exists: '{existing_doc.filename}' (ID: {existing_doc.document_id[:8]}...). Upload skipped.",
            )

        # 3. Generate unique ID and save file to disk
        document_id = str(uuid.uuid4())
        safe_name = sanitize_filename(filename)
        saved_filename = f"{document_id}_{safe_name}"
        saved_path = self.upload_dir / saved_filename

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        model_name = embedding_model or settings.EMBEDDING_MODEL

        # 4. Create initial DB record with PROCESSING status
        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            filepath=str(saved_path),
            file_hash=file_hash,
            file_size=file_size,
            page_count=0,
            chunk_count=0,
            status="PROCESSING",
            processing_time=0.0,
            embedding_model=model_name,
            pipeline_version=pipeline_version,
        )
        self.doc_repo.insert(record)

        # 5. Parse PDF to extract pages and validate text extractability
        start_time = time.perf_counter()
        parsed = PDFParser.parse_file(saved_path, safe_name, document_id)
        parse_duration_ms = (time.perf_counter() - start_time) * 1000.0

        if not parsed.is_extractable:
            record.status = "FAILED"
            record.processing_time = parse_duration_ms
            record.page_count = parsed.page_count
            self.doc_repo.update(record)
            msg = f"Failed to extract text: {parsed.error_message}"
            logger.warning(f"Document {document_id} marked as FAILED: {msg}")
            return record, True, msg

        # Update parsed metrics
        record.page_count = parsed.page_count
        record.processing_time = parse_duration_ms
        self.doc_repo.update(record)

        logger.info(
            f"Successfully uploaded and parsed '{safe_name}' (ID: {document_id}, {parsed.page_count} pages, {file_size} bytes)."
        )
        return record, True, f"Successfully uploaded '{safe_name}' ({parsed.page_count} pages)."

    def delete_document(
        self,
        document_id: str,
        vectorstore_client: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """Atomically delete a document across filesystem, SQLite metadata, and Qdrant.

        Returns:
            Tuple of (success_status, message)
        """
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            return False, f"Document ID '{document_id}' not found."

        with timed_operation("delete_document", {"document_id": document_id, "filename": doc.filename}):
            # 1. Clean Qdrant vectors if client provided
            if vectorstore_client is not None:
                try:
                    vectorstore_client.delete_document_vectors(document_id)
                    logger.info(f"Deleted vector points for document_id '{document_id}' from Qdrant.")
                except Exception as exc:
                    logger.error(f"Error removing vectors from Qdrant for '{document_id}': {exc}")

            # 2. Remove physical file from disk
            try:
                file_path = Path(doc.filepath)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted physical file: {file_path}")
            except Exception as exc:
                logger.warning(f"Failed to delete physical file {doc.filepath}: {exc}")

            # 3. Remove metadata from SQLite
            deleted = self.doc_repo.delete(document_id)
            if deleted:
                return True, f"Document '{doc.filename}' and all associated metadata and vectors deleted successfully."
            return False, f"Failed to remove document record for '{document_id}' from database."

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        """Fetch document record by document ID."""
        return self.doc_repo.get_by_id(document_id)

    def list_documents(self) -> List[DocumentRecord]:
        """List all tracked documents ordered by upload date descending."""
        return self.doc_repo.list_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate metrics for indexed documents."""
        return self.doc_repo.get_stats()

    def get_parsed_document(self, document_id: str) -> Optional[ParsedPDF]:
        """Retrieve and parse a document from disk on demand."""
        doc = self.get_document(document_id)
        if not doc:
            return None
        return PDFParser.parse_file(doc.filepath, doc.filename, doc.document_id)
