"""Incremental Document Indexer.

Coordinates PDF parsing, chunking, embedding generation, Qdrant vector storage,
and SQLite status updating. Enforces incremental processing so already-indexed
documents are never unnecessarily re-embedded.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.database.database import DatabaseManager, get_db
from app.database.models import DocumentRecord, PipelineRecord
from app.database.repository import DocumentRepository, PipelineRepository
from app.ingestion.chunker import RecursiveChunker, TextChunk
from app.ingestion.embedder import BaseEmbedder, get_embedder
from app.ingestion.parser import PDFParser
from app.logging_config import logger, timed_operation

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.vectorstore.qdrant_client import VectorStoreClient


class DocumentIndexer:
    """Orchestrates the ingestion pipeline from PDF to Qdrant vectors."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embedder: Optional[BaseEmbedder] = None,
        vectorstore: Optional[VectorStoreClient] = None,
    ) -> None:
        self.db = db_manager or get_db()
        self.doc_repo = DocumentRepository(db=self.db)
        self.pipeline_repo = PipelineRepository(db=self.db)
        self.embedder = embedder or get_embedder()
        if vectorstore is None:
            from app.vectorstore.qdrant_client import get_vectorstore
            self.vectorstore = get_vectorstore()
        else:
            self.vectorstore = vectorstore

    def index_document(
        self,
        document_id: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        pipeline_version: str = "v1",
        force_reindex: bool = False,
    ) -> Tuple[bool, int, str]:
        """Index a single document incrementally.

        Returns:
            Tuple of (success_status, chunk_count, message)
        """
        doc = self.doc_repo.get_by_id(document_id)
        if not doc:
            return False, 0, f"Document ID '{document_id}' not found in database."

        if doc.status == "INDEXED" and not force_reindex:
            try:
                vectors_in_store = self.vectorstore.count_document_vectors(document_id)
            except Exception:
                vectors_in_store = 0

            if vectors_in_store > 0:
                logger.info(
                    f"Document '{doc.filename}' already indexed ({vectors_in_store} vectors in Qdrant). Skipping."
                )
                return True, doc.chunk_count, f"Document '{doc.filename}' is already indexed."
            else:
                logger.warning(
                    f"Document '{doc.filename}' is marked INDEXED in database but has 0 vectors in Qdrant. Re-indexing automatically..."
                )

        start_time = time.perf_counter()
        with timed_operation("index_document", {"document_id": document_id, "filename": doc.filename}):
            # 1. If re-indexing, purge old vectors first
            if force_reindex:
                self.vectorstore.delete_document_vectors(document_id)

            # 2. Parse PDF with path resolution for host vs container
            from pathlib import Path
            p_obj = Path(doc.filepath)
            if not p_obj.exists():
                candidate = Path(settings.UPLOAD_DIR) / p_obj.name
                if candidate.exists():
                    p_obj = candidate
            parsed = PDFParser.parse_file(str(p_obj), doc.filename, doc.document_id)
            if not parsed.is_extractable or not parsed.pages:
                doc.status = "FAILED"
                doc.processing_time = (time.perf_counter() - start_time) * 1000.0
                self.doc_repo.update(doc)
                err = parsed.error_message or "No extractable text found in PDF."
                return False, 0, err

            # 3. Chunk pages
            chunker = RecursiveChunker(
                chunk_size=chunk_size or settings.CHUNK_SIZE,
                chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
            )
            chunks: List[TextChunk] = chunker.chunk_document(
                parsed, pipeline_version=pipeline_version
            )

            if not chunks:
                doc.status = "FAILED"
                self.doc_repo.update(doc)
                return False, 0, "No chunks could be generated from document."

            # 4. Generate embeddings
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_texts(texts)

            # 5. Upsert to Qdrant
            self.vectorstore.upsert_chunks(chunks, vectors)

            # 6. Update SQLite metadata
            total_duration_ms = (time.perf_counter() - start_time) * 1000.0
            doc.status = "INDEXED"
            doc.page_count = parsed.page_count
            doc.chunk_count = len(chunks)
            doc.processing_time = round(total_duration_ms, 2)
            doc.embedding_model = self.embedder.model_name
            doc.pipeline_version = pipeline_version
            self.doc_repo.update(doc)

            logger.info(
                f"Successfully indexed '{doc.filename}': {len(chunks)} chunks in {total_duration_ms:.1f}ms."
            )
            return True, len(chunks), f"Indexed {len(chunks)} chunks in {total_duration_ms:.1f}ms."

    def index_all_unindexed(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        pipeline_version: str = "v1",
    ) -> Dict[str, Any]:
        """Incrementally index any documents that are unindexed or missing vectors in Qdrant."""
        all_docs = self.doc_repo.list_all()
        unindexed = []
        for d in all_docs:
            if d.status != "INDEXED":
                unindexed.append(d)
            else:
                try:
                    if self.vectorstore.count_document_vectors(d.document_id) == 0:
                        unindexed.append(d)
                except Exception:
                    unindexed.append(d)

        results = {
            "total_discovered": len(all_docs),
            "unindexed_found": len(unindexed),
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": [],
        }

        start_time = time.perf_counter()
        for doc in unindexed:
            success, count, msg = self.index_document(
                document_id=doc.document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pipeline_version=pipeline_version,
                force_reindex=False,
            )
            if success:
                results["processed"] += 1
                results["total_chunks"] += count
            else:
                results["failed"] += 1
                results["errors"].append({"document_id": doc.document_id, "error": msg})

        results["duration_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return results

    def bootstrap_preingested_documents(self) -> Dict[str, Any]:
        """Automatically ensure all documents on disk and codebase are ingested & indexed.

        Runs safely on application startup. Detects any unindexed or missing vector points
        and indexes them automatically so the user never has to manually re-index on restart.
        """
        from pathlib import Path
        from app.documents.manager import DocumentManager

        doc_mgr = DocumentManager(db_manager=self.db)
        upload_path = Path(settings.UPLOAD_DIR)

        discovered = 0
        newly_uploaded = 0

        # 1. Discover any files on disk in data/documents
        if upload_path.exists():
            for fpath in upload_path.glob("*"):
                if fpath.is_file() and not fpath.name.startswith("."):
                    discovered += 1
                    try:
                        file_bytes = fpath.read_bytes()
                        parts = fpath.name.split("_", 1)
                        clean_name = (
                            parts[1]
                            if len(parts) > 1 and len(parts[0]) == 36
                            else fpath.name
                        )
                        _, is_new, _ = doc_mgr.upload_document(
                            file_bytes=file_bytes,
                            filename=clean_name,
                        )
                        if is_new:
                            newly_uploaded += 1
                    except Exception as exc:
                        logger.debug(f"Bootstrap scan error on '{fpath.name}': {exc}")

        # 2. Check and index any unindexed or missing vectors in Qdrant
        index_res = self.index_all_unindexed()
        index_res["discovered_files"] = discovered
        index_res["newly_uploaded"] = newly_uploaded
        return index_res

    def reindex_all(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        pipeline_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-index all documents under a new pipeline configuration."""
        ver = pipeline_version or f"v{int(time.time())}"
        all_docs = self.doc_repo.list_all()

        results = {
            "total_documents": len(all_docs),
            "processed": 0,
            "failed": 0,
            "total_chunks": 0,
            "pipeline_version": ver,
            "errors": [],
        }

        start_time = time.perf_counter()
        for doc in all_docs:
            success, count, msg = self.index_document(
                document_id=doc.document_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pipeline_version=ver,
                force_reindex=True,
            )
            if success:
                results["processed"] += 1
                results["total_chunks"] += count
            else:
                results["failed"] += 1
                results["errors"].append({"document_id": doc.document_id, "error": msg})

        results["duration_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return results
