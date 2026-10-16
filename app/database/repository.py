"""Repository classes for database CRUD operations."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from app.database.database import DatabaseManager, get_db
from app.database.models import DocumentRecord, PipelineRecord, QueryRecord


class DocumentRepository:
    """Handles CRUD operations for documents table."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or get_db()

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            document_id=row["document_id"],
            filename=row["filename"],
            filepath=row["filepath"],
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            upload_date=str(row["upload_date"]),
            status=row["status"],
            processing_time=row["processing_time"],
            embedding_model=row["embedding_model"],
            pipeline_version=row["pipeline_version"],
        )

    def insert(self, record: DocumentRecord) -> DocumentRecord:
        sql = """
        INSERT INTO documents (
            document_id, filename, filepath, file_hash, file_size,
            page_count, chunk_count, status, processing_time,
            embedding_model, pipeline_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.session() as conn:
            cursor = conn.execute(
                sql,
                (
                    record.document_id,
                    record.filename,
                    record.filepath,
                    record.file_hash,
                    record.file_size,
                    record.page_count,
                    record.chunk_count,
                    record.status,
                    record.processing_time,
                    record.embedding_model,
                    record.pipeline_version,
                ),
            )
            record.id = cursor.lastrowid
        return record

    def update(self, record: DocumentRecord) -> None:
        sql = """
        UPDATE documents SET
            filename = ?,
            filepath = ?,
            file_hash = ?,
            file_size = ?,
            page_count = ?,
            chunk_count = ?,
            status = ?,
            processing_time = ?,
            embedding_model = ?,
            pipeline_version = ?
        WHERE document_id = ?
        """
        with self.db.session() as conn:
            conn.execute(
                sql,
                (
                    record.filename,
                    record.filepath,
                    record.file_hash,
                    record.file_size,
                    record.page_count,
                    record.chunk_count,
                    record.status,
                    record.processing_time,
                    record.embedding_model,
                    record.pipeline_version,
                    record.document_id,
                ),
            )

    def get_by_id(self, document_id: str) -> Optional[DocumentRecord]:
        sql = "SELECT * FROM documents WHERE document_id = ?"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (document_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        sql = "SELECT * FROM documents WHERE file_hash = ?"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (file_hash,))
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def list_all(self) -> List[DocumentRecord]:
        sql = "SELECT * FROM documents ORDER BY upload_date DESC"
        with self.db.session() as conn:
            cursor = conn.execute(sql)
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def delete(self, document_id: str) -> bool:
        sql = "DELETE FROM documents WHERE document_id = ?"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (document_id,))
            return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        sql = """
        SELECT
            COUNT(*) as total_documents,
            COALESCE(SUM(page_count), 0) as total_pages,
            COALESCE(SUM(chunk_count), 0) as total_chunks,
            COALESCE(SUM(file_size), 0) as total_bytes
        FROM documents
        WHERE status = 'INDEXED'
        """
        with self.db.session() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            return {
                "total_documents": row["total_documents"],
                "total_pages": row["total_pages"],
                "total_chunks": row["total_chunks"],
                "total_bytes": row["total_bytes"],
            }


class PipelineRepository:
    """Handles CRUD operations for pipelines table."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or get_db()

    def _row_to_record(self, row: sqlite3.Row) -> PipelineRecord:
        return PipelineRecord(
            id=row["id"],
            pipeline_id=row["pipeline_id"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            chunking_strategy=row["chunking_strategy"],
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            distance_metric=row["distance_metric"],
            top_k=row["top_k"],
            similarity_threshold=row["similarity_threshold"],
            created_at=str(row["created_at"]),
        )

    def insert(self, record: PipelineRecord) -> PipelineRecord:
        sql = """
        INSERT INTO pipelines (
            pipeline_id, chunk_size, chunk_overlap, chunking_strategy,
            embedding_model, embedding_dimension, distance_metric,
            top_k, similarity_threshold
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.session() as conn:
            cursor = conn.execute(
                sql,
                (
                    record.pipeline_id,
                    record.chunk_size,
                    record.chunk_overlap,
                    record.chunking_strategy,
                    record.embedding_model,
                    record.embedding_dimension,
                    record.distance_metric,
                    record.top_k,
                    record.similarity_threshold,
                ),
            )
            record.id = cursor.lastrowid
        return record

    def get_latest(self) -> Optional[PipelineRecord]:
        sql = "SELECT * FROM pipelines ORDER BY created_at DESC, id DESC LIMIT 1"
        with self.db.session() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def list_all(self) -> List[PipelineRecord]:
        sql = "SELECT * FROM pipelines ORDER BY created_at DESC"
        with self.db.session() as conn:
            cursor = conn.execute(sql)
            return [self._row_to_record(row) for row in cursor.fetchall()]


class QueryRepository:
    """Handles CRUD operations and aggregation for queries table."""

    def __init__(self, db: Optional[DatabaseManager] = None) -> None:
        self.db = db or get_db()

    def _row_to_record(self, row: sqlite3.Row) -> QueryRecord:
        keys = row.keys()
        return QueryRecord(
            id=row["id"],
            query_id=row["query_id"],
            query=row["query"],
            timestamp=str(row["timestamp"]),
            model=row["model"],
            top_k=row["top_k"],
            threshold=row["threshold"],
            retrieved_count=row["retrieved_count"],
            retrieval_time=row["retrieval_time"],
            generation_time=row["generation_time"],
            total_time=row["total_time"],
            max_similarity=row["max_similarity"],
            is_low_confidence=bool(row["is_low_confidence"]),
            input_validation=row["input_validation"] if "input_validation" in keys else "PASSED",
            input_length=row["input_length"] if "input_length" in keys else len(row["query"]),
            retrieval_completed=bool(row["retrieval_completed"]) if "retrieval_completed" in keys else True,
            retrieved_candidate_count=row["retrieved_candidate_count"] if "retrieved_candidate_count" in keys else row["retrieved_count"],
            evidence_sufficient=bool(row["evidence_sufficient"]) if "evidence_sufficient" in keys else not bool(row["is_low_confidence"]),
            guardrail_triggered=bool(row["guardrail_triggered"]) if "guardrail_triggered" in keys else False,
            guardrail_reason=row["guardrail_reason"] if "guardrail_reason" in keys else "",
            llm_used=bool(row["llm_used"]) if "llm_used" in keys else (row["generation_time"] > 0),
            output_validation=row["output_validation"] if "output_validation" in keys else "PASSED",
            output_guardrail_triggered=bool(row["output_guardrail_triggered"]) if "output_guardrail_triggered" in keys else False,
            output_guardrail_reason=row["output_guardrail_reason"] if "output_guardrail_reason" in keys else "",
            final_result_type=row["final_result_type"] if "final_result_type" in keys else "GROUNDED_ANSWER",
        )

    def insert(self, record: QueryRecord) -> QueryRecord:
        sql = """
        INSERT INTO queries (
            query_id, query, model, top_k, threshold,
            retrieved_count, retrieval_time, generation_time,
            total_time, max_similarity, is_low_confidence,
            input_validation, input_length, retrieval_completed,
            retrieved_candidate_count, evidence_sufficient,
            guardrail_triggered, guardrail_reason, llm_used,
            output_validation, output_guardrail_triggered,
            output_guardrail_reason, final_result_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.session() as conn:
            cursor = conn.execute(
                sql,
                (
                    record.query_id,
                    record.query,
                    record.model,
                    record.top_k,
                    record.threshold,
                    record.retrieved_count,
                    record.retrieval_time,
                    record.generation_time,
                    record.total_time,
                    record.max_similarity,
                    record.is_low_confidence,
                    record.input_validation,
                    record.input_length,
                    record.retrieval_completed,
                    record.retrieved_candidate_count,
                    record.evidence_sufficient,
                    record.guardrail_triggered,
                    record.guardrail_reason,
                    record.llm_used,
                    record.output_validation,
                    record.output_guardrail_triggered,
                    record.output_guardrail_reason,
                    record.final_result_type,
                ),
            )
            record.id = cursor.lastrowid
        return record

    def list_recent(self, limit: int = 50) -> List[QueryRecord]:
        sql = "SELECT * FROM queries ORDER BY timestamp DESC, id DESC LIMIT ?"
        with self.db.session() as conn:
            cursor = conn.execute(sql, (limit,))
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        sql = """
        SELECT
            COUNT(*) as total_queries,
            COALESCE(AVG(retrieval_time), 0.0) as avg_retrieval_time,
            COALESCE(AVG(generation_time), 0.0) as avg_generation_time,
            COALESCE(AVG(total_time), 0.0) as avg_total_time,
            COALESCE(AVG(max_similarity), 0.0) as avg_similarity,
            SUM(CASE WHEN is_low_confidence = 1 THEN 1 ELSE 0 END) as low_confidence_queries,
            SUM(CASE WHEN is_low_confidence = 0 THEN 1 ELSE 0 END) as successful_retrievals
        FROM queries
        """
        with self.db.session() as conn:
            cursor = conn.execute(sql)
            row = cursor.fetchone()
            return {
                "total_queries": row["total_queries"],
                "avg_retrieval_time": round(float(row["avg_retrieval_time"]), 2),
                "avg_generation_time": round(float(row["avg_generation_time"]), 2),
                "avg_total_time": round(float(row["avg_total_time"]), 2),
                "avg_similarity": round(float(row["avg_similarity"]), 4),
                "low_confidence_queries": row["low_confidence_queries"] or 0,
                "successful_retrievals": row["successful_retrievals"] or 0,
            }
