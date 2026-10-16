"""SQLite Database Manager.

Handles thread-safe database connections, initialization, WAL mode,
and table schema migrations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from app.config import settings
from app.logging_config import logger


class DatabaseManager:
    """Manages SQLite connection lifecycle and schema setup."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.SQLITE_DB_PATH
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row

        # Configure SQLite for performance and concurrency
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a transactional connection."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error(f"Database transaction error: {exc}", exc_info=True)
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            page_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL CHECK(status IN ('PROCESSING', 'INDEXED', 'FAILED')),
            processing_time REAL DEFAULT 0.0,
            embedding_model TEXT NOT NULL,
            pipeline_version TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);
        CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
        CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date DESC);

        CREATE TABLE IF NOT EXISTS pipelines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id TEXT NOT NULL UNIQUE,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            chunking_strategy TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            distance_metric TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            similarity_threshold REAL NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_pipelines_created_at ON pipelines(created_at DESC);

        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL UNIQUE,
            query TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            threshold REAL NOT NULL,
            retrieved_count INTEGER NOT NULL,
            retrieval_time REAL NOT NULL,
            generation_time REAL NOT NULL,
            total_time REAL NOT NULL,
            max_similarity REAL NOT NULL,
            is_low_confidence BOOLEAN NOT NULL DEFAULT 0,
            input_validation TEXT DEFAULT 'PASSED',
            input_length INTEGER DEFAULT 0,
            retrieval_completed BOOLEAN DEFAULT 1,
            retrieved_candidate_count INTEGER DEFAULT 0,
            evidence_sufficient BOOLEAN DEFAULT 0,
            guardrail_triggered BOOLEAN DEFAULT 0,
            guardrail_reason TEXT DEFAULT '',
            llm_used BOOLEAN DEFAULT 0,
            output_validation TEXT DEFAULT 'PASSED',
            output_guardrail_triggered BOOLEAN DEFAULT 0,
            output_guardrail_reason TEXT DEFAULT '',
            final_result_type TEXT DEFAULT 'GROUNDED_ANSWER'
        );

        CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp DESC);
        """
        new_columns = [
            ("input_validation", "TEXT DEFAULT 'PASSED'"),
            ("input_length", "INTEGER DEFAULT 0"),
            ("retrieval_completed", "BOOLEAN DEFAULT 1"),
            ("retrieved_candidate_count", "INTEGER DEFAULT 0"),
            ("evidence_sufficient", "BOOLEAN DEFAULT 0"),
            ("guardrail_triggered", "BOOLEAN DEFAULT 0"),
            ("guardrail_reason", "TEXT DEFAULT ''"),
            ("llm_used", "BOOLEAN DEFAULT 0"),
            ("output_validation", "TEXT DEFAULT 'PASSED'"),
            ("output_guardrail_triggered", "BOOLEAN DEFAULT 0"),
            ("output_guardrail_reason", "TEXT DEFAULT ''"),
            ("final_result_type", "TEXT DEFAULT 'GROUNDED_ANSWER'"),
        ]
        with self.session() as conn:
            conn.executescript(schema_sql)
            cursor = conn.execute("PRAGMA table_info(queries);")
            existing_cols = {row[1] for row in cursor.fetchall()}
            for col_name, col_def in new_columns:
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE queries ADD COLUMN {col_name} {col_def};")
                    except Exception as err:
                        logger.debug(f"Migration note for {col_name}: {err}")
        logger.debug(f"Initialized SQLite database at {self.db_path}")


# Global default database manager
_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Retrieve or initialize singleton DatabaseManager instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
