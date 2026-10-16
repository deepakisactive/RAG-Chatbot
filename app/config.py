"""Centralized Application Configuration.

Loads settings from environment variables and .env file with validated defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _parse_env_file(filepath: Path) -> Dict[str, str]:
    """Parse a simple .env file into key-value pairs."""
    env_vars: Dict[str, str] = {}
    if not filepath.exists():
        return env_vars

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                env_vars[key.strip()] = val.strip().strip("'\"")
    return env_vars


class Settings:
    """Application configuration settings."""

    def __init__(self, env_file: str | Path | None = None) -> None:
        # Determine base directory
        base_dir = Path(__file__).resolve().parent.parent
        env_path = Path(env_file) if env_file else base_dir / ".env"
        file_vars = _parse_env_file(env_path)

        def get_val(key: str, default: Any, cast_type: type) -> Any:
            val = os.environ.get(key, file_vars.get(key, default))
            if cast_type == bool:
                if isinstance(val, bool):
                    return val
                return str(val).lower() in ("true", "1", "yes", "on")
            return cast_type(val)

        # Ollama LLM Runtime (Running on Windows Host)
        self.OLLAMA_BASE_URL: str = get_val("OLLAMA_BASE_URL", "http://10.0.2.2:11434", str)
        self.OLLAMA_MODEL: str = get_val("OLLAMA_MODEL", "llama3.2:1b", str)
        self.OLLAMA_TIMEOUT_SECONDS: float = get_val("OLLAMA_TIMEOUT_SECONDS", 60.0, float)
        self.LLM_TEMPERATURE: float = get_val("LLM_TEMPERATURE", 0.2, float)
        self.LLM_NUM_PREDICT: int = get_val("LLM_NUM_PREDICT", 256, int)

        # Qdrant Vector Database
        is_in_container = os.path.exists("/.dockerenv")
        default_qdrant_url = "http://qdrant:6333" if is_in_container else "http://localhost:6333"
        raw_qdrant = get_val("QDRANT_URL", default_qdrant_url, str)
        if not is_in_container and "://qdrant:" in raw_qdrant:
            raw_qdrant = raw_qdrant.replace("://qdrant:", "://localhost:")
        self.QDRANT_URL: str = raw_qdrant
        self.QDRANT_COLLECTION: str = get_val("QDRANT_COLLECTION", "rag_documents", str)

        # Embedding Configuration
        self.EMBEDDING_MODEL: str = get_val("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5", str)
        self.EMBEDDING_DIMENSION: int = get_val("EMBEDDING_DIMENSION", 384, int)
        self.DISTANCE_METRIC: str = get_val("DISTANCE_METRIC", "Cosine", str)

        # Ingestion & Chunking
        self.CHUNK_SIZE: int = get_val("CHUNK_SIZE", 512, int)
        self.CHUNK_OVERLAP: int = get_val("CHUNK_OVERLAP", 64, int)
        self.CHUNKING_STRATEGY: str = get_val("CHUNKING_STRATEGY", "recursive", str)

        # Retrieval
        self.TOP_K: int = get_val("TOP_K", 5, int)
        self.SIMILARITY_THRESHOLD: float = get_val("SIMILARITY_THRESHOLD", 0.55, float)

        # File & Database Paths
        raw_db_path = get_val("SQLITE_DB_PATH", "metadata/rag.db", str)
        raw_upload_dir = get_val("UPLOAD_DIR", "data/documents", str)

        # Normalize relative paths to base directory
        self.SQLITE_DB_PATH: Path = (
            Path(raw_db_path) if Path(raw_db_path).is_absolute() else (base_dir / raw_db_path).resolve()
        )
        self.UPLOAD_DIR: Path = (
            Path(raw_upload_dir) if Path(raw_upload_dir).is_absolute() else (base_dir / raw_upload_dir).resolve()
        )
        self.MAX_UPLOAD_SIZE_MB: int = get_val("MAX_UPLOAD_SIZE_MB", 50, int)
        self.LOG_LEVEL: str = get_val("LOG_LEVEL", "INFO", str)

        # Guardrails Configuration
        self.MAX_QUERY_LENGTH: int = get_val("MAX_QUERY_LENGTH", 500, int)
        self.GUARDRAILS_ENABLED: bool = get_val("GUARDRAILS_ENABLED", True, bool)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to a serializable dictionary."""
        return {
            "OLLAMA_BASE_URL": self.OLLAMA_BASE_URL,
            "OLLAMA_MODEL": self.OLLAMA_MODEL,
            "OLLAMA_TIMEOUT_SECONDS": self.OLLAMA_TIMEOUT_SECONDS,
            "LLM_TEMPERATURE": self.LLM_TEMPERATURE,
            "LLM_NUM_PREDICT": self.LLM_NUM_PREDICT,
            "QDRANT_URL": self.QDRANT_URL,
            "QDRANT_COLLECTION": self.QDRANT_COLLECTION,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "EMBEDDING_DIMENSION": self.EMBEDDING_DIMENSION,
            "DISTANCE_METRIC": self.DISTANCE_METRIC,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "CHUNKING_STRATEGY": self.CHUNKING_STRATEGY,
            "TOP_K": self.TOP_K,
            "SIMILARITY_THRESHOLD": self.SIMILARITY_THRESHOLD,
            "MAX_QUERY_LENGTH": self.MAX_QUERY_LENGTH,
            "GUARDRAILS_ENABLED": self.GUARDRAILS_ENABLED,
            "SQLITE_DB_PATH": str(self.SQLITE_DB_PATH),
            "UPLOAD_DIR": str(self.UPLOAD_DIR),
            "MAX_UPLOAD_SIZE_MB": self.MAX_UPLOAD_SIZE_MB,
            "LOG_LEVEL": self.LOG_LEVEL,
        }


# Global singleton instance
settings = Settings()
