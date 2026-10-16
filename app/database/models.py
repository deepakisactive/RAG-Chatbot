"""Data models for SQLite metadata records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DocumentRecord:
    """Represents a PDF document record in the metadata database."""

    document_id: str
    filename: str
    filepath: str
    file_hash: str
    file_size: int
    page_count: int = 0
    chunk_count: int = 0
    upload_date: Optional[str] = None
    status: str = "PROCESSING"  # 'PROCESSING', 'INDEXED', 'FAILED'
    processing_time: float = 0.0
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    pipeline_version: str = "v1"
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class PipelineRecord:
    """Represents an indexing pipeline configuration."""

    pipeline_id: str
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    embedding_model: str
    embedding_dimension: int
    distance_metric: str
    top_k: int
    similarity_threshold: float
    created_at: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class QueryRecord:
    """Represents a logged RAG query and execution telemetry."""

    query_id: str
    query: str
    model: str
    top_k: int
    threshold: float
    retrieved_count: int
    retrieval_time: float
    generation_time: float
    total_time: float
    max_similarity: float
    is_low_confidence: bool = False
    timestamp: Optional[str] = None
    id: Optional[int] = None
    input_validation: str = "PASSED"
    input_length: int = 0
    retrieval_completed: bool = True
    retrieved_candidate_count: int = 0
    evidence_sufficient: bool = False
    guardrail_triggered: bool = False
    guardrail_reason: str = ""
    llm_used: bool = False
    output_validation: str = "PASSED"
    output_guardrail_triggered: bool = False
    output_guardrail_reason: str = ""
    final_result_type: str = "GROUNDED_ANSWER"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

