"""Data transfer objects for retrieval traces and RAG responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.vectorstore.qdrant_client import ScoredChunk


@dataclass
class SourceCitation:
    """Represents an attributed citation for the user interface."""

    filename: str
    document_id: str
    page: int
    similarity_score: float
    chunk_id: str
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTrace:
    """Detailed observability trace of the retrieval process."""

    query: str
    top_k: int
    threshold: float
    distance_metric: str
    retrieved_chunks: List[ScoredChunk] = field(default_factory=list)
    filtered_chunks: List[ScoredChunk] = field(default_factory=list)
    highest_similarity: float = 0.0
    is_low_confidence: bool = False
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "distance_metric": self.distance_metric,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "filtered_chunks": [c.to_dict() for c in self.filtered_chunks],
            "highest_similarity": self.highest_similarity,
            "is_low_confidence": self.is_low_confidence,
            "embedding_time_ms": self.embedding_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class RAGResponse:
    """Complete structured response from the RAG pipeline."""

    query: str
    answer: str
    sources: List[SourceCitation]
    trace: RetrievalTrace
    context_sent: str
    model: str
    embedding_time_ms: float
    retrieval_time_ms: float
    context_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    is_low_confidence: bool = False
    is_general_knowledge: bool = False
    has_relevant_evidence: bool = False
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
    guardrails_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "trace": self.trace.to_dict(),
            "context_sent": self.context_sent,
            "model": self.model,
            "embedding_time_ms": self.embedding_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "context_time_ms": self.context_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "is_low_confidence": self.is_low_confidence,
            "is_general_knowledge": self.is_general_knowledge,
            "has_relevant_evidence": self.has_relevant_evidence,
            "input_validation": self.input_validation,
            "input_length": self.input_length,
            "retrieval_completed": self.retrieval_completed,
            "retrieved_candidate_count": self.retrieved_candidate_count,
            "evidence_sufficient": self.evidence_sufficient,
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_reason": self.guardrail_reason,
            "llm_used": self.llm_used,
            "output_validation": self.output_validation,
            "output_guardrail_triggered": self.output_guardrail_triggered,
            "output_guardrail_reason": self.output_guardrail_reason,
            "final_result_type": self.final_result_type,
            "guardrails_enabled": self.guardrails_enabled,
        }


@dataclass
class ComparisonResponse:
    """Side-by-side comparison response containing both Grounded RAG and RAG + LLM outputs."""

    query: str
    model: str
    grounded_response: RAGResponse
    rag_llm_response: RAGResponse
    trace: RetrievalTrace
    retrieved_count: int
    highest_similarity: float
    grounded_context_retrieved: bool
    grounded_llm_used: bool
    grounded_generation_time_s: float
    grounded_result_label: str
    rag_llm_context_retrieved: bool
    rag_llm_used: bool
    rag_llm_generation_time_s: float
    rag_llm_result_label: str
    has_relevant_evidence: bool = False
    relevance_reason: str = ""
    input_validation: str = "PASSED"
    input_length: int = 0
    retrieval_completed: bool = True
    retrieved_candidate_count: int = 0
    evidence_sufficient: bool = False
    guardrails_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "model": self.model,
            "grounded_response": self.grounded_response.to_dict(),
            "rag_llm_response": self.rag_llm_response.to_dict(),
            "trace": self.trace.to_dict(),
            "retrieved_count": self.retrieved_count,
            "highest_similarity": self.highest_similarity,
            "grounded_context_retrieved": self.grounded_context_retrieved,
            "grounded_llm_used": self.grounded_llm_used,
            "grounded_generation_time_s": self.grounded_generation_time_s,
            "grounded_result_label": self.grounded_result_label,
            "rag_llm_context_retrieved": self.rag_llm_context_retrieved,
            "rag_llm_used": self.rag_llm_used,
            "rag_llm_generation_time_s": self.rag_llm_generation_time_s,
            "rag_llm_result_label": self.rag_llm_result_label,
            "has_relevant_evidence": self.has_relevant_evidence,
            "relevance_reason": self.relevance_reason,
            "input_validation": self.input_validation,
            "input_length": self.input_length,
            "retrieval_completed": self.retrieval_completed,
            "retrieved_candidate_count": self.retrieved_candidate_count,
            "evidence_sufficient": self.evidence_sufficient,
            "guardrails_enabled": self.guardrails_enabled,
        }


