"""RAG core package: models, retrieval, prompt formatting, and pipeline orchestration."""

from app.rag.models import ComparisonResponse, RAGResponse, RetrievalTrace, SourceCitation
from app.rag.pipeline import RAGPipeline, get_pipeline
from app.rag.prompt import GENERAL_LLM_LABEL, LOW_CONFIDENCE_MESSAGE, PromptBuilder
from app.rag.retriever import RetrievalEngine

__all__ = [
    "ComparisonResponse",
    "RAGResponse",
    "RetrievalTrace",
    "SourceCitation",
    "RetrievalEngine",
    "PromptBuilder",
    "LOW_CONFIDENCE_MESSAGE",
    "GENERAL_LLM_LABEL",
    "RAGPipeline",
    "get_pipeline",
]
