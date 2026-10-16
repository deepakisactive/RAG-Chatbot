"""Ingestion package: parsing, chunking, embedding, and indexing."""

from app.ingestion.chunker import RecursiveChunker, TextChunk
from app.ingestion.embedder import (
    BaseEmbedder,
    FastEmbedEmbedder,
    MockEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.parser import PDFPage, ParsedPDF, PDFParser

__all__ = [
    "PDFPage",
    "ParsedPDF",
    "PDFParser",
    "RecursiveChunker",
    "TextChunk",
    "BaseEmbedder",
    "FastEmbedEmbedder",
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    "get_embedder",
    "DocumentIndexer",
]
