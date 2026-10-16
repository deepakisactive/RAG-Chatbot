"""RAG Prompt Construction Module.

Separates prompt engineering from presentation logic. Enforces strict
factuality, grounding against retrieved context, and refusal to hallucinate.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from app.rag.models import SourceCitation
from app.vectorstore.qdrant_client import ScoredChunk

DEFAULT_SYSTEM_INSTRUCTION = """You are an expert AI assistant specialized in document question-answering and codebase analysis.
Your primary task is to answer the user's question accurately by synthesizing information from the provided reference context.

Guidelines:
1. When answering factual questions about indexed documents or code, base your answer strictly on the provided context excerpts and cite your sources using [DocumentName, Page X].
2. Do not copy excerpts verbatim or repeat excerpt headers; synthesize a clear, coherent, and factual explanation in your own words.
3. For code explanation, architecture, or bug analysis questions, explain the relevant files, functions, and logic provided in the context.
4. For programming or code generation tasks (e.g., writing unit tests, implementing functions, refactoring), provide accurate, clean Python code as requested, utilizing any relevant context where available.
5. If the question asks for specific factual information that is completely absent from the context and cannot be answered from the indexed documents, clearly state: "Information not found in the uploaded documents." (Insufficient evidence was found in the uploaded documents to answer this question.)
6. Keep your responses clear, structured, and technically accurate.
"""

LOW_CONFIDENCE_MESSAGE = "Information not found in the uploaded documents."
GENERAL_LLM_LABEL = "General LLM Answer"


class PromptBuilder:
    """Formats retrieved context and builds structured prompts for the LLM."""

    def __init__(self, system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION) -> None:
        self.system_instruction = system_instruction

    def format_context(self, chunks: List[ScoredChunk]) -> str:
        """Format a list of scored chunks into structured context blocks."""
        if not chunks:
            return "No context retrieved."

        context_blocks = []
        for rank, chunk in enumerate(chunks, start=1):
            block = (
                f"[Source #{rank}: {chunk.filename} | Page: {chunk.page}]\n"
                f"{chunk.text.strip()}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def extract_sources(self, chunks: List[ScoredChunk]) -> List[SourceCitation]:
        """Convert scored chunks to SourceCitation objects for UI display."""
        sources = []
        for chunk in chunks:
            # Create preview snippet
            snippet = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            sources.append(
                SourceCitation(
                    filename=chunk.filename,
                    document_id=chunk.document_id,
                    page=chunk.page,
                    similarity_score=chunk.similarity_score,
                    chunk_id=chunk.chunk_id,
                    snippet=snippet,
                )
            )
        return sources

    def build_prompt(
        self,
        query: str,
        chunks: List[ScoredChunk],
    ) -> Tuple[str, str, float]:
        """Assemble the complete prompt sent to the LLM and measure duration.

        Returns:
            Tuple of (full_prompt, formatted_context_str, duration_ms)
        """
        t_start = time.perf_counter()
        context_str = self.format_context(chunks)

        prompt = (
            f"{self.system_instruction}\n\n"
            f"=== CONTEXT EXCERPTS ===\n"
            f"{context_str}\n\n"
            f"=== USER QUESTION ===\n"
            f"{query.strip()}\n\n"
            f"=== ANSWER ==="
        )

        duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return prompt, context_str, duration_ms
