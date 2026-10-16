"""Recursive Text Chunker Module.

Chunks document pages while preserving strict page-level attribution,
token counts, and configurable overlaps.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ingestion.parser import ParsedPDF, PDFPage
from app.logging_config import logger


@dataclass
class TextChunk:
    """Represents a text chunk with strict document and page provenance."""

    chunk_id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    token_estimate: int
    pipeline_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for Qdrant payload serialization."""
        return asdict(self)


class RecursiveChunker:
    """Chunks text preserving paragraph and sentence boundaries with token overlap."""

    # Default separators ordered by semantic cohesion
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> None:
        # chunk_size is in approximate tokens (1 token ~= 4 characters)
        self.chunk_size_tokens = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap_tokens = chunk_overlap or settings.CHUNK_OVERLAP
        self.strategy = strategy or settings.CHUNKING_STRATEGY

        # Target character counts based on 4 chars / token heuristic
        self.target_char_size = max(64, self.chunk_size_tokens * 4)
        self.target_char_overlap = max(0, self.chunk_overlap_tokens * 4)

        if self.target_char_overlap >= self.target_char_size:
            raise ValueError(
                f"Chunk overlap ({self.chunk_overlap_tokens} tokens) must be smaller than chunk size ({self.chunk_size_tokens} tokens)."
            )

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first separator that breaks it into pieces."""
        text = text.strip()
        if len(text) <= self.target_char_size:
            return [text] if text else []

        if not separators:
            # Slicing fallback when no separators remain
            pieces = []
            for i in range(0, len(text), self.target_char_size - self.target_char_overlap):
                slice_part = text[i : i + self.target_char_size].strip()
                if slice_part:
                    pieces.append(slice_part)
            return pieces

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece) + (len(sep) if current_chunk else 0)
            if current_len + piece_len <= self.target_char_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    joined = sep.join(current_chunk).strip()
                    if joined:
                        chunks.append(joined)
                    
                    # Carry over overlap if possible
                    overlap_parts: List[str] = []
                    overlap_len = 0
                    for prev_part in reversed(current_chunk):
                        if overlap_len + len(prev_part) <= self.target_char_overlap:
                            overlap_parts.insert(0, prev_part)
                            overlap_len += len(prev_part)
                        else:
                            break
                    current_chunk = overlap_parts
                    current_len = overlap_len

                # If single piece is larger than target_char_size, recurse on it
                if len(piece) > self.target_char_size:
                    sub_pieces = self._split_text(piece, remaining_seps)
                    chunks.extend(sub_pieces)
                else:
                    current_chunk.append(piece)
                    current_len += len(piece)

        if current_chunk:
            final_joined = sep.join(current_chunk).strip()
            if final_joined:
                chunks.append(final_joined)

        return chunks

    def chunk_page(
        self,
        page: PDFPage,
        document_id: str,
        filename: str,
        pipeline_version: str = "v1",
        start_chunk_idx: int = 0,
    ) -> List[TextChunk]:
        """Chunk a single page into one or more TextChunk instances."""
        if not page.text:
            return []

        raw_chunks = self._split_text(page.text, self.DEFAULT_SEPARATORS)
        text_chunks: List[TextChunk] = []

        for idx, text in enumerate(raw_chunks):
            chunk_idx = start_chunk_idx + idx
            chunk_id = f"{document_id}_p{page.page_number}_c{chunk_idx:04d}"
            char_count = len(text)
            token_est = max(1, char_count // 4)

            text_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page=page.page_number,
                    chunk_index=chunk_idx,
                    text=text,
                    char_count=char_count,
                    token_estimate=token_est,
                    pipeline_version=pipeline_version,
                )
            )

        return text_chunks

    def chunk_document(
        self,
        parsed_pdf: ParsedPDF,
        pipeline_version: str = "v1",
    ) -> List[TextChunk]:
        """Chunk an entire parsed PDF document page by page."""
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in parsed_pdf.pages:
            page_chunks = self.chunk_page(
                page=page,
                document_id=parsed_pdf.document_id,
                filename=parsed_pdf.filename,
                pipeline_version=pipeline_version,
                start_chunk_idx=global_chunk_idx,
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        logger.info(
            f"Chunked document '{parsed_pdf.filename}' ({len(parsed_pdf.pages)} pages) -> {len(all_chunks)} chunks."
        )
        return all_chunks
