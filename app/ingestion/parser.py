"""PDF Ingestion and Text Extraction Module using PyMuPDF (fitz).

Extracts page-level text, counts pages, computes SHA-256 hashes,
and detects scanned or non-extractable PDF files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.logging_config import logger

# Attempt to import PyMuPDF (fitz)
try:
    import pymupdf as fitz  # type: ignore  # pymupdf >= 1.24 replaces legacy `fitz` import
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import fitz  # type: ignore  # fallback for older pymupdf installations
        PYMUPDF_AVAILABLE = True
    except ImportError:
        fitz = None
        PYMUPDF_AVAILABLE = False
        logger.warning("PyMuPDF (pymupdf) is not installed in the local environment.")


@dataclass
class PDFPage:
    """Represents extracted text and metadata for a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int
    word_count: int
    token_estimate: int

    @classmethod
    def from_text(cls, page_number: int, text: str) -> PDFPage:
        clean_text = text.strip()
        char_count = len(clean_text)
        words = clean_text.split()
        word_count = len(words)
        # Standard token heuristic: ~4 characters per token or 0.75 words per token
        token_estimate = max(1, char_count // 4) if char_count > 0 else 0
        return cls(
            page_number=page_number,
            text=clean_text,
            char_count=char_count,
            word_count=word_count,
            token_estimate=token_estimate,
        )


@dataclass
class ParsedPDF:
    """Complete parsed output and metadata for an ingested PDF document."""

    document_id: str
    filename: str
    filepath: str
    file_size_bytes: int
    file_hash: str
    page_count: int
    total_characters: int
    pages: List[PDFPage] = field(default_factory=list)
    is_extractable: bool = True
    error_message: Optional[str] = None


class PDFParser:
    """Parser responsible for extracting text and metadata from PDF files."""

    @staticmethod
    def calculate_hash(file_bytes_or_path: bytes | Path | str) -> str:
        """Calculate SHA-256 checksum for a file path or raw bytes."""
        hasher = hashlib.sha256()
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            hasher.update(file_bytes_or_path)
            return hasher.hexdigest()

        path = Path(file_bytes_or_path)
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def parse_file(cls, filepath: Path | str, filename: str, document_id: str) -> ParsedPDF:
        """Parse a PDF from the filesystem and extract page-level text."""
        path = Path(filepath)
        if not path.exists():
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=0,
                file_hash="",
                page_count=0,
                total_characters=0,
                is_extractable=False,
                error_message=f"File not found: {path}",
            )

        file_size = path.stat().st_size
        file_hash = cls.calculate_hash(path)

        # If file is source code, markdown, or plain text, parse directly
        text_extensions = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".sql", ".rst"}
        if path.suffix.lower() in text_extensions:
            return cls._parse_text_file(path, filename, document_id, file_size, file_hash)

        if not PYMUPDF_AVAILABLE:
            # Fallback for environments without PyMuPDF installed (e.g. lightweight unit test host)
            logger.warning("PyMuPDF unavailable: running basic fallback parser.")
            return cls._fallback_parse(path, filename, document_id, file_size, file_hash)

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.error(f"Failed to open PDF '{filename}': {exc}")
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=0,
                total_characters=0,
                is_extractable=False,
                error_message=f"Corrupted or unreadable PDF: {exc}",
            )

        pages: List[PDFPage] = []
        total_chars = 0

        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text") or ""
                page_data = PDFPage.from_text(page_number=page_idx + 1, text=text)
                pages.append(page_data)
                total_chars += page_data.char_count
        except Exception as exc:
            logger.error(f"Error during page extraction for '{filename}': {exc}")
            doc.close()
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=len(pages),
                total_characters=total_chars,
                pages=pages,
                is_extractable=False,
                error_message=f"Error extracting page text: {exc}",
            )
        finally:
            doc.close()

        # Validate extractability (e.g. scanned image PDF with no text layer)
        is_extractable = total_chars > 0
        error_msg = None if is_extractable else "PDF contains no extractable text (scanned image or empty)."

        if not is_extractable:
            logger.warning(f"Extracted 0 characters from '{filename}'. {error_msg}")

        return ParsedPDF(
            document_id=document_id,
            filename=filename,
            filepath=str(path),
            file_size_bytes=file_size,
            file_hash=file_hash,
            page_count=len(pages),
            total_characters=total_chars,
            pages=pages,
            is_extractable=is_extractable,
            error_message=error_msg,
        )

    @classmethod
    def _parse_text_file(
        cls, path: Path, filename: str, document_id: str, file_size: int, file_hash: str
    ) -> ParsedPDF:
        """Parse source code, markdown, or plain text file into paginated structure."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            if not raw_content.strip():
                return ParsedPDF(
                    document_id=document_id,
                    filename=filename,
                    filepath=str(path),
                    file_size_bytes=file_size,
                    file_hash=file_hash,
                    page_count=0,
                    total_characters=0,
                    is_extractable=False,
                    error_message="Empty file content",
                )

            lines = raw_content.splitlines(keepends=True)
            PAGE_LINES = 60
            pages: List[PDFPage] = []
            total_chars = 0
            for page_num, i in enumerate(range(0, max(1, len(lines)), PAGE_LINES), start=1):
                page_slice = lines[i : i + PAGE_LINES]
                page_text = "".join(page_slice).strip()
                if page_text:
                    p = PDFPage.from_text(page_number=page_num, text=page_text)
                    pages.append(p)
                    total_chars += p.char_count

            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=len(pages),
                total_characters=total_chars,
                pages=pages,
                is_extractable=total_chars > 0,
            )
        except Exception as exc:
            logger.error(f"Failed to parse text file '{filename}': {exc}")
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=0,
                total_characters=0,
                is_extractable=False,
                error_message=str(exc),
            )

    @classmethod
    def _fallback_parse(
        cls, path: Path, filename: str, document_id: str, file_size: int, file_hash: str
    ) -> ParsedPDF:
        """Lightweight fallback parser for environment bootstrap or mock testing."""
        # Check if file has mock text header or basic text content
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            text = " ".join(lines)
            page = PDFPage.from_text(page_number=1, text=text)
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=1,
                total_characters=page.char_count,
                pages=[page],
                is_extractable=page.char_count > 0,
                error_message=None if page.char_count > 0 else "Empty document content",
            )
        except Exception as exc:
            return ParsedPDF(
                document_id=document_id,
                filename=filename,
                filepath=str(path),
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=0,
                total_characters=0,
                is_extractable=False,
                error_message=str(exc),
            )
