"""Input Guardrail Component.

Inspects incoming user queries before embedding, Qdrant retrieval, or LLM generation.
Enforces:
1. Rejection of empty or whitespace-only queries.
2. Rejection of queries exceeding max character limit (default 500 chars).
3. Whitespace trimming and normalization.
4. Rejection of obviously malformed input (e.g. pure control characters, null bytes).
"""

from __future__ import annotations

import re
from typing import Optional

from app.config import settings
from app.guardrails.models import InputGuardrailResult

QUERY_TOO_LONG_MESSAGE = "Query too long. Please shorten your question to 500 characters or fewer."
QUERY_EMPTY_MESSAGE = "Query is empty. Please provide a valid question."
QUERY_MALFORMED_MESSAGE = "Malformed query. Please provide valid text."


class InputGuardrail:
    """Validates and sanitizes incoming user queries before pipeline execution."""

    def __init__(self, max_length: Optional[int] = None) -> None:
        self.max_length = max_length or getattr(settings, "MAX_QUERY_LENGTH", 500)

    def validate(self, query: str, enabled: bool = True) -> InputGuardrailResult:
        """Validate query text against input guardrail rules.

        Args:
            query: The raw user query.
            enabled: If False, bypasses guardrail checks (for Before/After demonstration).

        Returns:
            InputGuardrailResult containing validation status, lengths, and reasons.
        """
        if query is None:
            query = ""

        raw_length = len(query)

        # Demonstration bypass
        if not enabled:
            trimmed = query.strip()
            return InputGuardrailResult(
                is_valid=True,
                sanitized_query=trimmed,
                original_length=raw_length,
                sanitized_length=len(trimmed),
                guardrail_triggered=False,
                reason="Guardrail disabled (Demonstration Mode)",
                error_message=None,
            )

        # 1. Reject empty queries
        if not query or not query.strip():
            return InputGuardrailResult(
                is_valid=False,
                sanitized_query="",
                original_length=raw_length,
                sanitized_length=0,
                guardrail_triggered=True,
                reason="Empty query",
                error_message=QUERY_EMPTY_MESSAGE,
            )

        # 2. Reject queries exceeding max character length
        if raw_length > self.max_length:
            return InputGuardrailResult(
                is_valid=False,
                sanitized_query=query,
                original_length=raw_length,
                sanitized_length=raw_length,
                guardrail_triggered=True,
                reason=f"Query length ({raw_length}) exceeds maximum limit ({self.max_length})",
                error_message=QUERY_TOO_LONG_MESSAGE,
            )

        # 3. Check for obviously malformed input (pure control chars or null bytes)
        if "\x00" in query:
            return InputGuardrailResult(
                is_valid=False,
                sanitized_query="",
                original_length=raw_length,
                sanitized_length=0,
                guardrail_triggered=True,
                reason="Malformed query contains null bytes",
                error_message=QUERY_MALFORMED_MESSAGE,
            )

        # Check for queries that contain no alphanumeric or standard punctuation characters
        printable_content = re.sub(r"[\s\x00-\x1f\x7f-\x9f]", "", query)
        if not printable_content:
            return InputGuardrailResult(
                is_valid=False,
                sanitized_query="",
                original_length=raw_length,
                sanitized_length=0,
                guardrail_triggered=True,
                reason="Malformed query contains only unprintable/control characters",
                error_message=QUERY_MALFORMED_MESSAGE,
            )

        # 4. Trim unnecessary whitespace
        sanitized = re.sub(r"\s+", " ", query).strip()

        return InputGuardrailResult(
            is_valid=True,
            sanitized_query=sanitized,
            original_length=raw_length,
            sanitized_length=len(sanitized),
            guardrail_triggered=False,
            reason="Input validation passed",
            error_message=None,
        )
