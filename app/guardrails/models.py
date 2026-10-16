"""Guardrail data structures and validation result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InputGuardrailResult:
    """Result of input guardrail inspection."""

    is_valid: bool
    sanitized_query: str
    original_length: int
    sanitized_length: int
    guardrail_triggered: bool
    reason: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGuardrailResult:
    """Result of evidence sufficiency guardrail inspection."""

    is_sufficient: bool
    guardrail_triggered: bool
    reason: str
    token_coverage: float
    max_similarity: float
    retrieved_count: int
    passing_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputGuardrailResult:
    """Result of LLM output guardrail inspection."""

    is_valid: bool
    sanitized_output: str
    guardrail_triggered: bool
    reason: str
    stripped_tokens: List[str] = field(default_factory=list)
    leaked_prompts_detected: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullGuardrailReport:
    """Consolidated telemetry report from all 3 guardrail layers."""

    input_validation: str  # "PASSED" | "FAILED"
    input_length: int
    retrieval_completed: bool
    retrieved_candidate_count: int
    max_similarity: float
    evidence_sufficient: bool
    guardrail_triggered: bool
    guardrail_reason: str
    llm_used: bool
    output_validation: str  # "PASSED" | "FAILED" | "NOT_APPLICABLE"
    output_guardrail_triggered: bool
    output_guardrail_reason: str
    final_result_type: str  # "GROUNDED_ANSWER" | "GENERAL_LLM_ANSWER" | "REFUSAL" | "BLOCKED_INPUT" | "BLOCKED_OUTPUT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
