"""Output Guardrail Component.

Treats all LLM generated output as untrusted.
Validates and sanitizes responses before presentation to the user:
1. Rejects empty outputs.
2. Rejects extremely short / malformed outputs.
3. Detects and safely strips known model special tokens (<end_of_text>, <|endoftext|>, etc.).
4. Detects and cleans prompt leakage (System:, Context:, internal prompt instructions).
5. Enforces grounding consistency: prevents unsupported responses from being labeled as document-grounded.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from app.guardrails.models import OutputGuardrailResult

OUTPUT_VALIDATION_FAILED_MESSAGE = "The generated response could not be validated against the application's requirements."

KNOWN_SPECIAL_TOKENS = [
    "<end_of_text>",
    "<|endoftext|>",
    "<|end_of_answer|>",
    "<|im_end|>",
    "<|im_start|>",
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
]

PROMPT_LEAKAGE_PATTERNS = [
    r"^(?:System|SYSTEM|CONTEXT|Context|User|USER|Instruction|Answer):\s*",
    r"Provide a more detailed answer:\s*",
    r"Provide a concise synthesized summary answer:\s*",
    r"Provide a concise synthesized answer:\s*",
    r"Answer the user's question using the provided context\.\s*",
    r"Do not copy the context unnecessarily\.\s*",
    r"You are a helpful question-answering assistant\.\s*",
    r"Give a direct and concise answer\.\s*",
]


class OutputGuardrail:
    """Validates and sanitizes LLM-generated output before presentation."""

    @classmethod
    def sanitize_special_tokens(cls, text: str) -> Tuple[str, List[str]]:
        """Safely remove known special tokens without modifying legitimate content."""
        stripped = []
        cleaned = text
        for token in KNOWN_SPECIAL_TOKENS:
            if token in cleaned:
                stripped.append(token)
                cleaned = cleaned.replace(token, "")
        return cleaned, stripped

    @classmethod
    def sanitize_prompt_leakage(cls, text: str) -> Tuple[str, List[str]]:
        """Remove identified prompt instructions and role markers from output."""
        leaked = []
        cleaned = text
        for pattern in PROMPT_LEAKAGE_PATTERNS:
            if re.search(pattern, cleaned, re.MULTILINE | re.IGNORECASE):
                leaked.append(pattern)
                cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

        # Clean any multiline "Answer:" prefixes and empty lines
        lines = [line.strip() for line in cleaned.splitlines()]
        filtered_lines = [
            line for line in lines
            if line and not re.match(r"^(?:System|Context|User|Instruction|Answer):\s*$", line, re.IGNORECASE)
        ]
        cleaned = "\n".join(filtered_lines)
        return cleaned.strip(), leaked

    @classmethod
    def validate(
        cls,
        raw_output: str,
        is_grounded_mode: bool = False,
        has_evidence: bool = True,
        enabled: bool = True,
    ) -> OutputGuardrailResult:
        """Validate generated LLM output.

        Args:
            raw_output: The text produced by Ollama.
            is_grounded_mode: Whether the answering strategy requires strict document grounding.
            has_evidence: Whether genuine document evidence was retrieved.
            enabled: If False, bypasses output guardrail (demonstrating unconstrained behavior with tokens).

        Returns:
            OutputGuardrailResult with validation verdict, sanitized text, and failure reason if any.
        """
        # Demonstration bypass: returns raw output with tokens and prompt leaks intact
        if not enabled:
            return OutputGuardrailResult(
                is_valid=True,
                sanitized_output=raw_output or "",
                guardrail_triggered=False,
                reason="Guardrail disabled (raw untrusted output permitted)",
                stripped_tokens=[],
                leaked_prompts_detected=[],
                error_message=None,
            )

        # 1. Reject empty output
        if not raw_output or not raw_output.strip():
            return OutputGuardrailResult(
                is_valid=False,
                sanitized_output="",
                guardrail_triggered=True,
                reason="Empty LLM output",
                stripped_tokens=[],
                leaked_prompts_detected=[],
                error_message=OUTPUT_VALIDATION_FAILED_MESSAGE,
            )

        # 2. Detect & safely strip special tokens
        cleaned_tokens, stripped_tokens = cls.sanitize_special_tokens(raw_output)

        # 3. Detect & strip prompt leakage
        cleaned_text, leaked_prompts = cls.sanitize_prompt_leakage(cleaned_tokens)

        # 4. Check if output was purely special tokens or prompt leakage
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return OutputGuardrailResult(
                is_valid=False,
                sanitized_output="",
                guardrail_triggered=True,
                reason="Output contained only special tokens or leaked instructions",
                stripped_tokens=stripped_tokens,
                leaked_prompts_detected=leaked_prompts,
                error_message=OUTPUT_VALIDATION_FAILED_MESSAGE,
            )

        # 5. Extremely short / malformed output check (e.g. single character or punctuation only)
        if len(cleaned_text) < 2 or not re.search(r"[a-zA-Z0-9]", cleaned_text):
            return OutputGuardrailResult(
                is_valid=False,
                sanitized_output="",
                guardrail_triggered=True,
                reason="Extremely short or malformed output",
                stripped_tokens=stripped_tokens,
                leaked_prompts_detected=leaked_prompts,
                error_message=OUTPUT_VALIDATION_FAILED_MESSAGE,
            )

        # 6. Grounding consistency: Grounded RAG requires evidence
        if is_grounded_mode and not has_evidence:
            return OutputGuardrailResult(
                is_valid=False,
                sanitized_output="",
                guardrail_triggered=True,
                reason="Unsupported response cannot be presented as document-grounded",
                stripped_tokens=stripped_tokens,
                leaked_prompts_detected=leaked_prompts,
                error_message=OUTPUT_VALIDATION_FAILED_MESSAGE,
            )

        # Successfully validated
        return OutputGuardrailResult(
            is_valid=True,
            sanitized_output=cleaned_text,
            guardrail_triggered=bool(stripped_tokens or leaked_prompts),
            reason=(
                f"Output validated (sanitized tokens: {stripped_tokens})"
                if stripped_tokens
                else "Output validation passed"
            ),
            stripped_tokens=stripped_tokens,
            leaked_prompts_detected=leaked_prompts,
            error_message=None,
        )
