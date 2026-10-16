"""Evidence / RAG Guardrail Component.

Determines whether retrieved candidates contain genuine factual evidence to answer
the query, distinguishing between raw numerical cosine similarity and actual evidence.
"""

from __future__ import annotations

from typing import List, Union

from app.guardrails.models import EvidenceGuardrailResult
from app.vectorstore.qdrant_client import ScoredChunk


class EvidenceGuardrail:
    """Validates whether retrieved candidates provide sufficient evidence for grounded answering."""

    @classmethod
    def validate(
        cls,
        query: str,
        chunks: Union[List[ScoredChunk], List[str]],
        max_similarity: float,
        threshold: float,
        enabled: bool = True,
    ) -> EvidenceGuardrailResult:
        """Evaluate evidence sufficiency against guardrail rules.

        Args:
            query: The user query.
            chunks: Retrieved candidate chunks.
            max_similarity: Highest similarity score among candidates.
            threshold: Cosine similarity cutoff.
            enabled: If False, bypasses deep evidence verification (demonstrating unconstrained behavior).

        Returns:
            EvidenceGuardrailResult with sufficiency verdict, reason, and telemetry.
        """
        retrieved_count = len(chunks)
        passing_chunks = [
            c for c in chunks
            if (isinstance(c, ScoredChunk) and c.similarity_score >= threshold)
        ]
        passing_count = len(passing_chunks) if chunks and isinstance(chunks[0], ScoredChunk) else retrieved_count

        # When Guardrail is disabled (Demonstration Mode):
        # We simulate the unconstrained behavior: if any chunk meets threshold, it is treated as sufficient!
        if not enabled:
            if max_similarity >= threshold and retrieved_count > 0:
                return EvidenceGuardrailResult(
                    is_sufficient=True,
                    guardrail_triggered=False,
                    reason="Guardrail disabled (unconstrained candidate acceptance)",
                    token_coverage=1.0,
                    max_similarity=max_similarity,
                    retrieved_count=retrieved_count,
                    passing_count=passing_count,
                )
            return EvidenceGuardrailResult(
                is_sufficient=False,
                guardrail_triggered=True,
                reason="Similarity below threshold (Guardrail disabled)",
                token_coverage=0.0,
                max_similarity=max_similarity,
                retrieved_count=retrieved_count,
                passing_count=0,
            )

        # 1. Similarity threshold check
        if max_similarity < threshold or retrieved_count == 0 or passing_count == 0:
            return EvidenceGuardrailResult(
                is_sufficient=False,
                guardrail_triggered=True,
                reason="Insufficient document evidence (similarity below threshold)",
                token_coverage=0.0,
                max_similarity=max_similarity,
                retrieved_count=retrieved_count,
                passing_count=0,
            )

        # 2. Deep evidence evaluation (entity verification, target focus attribute, token coverage)
        from app.rag.evaluator import EvidenceDecision, EvidenceEvaluator
        eval_decision: EvidenceDecision = EvidenceEvaluator.evaluate(
            query=query,
            chunks=chunks,
            max_similarity=max_similarity,
            threshold=threshold,
        )

        if not eval_decision.has_relevant_evidence:
            return EvidenceGuardrailResult(
                is_sufficient=False,
                guardrail_triggered=True,
                reason=f"Insufficient document evidence: {eval_decision.reason}",
                token_coverage=eval_decision.token_coverage,
                max_similarity=max_similarity,
                retrieved_count=retrieved_count,
                passing_count=passing_count,
            )

        return EvidenceGuardrailResult(
            is_sufficient=True,
            guardrail_triggered=False,
            reason="Sufficient document evidence verified",
            token_coverage=eval_decision.token_coverage,
            max_similarity=max_similarity,
            retrieved_count=retrieved_count,
            passing_count=passing_count,
        )
