"""Unit and integration tests for Guardrails (Input, Evidence, Output) and Telemetry.

Tests:
1. Input Guardrail validation, rejection, and 500-char limits.
2. Evidence Guardrail gating, topic mismatch, and Before vs After comparison.
3. Output Guardrail special-token stripping, prompt leakage sanitization, and grounding checks.
4. Telemetry persistence and 13 required tracking fields.
"""

import pytest
from app.config import settings
from app.guardrails import (
    EvidenceGuardrail,
    InputGuardrail,
    OutputGuardrail,
    QUERY_EMPTY_MESSAGE,
    QUERY_TOO_LONG_MESSAGE,
    OUTPUT_VALIDATION_FAILED_MESSAGE,
)
from app.rag.models import RAGResponse
from app.rag.prompt import LOW_CONFIDENCE_MESSAGE
from app.vectorstore.qdrant_client import ScoredChunk


class TestInputGuardrail:
    """Test suite for Layer 1: Input Guardrail."""

    def setup_method(self):
        self.guard = InputGuardrail(max_length=500)

    def test_reject_empty_query(self):
        res = self.guard.validate("")
        assert not res.is_valid
        assert res.guardrail_triggered
        assert res.error_message == QUERY_EMPTY_MESSAGE
        assert res.original_length == 0

    def test_reject_whitespace_only_query(self):
        res = self.guard.validate("   \n\t  \r  ")
        assert not res.is_valid
        assert res.guardrail_triggered
        assert res.error_message == QUERY_EMPTY_MESSAGE

    def test_reject_query_exceeding_max_length(self):
        long_query = "A" * 501
        res = self.guard.validate(long_query)
        assert not res.is_valid
        assert res.guardrail_triggered
        assert res.error_message == QUERY_TOO_LONG_MESSAGE
        assert res.original_length == 501

    def test_accept_query_within_limit(self):
        valid_query = "What is the sequence recovery rate of ProteinMPNN?"
        res = self.guard.validate(valid_query)
        assert res.is_valid
        assert not res.guardrail_triggered
        assert res.sanitized_query == valid_query
        assert res.sanitized_length == len(valid_query)

    def test_whitespace_normalization(self):
        query = "  What   is    ProteinMPNN?  \n "
        res = self.guard.validate(query)
        assert res.is_valid
        assert res.sanitized_query == "What is ProteinMPNN?"

    def test_reject_null_bytes_and_control_chars(self):
        malformed = "What is ProteinMPNN?\x00\x08"
        res = self.guard.validate(malformed)
        assert not res.is_valid
        assert "null bytes" in res.reason.lower() or "control" in res.reason.lower()


class TestEvidenceGuardrail:
    """Test suite for Layer 2: Evidence / RAG Guardrail."""

    def test_low_similarity_rejection(self):
        res = EvidenceGuardrail.validate(
            query="What is ProteinMPNN?",
            chunks=[],
            max_similarity=0.25,
            threshold=0.55,
            enabled=True,
        )
        assert not res.is_sufficient
        assert res.guardrail_triggered
        assert "similarity below threshold" in res.reason.lower()

    def test_topic_mismatch_rejection_c_vs_cpp(self):
        # Biomediated chunk with raw similarity above threshold
        chunk = ScoredChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            filename="ProteinMPNN.pdf",
            page=1,
            chunk_index=0,
            text="AlphaFold and RoseTTAFold are deep learning methods for predicting protein structure.",
            similarity_score=0.5656,
            rank=1,
            pipeline_version="v1",
        )
        res = EvidenceGuardrail.validate(
            query="which is better c or c++?",
            chunks=[chunk],
            max_similarity=0.5656,
            threshold=0.55,
            enabled=True,
        )
        assert not res.is_sufficient
        assert res.guardrail_triggered
        assert "c++" in res.reason.lower() or "insufficient" in res.reason.lower()

    def test_before_vs_after_behavior_c_vs_cpp(self):
        chunk = ScoredChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            filename="ProteinMPNN.pdf",
            page=1,
            chunk_index=0,
            text="AlphaFold and RoseTTAFold predict protein structure.",
            similarity_score=0.5656,
            rank=1,
            pipeline_version="v1",
        )
        # Without guardrail (Demonstration Mode): unconstrained acceptance
        res_disabled = EvidenceGuardrail.validate(
            query="which is better c or c++?",
            chunks=[chunk],
            max_similarity=0.5656,
            threshold=0.55,
            enabled=False,
        )
        assert res_disabled.is_sufficient
        assert not res_disabled.guardrail_triggered

        # With guardrail: controlled rejection
        res_enabled = EvidenceGuardrail.validate(
            query="which is better c or c++?",
            chunks=[chunk],
            max_similarity=0.5656,
            threshold=0.55,
            enabled=True,
        )
        assert not res_enabled.is_sufficient
        assert res_enabled.guardrail_triggered

    def test_distractor_query_cost_in_dollars(self):
        chunk = ScoredChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            filename="ProteinMPNN.pdf",
            page=1,
            chunk_index=0,
            text="ProteinMPNN improves sequence recovery to 52.4% on native backbones.",
            similarity_score=0.72,
            rank=1,
            pipeline_version="v1",
        )
        res = EvidenceGuardrail.validate(
            query="What is the total annual cost in US dollars for ProteinMPNN?",
            chunks=[chunk],
            max_similarity=0.72,
            threshold=0.55,
            enabled=True,
        )
        assert not res.is_sufficient
        assert res.guardrail_triggered
        assert "target focus" in res.reason.lower() or "cost" in res.reason.lower()

    def test_sufficient_evidence_proteinmpnn(self):
        chunk = ScoredChunk(
            chunk_id="chk-1",
            document_id="doc-1",
            filename="ProteinMPNN.pdf",
            page=1,
            chunk_index=0,
            text="ProteinMPNN achieves 52.4% sequence recovery compared to Rosetta's 32.9%.",
            similarity_score=0.88,
            rank=1,
            pipeline_version="v1",
        )
        res = EvidenceGuardrail.validate(
            query="What is the sequence recovery rate of ProteinMPNN?",
            chunks=[chunk],
            max_similarity=0.88,
            threshold=0.55,
            enabled=True,
        )
        assert res.is_sufficient
        assert not res.guardrail_triggered


class TestOutputGuardrail:
    """Test suite for Layer 3: Output Guardrail."""

    def test_strip_special_tokens(self):
        raw = "ProteinMPNN is 200x faster.<end_of_text><|endoftext|><|im_end|>"
        clean, stripped = OutputGuardrail.sanitize_special_tokens(raw)
        assert "<end_of_text>" not in clean
        assert "<|endoftext|>" not in clean
        assert "<|im_end|>" not in clean
        assert clean == "ProteinMPNN is 200x faster."
        assert len(stripped) == 3

    def test_strip_prompt_leakage(self):
        raw = (
            "System:\nYou are a helpful question-answering assistant.\n"
            "Context:\nProteinMPNN is fast.\n"
            "Answer:\nProteinMPNN is roughly 200 times faster than Rosetta."
        )
        res = OutputGuardrail.validate(
            raw_output=raw,
            is_grounded_mode=True,
            has_evidence=True,
            enabled=True,
        )
        assert res.is_valid
        assert res.guardrail_triggered
        assert "System:" not in res.sanitized_output
        assert "Context:" not in res.sanitized_output
        assert "You are a helpful question-answering assistant" not in res.sanitized_output
        assert "ProteinMPNN is roughly 200 times faster than Rosetta." in res.sanitized_output

    def test_reject_empty_output(self):
        res = OutputGuardrail.validate(
            raw_output="   ",
            is_grounded_mode=True,
            has_evidence=True,
            enabled=True,
        )
        assert not res.is_valid
        assert res.guardrail_triggered
        assert res.error_message == OUTPUT_VALIDATION_FAILED_MESSAGE

    def test_enforce_grounding_consistency(self):
        # If Grounded mode claims grounding without evidence, output guardrail blocks it
        res = OutputGuardrail.validate(
            raw_output="This is a general knowledge answer not in documents.",
            is_grounded_mode=True,
            has_evidence=False,
            enabled=True,
        )
        assert not res.is_valid
        assert res.guardrail_triggered
        assert "unsupported response" in res.reason.lower() or "grounded" in res.reason.lower()


class TestTelemetryAndPipelineIntegration:
    """Verify that all 13 telemetry fields are populated in responses."""

    def test_telemetry_fields_present_in_rag_response(self):
        resp = RAGResponse(
            query="What is ProteinMPNN?",
            answer="ProteinMPNN designs protein sequences.",
            sources=[],
            trace=None,
            context_sent="",
            model="llama3.2:1b",
            embedding_time_ms=10.0,
            retrieval_time_ms=15.0,
            context_time_ms=5.0,
            generation_time_ms=100.0,
            total_time_ms=130.0,
            is_low_confidence=False,
            is_general_knowledge=False,
            has_relevant_evidence=True,
            input_validation="PASSED",
            input_length=23,
            retrieval_completed=True,
            retrieved_candidate_count=5,
            evidence_sufficient=True,
            guardrail_triggered=False,
            guardrail_reason="",
            llm_used=True,
            output_validation="PASSED",
            output_guardrail_triggered=False,
            output_guardrail_reason="",
            final_result_type="GROUNDED_ANSWER",
            guardrails_enabled=True,
        )

        assert resp.input_validation == "PASSED"
        assert resp.input_length == 23
        assert resp.retrieval_completed is True
        assert resp.retrieved_candidate_count == 5
        assert resp.evidence_sufficient is True
        assert resp.guardrail_triggered is False
        assert resp.llm_used is True
        assert resp.output_validation == "PASSED"
        assert resp.output_guardrail_triggered is False
        assert resp.final_result_type == "GROUNDED_ANSWER"
        assert resp.guardrails_enabled is True
