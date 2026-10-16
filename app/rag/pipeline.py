"""End-to-End RAG Pipeline Orchestrator.

Integrates semantic retrieval, confidence gating, prompt construction,
Ollama LLM generation, real execution timing, and query logging.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

from app.config import settings
from app.database.database import DatabaseManager, get_db
from app.database.models import QueryRecord
from app.database.repository import QueryRepository
from app.guardrails import (
    EvidenceGuardrail,
    InputGuardrail,
    OutputGuardrail,
    OUTPUT_VALIDATION_FAILED_MESSAGE,
    QUERY_EMPTY_MESSAGE,
    QUERY_TOO_LONG_MESSAGE,
)
from app.llm.ollama_client import OllamaClient, get_ollama_client
from app.logging_config import logger, timed_operation
from app.rag.evaluator import EvidenceDecision, EvidenceEvaluator
from app.rag.models import ComparisonResponse, RAGResponse, RetrievalTrace, SourceCitation
from app.rag.prompt import GENERAL_LLM_LABEL, LOW_CONFIDENCE_MESSAGE, PromptBuilder
from app.rag.retriever import RetrievalEngine


class RAGPipeline:
    """Complete, transparent RAG execution pipeline."""

    def __init__(
        self,
        retriever: Optional[RetrievalEngine] = None,
        ollama_client: Optional[OllamaClient] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        db_manager: Optional[DatabaseManager] = None,
        input_guardrail: Optional[InputGuardrail] = None,
        evidence_guardrail: Optional[EvidenceGuardrail] = None,
        output_guardrail: Optional[OutputGuardrail] = None,
    ) -> None:
        self.retriever = retriever or RetrievalEngine()
        self.ollama_client = ollama_client or get_ollama_client()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.db = db_manager or get_db()
        self.query_repo = QueryRepository(db=self.db)
        self.input_guardrail = input_guardrail or InputGuardrail(max_length=settings.MAX_QUERY_LENGTH)
        self.evidence_guardrail = evidence_guardrail or EvidenceGuardrail()
        self.output_guardrail = output_guardrail or OutputGuardrail()

    def _generate_general_llm(
        self,
        query: str,
        model: str,
    ) -> Tuple[str, float]:
        """Query LLM directly using its general knowledge without RAG document context."""
        general_system = (
            "You are a helpful and knowledgeable AI assistant. "
            "Answer the user's question directly, accurately, and clearly using your general knowledge."
        )
        messages = [
            {"role": "system", "content": general_system},
            {"role": "user", "content": f"Question: {query}"},
        ]
        try:
            if hasattr(self.ollama_client, "chat"):
                return self.ollama_client.chat(
                    messages=messages,
                    model=model,
                )
            return self.ollama_client.generate(
                prompt=query,
                system=general_system,
                model=model,
            )
        except Exception as exc:
            logger.error(f"Error during general LLM generation: {exc}")
            return f"Error communicating with local LLM on Windows host: {exc}", 0.0

    def _execute_grounded_strategy(
        self,
        clean_query: str,
        trace: RetrievalTrace,
        active_model: str,
        t_start: float,
        has_relevant_evidence: bool = True,
        guardrail_reason: str = "",
        guardrails_enabled: bool = True,
        input_length: int = 0,
    ) -> Tuple[RAGResponse, bool]:
        """Execute Grounded RAG answering strategy using provided retrieval trace."""
        is_low_conf = trace.is_low_confidence or not trace.filtered_chunks or not has_relevant_evidence
        if is_low_conf:
            total_duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            resp = RAGResponse(
                query=clean_query,
                answer=LOW_CONFIDENCE_MESSAGE,
                sources=[],
                trace=trace,
                context_sent="",
                model=active_model,
                embedding_time_ms=trace.embedding_time_ms,
                retrieval_time_ms=trace.retrieval_time_ms,
                context_time_ms=0.0,
                generation_time_ms=0.0,
                total_time_ms=total_duration_ms,
                is_low_confidence=True,
                is_general_knowledge=False,
                has_relevant_evidence=False,
                input_validation="PASSED",
                input_length=input_length or len(clean_query),
                retrieval_completed=True,
                retrieved_candidate_count=len(trace.filtered_chunks),
                evidence_sufficient=False,
                guardrail_triggered=True,
                guardrail_reason=guardrail_reason or "Insufficient document evidence",
                llm_used=False,
                output_validation="NOT_APPLICABLE",
                output_guardrail_triggered=False,
                output_guardrail_reason="",
                final_result_type="REFUSAL",
                guardrails_enabled=guardrails_enabled,
            )
            return resp, False

        prompt, context_str, context_time_ms = self.prompt_builder.build_prompt(
            query=clean_query,
            chunks=trace.filtered_chunks,
        )
        sources = self.prompt_builder.extract_sources(trace.filtered_chunks)

        system_instruction = (
            "You are a helpful question-answering assistant. "
            "Answer the user's question using the provided context. "
            "Do not copy the context unnecessarily. Give a direct and concise answer."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {clean_query}",
            },
        ]

        try:
            if hasattr(self.ollama_client, "chat"):
                raw_answer, gen_ms = self.ollama_client.chat(
                    messages=messages,
                    model=active_model,
                )
            else:
                raw_answer, gen_ms = self.ollama_client.generate(
                    prompt=prompt,
                    model=active_model,
                )
        except Exception as exc:
            logger.error(f"Error during Grounded LLM generation: {exc}")
            raw_answer = f"Error communicating with local LLM on Windows host: {exc}"
            gen_ms = 0.0

        # Validate with Output Guardrail
        out_res = self.output_guardrail.validate(
            raw_output=raw_answer,
            is_grounded_mode=True,
            has_evidence=True,
            enabled=guardrails_enabled,
        )

        if not out_res.is_valid:
            final_answer = out_res.error_message or OUTPUT_VALIDATION_FAILED_MESSAGE
            out_validation = "FAILED"
            out_triggered = True
            out_reason = out_res.reason
            final_res_type = "BLOCKED_OUTPUT"
            guard_trig = True
            guard_r = out_res.reason
        else:
            final_answer = out_res.sanitized_output
            out_validation = "PASSED"
            out_triggered = out_res.guardrail_triggered
            out_reason = out_res.reason
            final_res_type = "GROUNDED_ANSWER"
            guard_trig = False
            guard_r = "All guardrails passed"

        total_duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        resp = RAGResponse(
            query=clean_query,
            answer=final_answer,
            sources=sources,
            trace=trace,
            context_sent=context_str,
            model=active_model,
            embedding_time_ms=trace.embedding_time_ms,
            retrieval_time_ms=trace.retrieval_time_ms,
            context_time_ms=context_time_ms,
            generation_time_ms=gen_ms,
            total_time_ms=total_duration_ms,
            is_low_confidence=False,
            is_general_knowledge=False,
            has_relevant_evidence=True,
            input_validation="PASSED",
            input_length=input_length or len(clean_query),
            retrieval_completed=True,
            retrieved_candidate_count=len(trace.filtered_chunks),
            evidence_sufficient=True,
            guardrail_triggered=guard_trig,
            guardrail_reason=guard_r,
            llm_used=True,
            output_validation=out_validation,
            output_guardrail_triggered=out_triggered,
            output_guardrail_reason=out_reason,
            final_result_type=final_res_type,
            guardrails_enabled=guardrails_enabled,
        )
        return resp, True

    def _execute_rag_llm_strategy(
        self,
        clean_query: str,
        trace: RetrievalTrace,
        active_model: str,
        t_start: float,
        has_relevant_evidence: bool = True,
        guardrail_reason: str = "",
        guardrails_enabled: bool = True,
        input_length: int = 0,
    ) -> Tuple[RAGResponse, bool]:
        """Execute RAG + LLM hybrid answering strategy using provided retrieval trace."""
        is_low_conf = trace.is_low_confidence or not trace.filtered_chunks or not has_relevant_evidence
        if is_low_conf:
            raw_answer, gen_ms = self._generate_general_llm(clean_query, active_model)
            out_res = self.output_guardrail.validate(
                raw_output=raw_answer,
                is_grounded_mode=False,
                has_evidence=False,
                enabled=guardrails_enabled,
            )
            if not out_res.is_valid:
                final_answer = out_res.error_message or OUTPUT_VALIDATION_FAILED_MESSAGE
                out_validation = "FAILED"
                out_triggered = True
                out_reason = out_res.reason
                final_res_type = "BLOCKED_OUTPUT"
            else:
                final_answer = out_res.sanitized_output
                out_validation = "PASSED"
                out_triggered = out_res.guardrail_triggered
                out_reason = out_res.reason
                final_res_type = "GENERAL_LLM_ANSWER"

            total_duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            resp = RAGResponse(
                query=clean_query,
                answer=final_answer,
                sources=[],
                trace=trace,
                context_sent="",
                model=active_model,
                embedding_time_ms=trace.embedding_time_ms,
                retrieval_time_ms=trace.retrieval_time_ms,
                context_time_ms=0.0,
                generation_time_ms=gen_ms,
                total_time_ms=total_duration_ms,
                is_low_confidence=True,
                is_general_knowledge=True,
                has_relevant_evidence=False,
                input_validation="PASSED",
                input_length=input_length or len(clean_query),
                retrieval_completed=True,
                retrieved_candidate_count=len(trace.filtered_chunks),
                evidence_sufficient=False,
                guardrail_triggered=True,
                guardrail_reason=guardrail_reason or "Insufficient document evidence (general fallback used)",
                llm_used=True,
                output_validation=out_validation,
                output_guardrail_triggered=out_triggered,
                output_guardrail_reason=out_reason,
                final_result_type=final_res_type,
                guardrails_enabled=guardrails_enabled,
            )
            return resp, True

        prompt, context_str, context_time_ms = self.prompt_builder.build_prompt(
            query=clean_query,
            chunks=trace.filtered_chunks,
        )
        sources = self.prompt_builder.extract_sources(trace.filtered_chunks)

        system_instruction = (
            "You are a helpful question-answering assistant. "
            "Answer the user's question using the provided context. "
            "Do not copy the context unnecessarily. Give a direct and concise answer."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {clean_query}",
            },
        ]

        try:
            if hasattr(self.ollama_client, "chat"):
                raw_answer, gen_ms = self.ollama_client.chat(
                    messages=messages,
                    model=active_model,
                )
            else:
                raw_answer, gen_ms = self.ollama_client.generate(
                    prompt=prompt,
                    model=active_model,
                )
        except Exception as exc:
            logger.error(f"Error during RAG+LLM generation: {exc}")
            raw_answer = f"Error communicating with local LLM on Windows host: {exc}"
            gen_ms = 0.0

        is_general_knowledge = False
        refusal_indicators = [
            "no mention",
            "not mentioned",
            "not found in",
            "insufficient evidence",
            "cannot be answered from",
            "does not contain",
            "does not mention",
            "is not mentioned",
            "not provided in the",
            "no information provided",
            "unable to find",
        ]
        if any(ind in raw_answer.lower() for ind in refusal_indicators):
            logger.info(
                f"Grounded answer indicated missing info: '{raw_answer[:60]}...'. Falling back to general LLM."
            )
            fallback_ans, fallback_ms = self._generate_general_llm(clean_query, active_model)
            if fallback_ans and len(fallback_ans.strip()) > 0:
                raw_answer = fallback_ans
                gen_ms += fallback_ms
                is_general_knowledge = True
                sources = []
                context_str = ""

        out_res = self.output_guardrail.validate(
            raw_output=raw_answer,
            is_grounded_mode=not is_general_knowledge,
            has_evidence=has_relevant_evidence,
            enabled=guardrails_enabled,
        )

        if not out_res.is_valid:
            final_answer = out_res.error_message or OUTPUT_VALIDATION_FAILED_MESSAGE
            out_validation = "FAILED"
            out_triggered = True
            out_reason = out_res.reason
            final_res_type = "BLOCKED_OUTPUT"
        else:
            final_answer = out_res.sanitized_output
            out_validation = "PASSED"
            out_triggered = out_res.guardrail_triggered
            out_reason = out_res.reason
            final_res_type = "GENERAL_LLM_ANSWER" if is_general_knowledge else "GROUNDED_ANSWER"

        total_duration_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        resp = RAGResponse(
            query=clean_query,
            answer=final_answer,
            sources=sources,
            trace=trace,
            context_sent=context_str,
            model=active_model,
            embedding_time_ms=trace.embedding_time_ms,
            retrieval_time_ms=trace.retrieval_time_ms,
            context_time_ms=context_time_ms,
            generation_time_ms=gen_ms,
            total_time_ms=total_duration_ms,
            is_low_confidence=is_low_conf,
            is_general_knowledge=is_general_knowledge,
            has_relevant_evidence=not is_general_knowledge,
            input_validation="PASSED",
            input_length=input_length or len(clean_query),
            retrieval_completed=True,
            retrieved_candidate_count=len(trace.filtered_chunks),
            evidence_sufficient=not is_general_knowledge,
            guardrail_triggered=is_general_knowledge or out_triggered,
            guardrail_reason="Fallback to general LLM" if is_general_knowledge else out_reason,
            llm_used=True,
            output_validation=out_validation,
            output_guardrail_triggered=out_triggered,
            output_guardrail_reason=out_reason,
            final_result_type=final_res_type,
            guardrails_enabled=guardrails_enabled,
        )
        return resp, True

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        model: Optional[str] = None,
        chat_mode: str = "grounded",
        guardrails_enabled: Optional[bool] = None,
    ) -> RAGResponse | ComparisonResponse:
        """Execute synchronous RAG pipeline with 3-layer guardrail protection."""
        active_guardrails = (
            settings.GUARDRAILS_ENABLED if guardrails_enabled is None else guardrails_enabled
        )

        if chat_mode in ("compare", "compare_both"):
            return self.compare(
                query_text=query_text,
                top_k=top_k,
                threshold=threshold,
                document_id=document_id,
                model=model,
                guardrails_enabled=active_guardrails,
            )

        t_pipeline_start = time.perf_counter()
        query_id = str(uuid.uuid4())
        active_model = model or self.ollama_client.model

        # ── LAYER 1: Input Guardrail ───────────────────────────────────────────
        input_res = self.input_guardrail.validate(query_text, enabled=active_guardrails)
        if not input_res.is_valid:
            logger.info(f"Input guardrail rejected query: {input_res.reason}")
            empty_trace = RetrievalTrace(
                query=query_text,
                top_k=top_k or settings.TOP_K,
                threshold=threshold or settings.SIMILARITY_THRESHOLD,
                distance_metric=settings.DISTANCE_METRIC,
            )
            response = RAGResponse(
                query=query_text,
                answer=input_res.error_message or QUERY_MALFORMED_MESSAGE,
                sources=[],
                trace=empty_trace,
                context_sent="",
                model=active_model,
                embedding_time_ms=0.0,
                retrieval_time_ms=0.0,
                context_time_ms=0.0,
                generation_time_ms=0.0,
                total_time_ms=0.0,
                is_low_confidence=True,
                is_general_knowledge=False,
                has_relevant_evidence=False,
                input_validation="FAILED",
                input_length=input_res.original_length,
                retrieval_completed=False,
                retrieved_candidate_count=0,
                evidence_sufficient=False,
                guardrail_triggered=True,
                guardrail_reason=input_res.reason,
                llm_used=False,
                output_validation="NOT_APPLICABLE",
                output_guardrail_triggered=False,
                output_guardrail_reason="",
                final_result_type="BLOCKED_INPUT",
                guardrails_enabled=active_guardrails,
            )
            self._log_query(response, query_id)
            return response

        clean_query = input_res.sanitized_query

        # ── LAYER 2: Qdrant Retrieval & Evidence Guardrail ─────────────────────
        trace = self.retriever.retrieve(
            query=clean_query,
            top_k=top_k,
            threshold=threshold,
            document_id=document_id,
        )

        ev_res = self.evidence_guardrail.validate(
            query=clean_query,
            chunks=trace.filtered_chunks,
            max_similarity=trace.highest_similarity,
            threshold=trace.threshold,
            enabled=active_guardrails,
        )

        # ── LAYER 3: Strategy Execution & Output Guardrail ─────────────────────
        if chat_mode == "grounded":
            response, _ = self._execute_grounded_strategy(
                clean_query=clean_query,
                trace=trace,
                active_model=active_model,
                t_start=t_pipeline_start,
                has_relevant_evidence=ev_res.is_sufficient,
                guardrail_reason=ev_res.reason,
                guardrails_enabled=active_guardrails,
                input_length=input_res.original_length,
            )
        else:
            response, _ = self._execute_rag_llm_strategy(
                clean_query=clean_query,
                trace=trace,
                active_model=active_model,
                t_start=t_pipeline_start,
                has_relevant_evidence=ev_res.is_sufficient,
                guardrail_reason=ev_res.reason,
                guardrails_enabled=active_guardrails,
                input_length=input_res.original_length,
            )

        self._log_query(response, query_id)
        return response

    def compare(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        model: Optional[str] = None,
        guardrails_enabled: Optional[bool] = None,
    ) -> ComparisonResponse:
        """Execute side-by-side comparison of Grounded RAG and RAG + LLM under guardrails."""
        active_guardrails = (
            settings.GUARDRAILS_ENABLED if guardrails_enabled is None else guardrails_enabled
        )
        active_model = model or self.ollama_client.model

        # ── LAYER 1: Input Guardrail ───────────────────────────────────────────
        input_res = self.input_guardrail.validate(query_text, enabled=active_guardrails)
        if not input_res.is_valid:
            logger.info(f"Input guardrail rejected compare query: {input_res.reason}")
            empty_trace = RetrievalTrace(
                query=query_text,
                top_k=top_k or settings.TOP_K,
                threshold=threshold or settings.SIMILARITY_THRESHOLD,
                distance_metric=settings.DISTANCE_METRIC,
            )
            blocked_resp = RAGResponse(
                query=query_text,
                answer=input_res.error_message or QUERY_MALFORMED_MESSAGE,
                sources=[],
                trace=empty_trace,
                context_sent="",
                model=active_model,
                embedding_time_ms=0.0,
                retrieval_time_ms=0.0,
                context_time_ms=0.0,
                generation_time_ms=0.0,
                total_time_ms=0.0,
                is_low_confidence=True,
                is_general_knowledge=False,
                has_relevant_evidence=False,
                input_validation="FAILED",
                input_length=input_res.original_length,
                retrieval_completed=False,
                retrieved_candidate_count=0,
                evidence_sufficient=False,
                guardrail_triggered=True,
                guardrail_reason=input_res.reason,
                llm_used=False,
                output_validation="NOT_APPLICABLE",
                output_guardrail_triggered=False,
                output_guardrail_reason="",
                final_result_type="BLOCKED_INPUT",
                guardrails_enabled=active_guardrails,
            )
            self._log_query(blocked_resp, str(uuid.uuid4()))
            return ComparisonResponse(
                query=query_text,
                model=active_model,
                grounded_response=blocked_resp,
                rag_llm_response=blocked_resp,
                trace=empty_trace,
                retrieved_count=0,
                highest_similarity=0.0,
                grounded_context_retrieved=False,
                grounded_llm_used=False,
                grounded_generation_time_s=0.0,
                grounded_result_label=input_res.error_message or "Blocked by Input Guardrail",
                rag_llm_context_retrieved=False,
                rag_llm_used=False,
                rag_llm_generation_time_s=0.0,
                rag_llm_result_label=input_res.error_message or "Blocked by Input Guardrail",
                has_relevant_evidence=False,
                relevance_reason=input_res.reason,
                input_validation="FAILED",
                input_length=input_res.original_length,
                retrieval_completed=False,
                retrieved_candidate_count=0,
                evidence_sufficient=False,
                guardrails_enabled=active_guardrails,
            )

        clean_query = input_res.sanitized_query

        # ── LAYER 2: Qdrant Retrieval (ONCE) & Evidence Guardrail (ONCE) ───────
        trace = self.retriever.retrieve(
            query=clean_query,
            top_k=top_k,
            threshold=threshold,
            document_id=document_id,
        )

        ev_res = self.evidence_guardrail.validate(
            query=clean_query,
            chunks=trace.filtered_chunks,
            max_similarity=trace.highest_similarity,
            threshold=trace.threshold,
            enabled=active_guardrails,
        )
        has_evidence = ev_res.is_sufficient

        # ── LAYER 3: Grounded RAG & Output Guardrail ───────────────────────────
        t_g_start = time.perf_counter()
        grounded_resp, grounded_llm_used = self._execute_grounded_strategy(
            clean_query=clean_query,
            trace=trace,
            active_model=active_model,
            t_start=t_g_start,
            has_relevant_evidence=has_evidence,
            guardrail_reason=ev_res.reason,
            guardrails_enabled=active_guardrails,
            input_length=input_res.original_length,
        )
        self._log_query(grounded_resp, str(uuid.uuid4()))

        # ── LAYER 3: RAG + LLM & Output Guardrail ─────────────────────────────
        t_r_start = time.perf_counter()
        rag_llm_resp, rag_llm_used = self._execute_rag_llm_strategy(
            clean_query=clean_query,
            trace=trace,
            active_model=active_model,
            t_start=t_r_start,
            has_relevant_evidence=has_evidence,
            guardrail_reason=ev_res.reason,
            guardrails_enabled=active_guardrails,
            input_length=input_res.original_length,
        )
        self._log_query(rag_llm_resp, str(uuid.uuid4()))

        # ── Metadata Assembly ──────────────────────────────────────────────────
        retrieved_count = len(trace.filtered_chunks)
        grounded_context_retrieved = has_evidence
        grounded_gen_time_s = round(grounded_resp.generation_time_ms / 1000.0, 2)
        grounded_label = (
            "Grounded Document Answer"
            if has_evidence
            else LOW_CONFIDENCE_MESSAGE
        )

        rag_llm_context_retrieved = has_evidence and not rag_llm_resp.is_general_knowledge
        rag_llm_gen_time_s = round(rag_llm_resp.generation_time_ms / 1000.0, 2)
        rag_llm_label = (
            "Grounded Document Answer"
            if rag_llm_context_retrieved
            else GENERAL_LLM_LABEL
        )

        return ComparisonResponse(
            query=clean_query,
            model=active_model,
            grounded_response=grounded_resp,
            rag_llm_response=rag_llm_resp,
            trace=trace,
            retrieved_count=retrieved_count,
            highest_similarity=trace.highest_similarity,
            grounded_context_retrieved=grounded_context_retrieved,
            grounded_llm_used=grounded_llm_used,
            grounded_generation_time_s=grounded_gen_time_s,
            grounded_result_label=grounded_label,
            rag_llm_context_retrieved=rag_llm_context_retrieved,
            rag_llm_used=rag_llm_used,
            rag_llm_generation_time_s=rag_llm_gen_time_s,
            rag_llm_result_label=rag_llm_label,
            has_relevant_evidence=has_evidence,
            relevance_reason=ev_res.reason,
            input_validation="PASSED",
            input_length=input_res.original_length,
            retrieval_completed=True,
            retrieved_candidate_count=retrieved_count,
            evidence_sufficient=has_evidence,
            guardrails_enabled=active_guardrails,
        )

    def _log_query(self, response: RAGResponse, query_id: str) -> None:
        """Record query telemetry and performance timings to SQLite."""
        try:
            record = QueryRecord(
                query_id=query_id,
                query=response.query,
                model=response.model,
                top_k=response.trace.top_k,
                threshold=response.trace.threshold,
                retrieved_count=len(response.trace.filtered_chunks),
                retrieval_time=response.retrieval_time_ms,
                generation_time=response.generation_time_ms,
                total_time=response.total_time_ms,
                max_similarity=response.trace.highest_similarity,
                is_low_confidence=response.is_low_confidence,
                input_validation=response.input_validation,
                input_length=response.input_length,
                retrieval_completed=response.retrieval_completed,
                retrieved_candidate_count=response.retrieved_candidate_count,
                evidence_sufficient=response.evidence_sufficient,
                guardrail_triggered=response.guardrail_triggered,
                guardrail_reason=response.guardrail_reason,
                llm_used=response.llm_used,
                output_validation=response.output_validation,
                output_guardrail_triggered=response.output_guardrail_triggered,
                output_guardrail_reason=response.output_guardrail_reason,
                final_result_type=response.final_result_type,
            )
            self.query_repo.insert(record)
        except Exception as exc:
            logger.warning(f"Failed to log query telemetry to database: {exc}")


_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Retrieve or initialize singleton RAGPipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
