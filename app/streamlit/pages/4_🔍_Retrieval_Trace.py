"""Streamlit Retrieval Trace & Observability Workbench Page."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.config import settings
from app.guardrails import InputGuardrail
from app.rag.evaluator import EvidenceEvaluator
from app.rag.retriever import RetrievalEngine
from app.streamlit.components.status import render_sidebar_status

st.set_page_config(page_title="Retrieval Trace Workbench", page_icon="🔍", layout="wide")

render_sidebar_status()

st.title("🔍 Retrieval Trace & Observability Workbench")
st.markdown(
    "Deeply inspect how semantic search works: view vector similarity distributions, "
    "per-chunk similarity rankings, cutoff threshold dynamics, and **guardrail evaluations**."
)

retriever = RetrievalEngine()
input_guardrail = InputGuardrail(max_length=settings.MAX_QUERY_LENGTH)

col_query, col_params = st.columns([3, 1])

with col_query:
    test_query = st.text_input(
        "Search Query",
        value="What is the core contribution of this work?",
        placeholder="Enter a test query to trace retrieval...",
    )

with col_params:
    top_k = st.slider("Top-K Limit", min_value=1, max_value=25, value=5, step=1)
    threshold = st.slider("Threshold Cutoff", min_value=0.0, max_value=1.0, value=float(settings.SIMILARITY_THRESHOLD), step=0.05)

if test_query:
    inp_res = input_guardrail.validate(test_query)
    if not inp_res.is_valid:
        st.warning(f"🛡️ **Input Guardrail Flagged Query:** {inp_res.error_message} *(Reason: {inp_res.reason})*")

    with st.spinner("Executing retrieval trace..."):
        trace = retriever.retrieve(query=test_query, top_k=top_k, threshold=threshold)
        evidence_decision = EvidenceEvaluator.evaluate(
            query=test_query,
            chunks=trace.filtered_chunks,
            max_similarity=trace.highest_similarity,
            threshold=threshold,
        )

    st.markdown("### 📊 Trace Telemetry & Grounding Evidence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Query Embedding Latency", f"{trace.embedding_time_ms:.2f} ms")
    c2.metric("Qdrant Search Latency", f"{trace.retrieval_time_ms:.2f} ms")
    c3.metric("Peak Similarity Score", f"{trace.highest_similarity:.4f}")
    conf_status = "🔴 Low Confidence" if trace.is_low_confidence else "🟢 Confident"
    c4.metric("Confidence Status", conf_status)

    c5, c6 = st.columns([1, 3])
    ev_label = "✅ Grounded Evidence" if evidence_decision.has_relevant_evidence else "⚠️ Insufficient Evidence"
    c5.metric("Evidence Decision", ev_label)
    c6.info(f"**Evidence Reasoning:** {evidence_decision.reason} *(Token coverage: {evidence_decision.token_coverage * 100:.0f}% | Input length: {len(test_query)}/{settings.MAX_QUERY_LENGTH} chars)*")

    st.markdown("---")
    st.markdown(f"### 📑 Ranked Chunks ({len(trace.retrieved_chunks)} Retrieved | {len(trace.filtered_chunks)} Passed)")

    if not trace.retrieved_chunks:
        st.warning("No chunks found matching the query in the vector store.")
    else:
        for chunk in trace.retrieved_chunks:
            passed = chunk.similarity_score >= threshold
            status_tag = "✅ PASS" if passed else "❌ BELOW THRESHOLD"

            with st.container():
                st.markdown(
                    f"#### Rank #{chunk.rank}: `{chunk.filename}` (Page {chunk.page})"
                )
                col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
                col_info1.caption(f"Chunk ID: `{chunk.chunk_id}` | Version: `{chunk.pipeline_version}`")
                col_info2.markdown(f"**Score:** `{chunk.similarity_score:.4f}`")
                col_info3.markdown(f"**Gate:** {status_tag}")

                st.code(chunk.text, language="text")
                st.markdown("---")
