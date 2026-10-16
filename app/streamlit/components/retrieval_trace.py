"""Retrieval Trace and Observability Component."""

from __future__ import annotations

import streamlit as st

from app.rag.models import RAGResponse, RetrievalTrace


def render_retrieval_trace(response: RAGResponse) -> None:
    """Render full retrieval transparency details in an expandable view."""
    trace = response.trace

    with st.expander("🔍 Inspect Retrieval Trace & Grounding Evidence", expanded=False):
        st.markdown("#### 🔬 Retrieval Parameters & Telemetry")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Query:** `{trace.query}`")
        c2.markdown(f"**Top-K:** `{trace.top_k}`")
        c3.markdown(f"**Similarity Metric:** `{trace.distance_metric}`")
        c4.markdown(f"**Threshold:** `{trace.threshold:.2f}`")

        c5, c6, c7, c8 = st.columns(4)
        c5.markdown(f"**Retrieved Chunks:** `{len(trace.retrieved_chunks)}`")
        c6.markdown(f"**Passing Chunks:** `{len(trace.filtered_chunks)}`")
        c7.markdown(f"**Peak Similarity:** `{trace.highest_similarity:.4f}`")
        has_ev = getattr(response, "has_relevant_evidence", False)
        c8.markdown(f"**Evidence Grounded:** `{'Yes' if has_ev else 'No'}`")

        st.markdown("---")
        st.markdown("#### 📑 Ranked Chunks")

        if not trace.retrieved_chunks:
            st.info("No chunks were retrieved from the vector index.")
        else:
            for chunk in trace.retrieved_chunks:
                is_passed = chunk.similarity_score >= trace.threshold
                badge = "🟢 PASS" if is_passed else "🔴 FILTERED OUT"

                with st.container():
                    st.markdown(
                        f"**Rank #{chunk.rank}** | **Doc:** `{chunk.filename}` | "
                        f"**Page:** {chunk.page} | **Score:** `{chunk.similarity_score:.4f}` | "
                        f"**Status:** {badge}"
                    )
                    st.code(chunk.text, language="text")

        st.markdown("---")
        st.markdown("#### 📝 Context Passed to LLM")
        if response.context_sent:
            st.code(response.context_sent, language="text")
        elif not getattr(response, "has_relevant_evidence", True):
            st.caption("No context was forwarded (no relevant evidence found in retrieved chunks).")
        else:
            st.caption("No context was forwarded (query failed confidence threshold).")
