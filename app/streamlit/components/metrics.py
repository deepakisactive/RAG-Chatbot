"""Performance and Telemetry Metrics Display Component."""

from __future__ import annotations

import streamlit as st

from app.rag.models import RAGResponse


def render_latency_breakdown(response: RAGResponse) -> None:
    """Render a clean latency breakdown and telemetry metrics row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Query Embedding", f"{response.embedding_time_ms:.1f} ms")
    with col2:
        st.metric("Vector Search", f"{response.retrieval_time_ms:.1f} ms")
    with col3:
        st.metric("Context Assembly", f"{response.context_time_ms:.1f} ms")
    with col4:
        if response.generation_time_ms <= 0.0 or (response.is_low_confidence and not response.is_general_knowledge):
            st.metric("LLM Generation", "Not Used")
        else:
            gen_display = (
                f"{response.generation_time_ms / 1000.0:.2f} s"
                if response.generation_time_ms >= 1000
                else f"{response.generation_time_ms:.0f} ms"
            )
            st.metric("LLM Generation", gen_display)
    with col5:
        total_display = (
            f"{response.total_time_ms / 1000.0:.2f} s"
            if response.total_time_ms >= 1000
            else f"{response.total_time_ms:.0f} ms"
        )
        st.metric("Total Latency", total_display)
