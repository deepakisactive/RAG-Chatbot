"""Streamlit Unified Chat Interface Page.

Supports Grounded RAG, RAG + LLM, and side-by-side Compare Both modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.config import settings
from app.llm.ollama_client import get_ollama_client
from app.rag.models import ComparisonResponse, RAGResponse
from app.rag.pipeline import get_pipeline
from app.rag.prompt import GENERAL_LLM_LABEL, LOW_CONFIDENCE_MESSAGE
from app.streamlit.components.metrics import render_latency_breakdown
from app.streamlit.components.retrieval_trace import render_retrieval_trace
from app.streamlit.components.status import render_sidebar_status

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

render_sidebar_status()

# ── Sidebar Controls ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Selection")

_ollama_client = get_ollama_client()
_available_models = [m.get("name") for m in _ollama_client.list_models()] or [
    "llama3.2:1b",
    "qwen2.5-coder:1.5b",
    "starcoder2:3b",
]
default_idx = (
    _available_models.index(settings.OLLAMA_MODEL)
    if settings.OLLAMA_MODEL in _available_models
    else 0
)
selected_model = st.sidebar.selectbox(
    "Active LLM Model",
    options=_available_models,
    index=default_idx,
    help="Select which local LLM to query on the host. Used for both modes during comparison.",
)

# Show model details
_all_model_data = _ollama_client.list_models()
_model_detail = next((m for m in _all_model_data if m.get("name") == selected_model), {})
if _model_detail:
    _size = _model_detail.get("details", {}).get("parameter_size", "?")
    _quant = _model_detail.get("details", {}).get("quantization_level", "?")
    st.sidebar.caption(
        f"Parameters: **{_size}** | Quantization: **{_quant}** | "
        f"num_predict: **{settings.LLM_NUM_PREDICT}** | temp: **{settings.LLM_TEMPERATURE}**"
    )

st.sidebar.markdown("### 🔀 Chat Mode")
chat_mode = st.sidebar.radio(
    "Select Answering Strategy",
    options=["Grounded RAG", "RAG + LLM", "Compare Both"],
    index=0,
    help=(
        "• Grounded RAG: Answers strictly from uploaded documents; refuses if missing.\n"
        "• RAG + LLM: Grounded in documents; falls back to general LLM if missing.\n"
        "• Compare Both: Executes both strategies side-by-side using the exact same retrieval results."
    ),
)

st.sidebar.markdown("### ⚙️ Retrieval Controls")
top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=15, value=settings.TOP_K, step=1)
threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(settings.SIMILARITY_THRESHOLD),
    step=0.05,
    key="chat_threshold_slider",
    help="Cosine similarity threshold for retrieved chunks. (Recommended: 0.45 - 0.60)",
)

st.sidebar.markdown("### 🛡️ Guardrails")
guardrails_enabled = st.sidebar.checkbox(
    "Enable Guardrails",
    value=settings.GUARDRAILS_ENABLED,
    help="Toggle Input, Evidence, and Output Guardrails on or off to demonstrate Before vs After behavior.",
)

st.sidebar.markdown("### 🛠️ Options")
show_token_stats = st.sidebar.checkbox("Show Token Statistics", value=True)
show_context = st.sidebar.checkbox("Show Retrieved Context", value=False)

if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state["chat_messages"] = []
    st.rerun()

# ── Main Chat Area ─────────────────────────────────────────────────────────────
st.title("💬 Chat")

mode_badges = {
    "Grounded RAG": "📚 **Grounded RAG** *(Strict document answering)*",
    "RAG + LLM": "🤖 **RAG + LLM** *(Document-grounded with general fallback)*",
    "Compare Both": "⚖️ **Compare Both** *(Side-by-side strategy evaluation on identical retrieval)*",
}
st.caption(
    f"Active Mode: {mode_badges[chat_mode]} | Model: `{selected_model}` | "
    f"Top-K: {top_k} | Cutoff: {threshold:.2f} | num_predict: {settings.LLM_NUM_PREDICT}"
)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []


def _render_single_response(resp: RAGResponse, mode_name: str) -> None:
    """Render a single RAGResponse object."""
    # ── Guardrail Status Indicator ──
    if getattr(resp, "guardrail_triggered", False):
        st.warning(f"🛡️ **Guardrail Triggered** — *{resp.guardrail_reason or 'Verification flagged'}*")
    else:
        st.success("✓ **Guardrail Passed**")

    if mode_name == "Grounded RAG":
        if resp.is_low_confidence or not resp.has_relevant_evidence:
            st.warning("⚠️ **Information Not Found**")
            st.markdown("Information not found in the uploaded documents.")
            st.caption("🔒 **LLM:** Not Used *(Refused: no relevant evidence found in uploaded documents)*")
        else:
            st.success("📚 **Grounded Document Answer**")
            st.markdown(resp.answer)
    else:  # RAG + LLM
        if resp.is_general_knowledge or not resp.has_relevant_evidence:
            st.info("🤖 **General LLM Answer**")
            st.caption(
                "ℹ️ *No relevant information was found in the uploaded documents. "
                "This answer was generated using the selected LLM's general knowledge.*"
            )
            st.markdown(resp.answer)
            st.caption("🔒 **RAG Context:** Not Used | **Reason:** No relevant evidence in retrieved documents")
        else:
            st.success("📚 **Grounded Document Answer**")
            st.markdown(resp.answer)

    with st.expander("🛡️ Guardrail Telemetry & Verification", expanded=False):
        gt1, gt2, gt3 = st.columns(3)
        gt1.markdown(f"**Input:** `{resp.input_validation}` ({resp.input_length} chars)")
        gt2.markdown(
            f"**Evidence:** `{'Sufficient' if resp.evidence_sufficient else 'Insufficient'}` "
            f"({resp.retrieved_candidate_count} candidates)"
        )
        gt3.markdown(f"**Output:** `{resp.output_validation}`")
        st.caption(
            f"Result Type: `{resp.final_result_type}` | "
            f"LLM Called: `{'Yes' if resp.llm_used else 'No'}` | "
            f"Guardrails Active: `{resp.guardrails_enabled}`"
        )

    if show_token_stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"🔍 Embed: `{resp.embedding_time_ms:.0f}` ms")
        c2.caption(f"📦 Retrieve: `{resp.retrieval_time_ms:.0f}` ms")
        gen_caption = (
            "**Not Used**"
            if (resp.generation_time_ms <= 0.0 or (resp.is_low_confidence and not resp.is_general_knowledge))
            else f"`{resp.generation_time_ms:.0f}` ms"
        )
        c3.caption(f"🤖 LLM Gen: {gen_caption}")
        c4.caption(f"⏱️ Total: `{resp.total_time_ms:.0f}` ms")

    if resp.has_relevant_evidence and not resp.is_general_knowledge and not resp.is_low_confidence:
        if resp.sources:
            with st.expander("📚 Sources & Citations", expanded=False):
                for src in resp.sources:
                    st.markdown(
                        f"- **{src.filename}** (Page {src.page}) — Similarity: `{src.similarity_score:.4f}`"
                    )
                    st.caption(f"Snippet: {src.snippet}")

        if show_context and resp.context_sent:
            with st.expander("📄 Full Context Sent to LLM", expanded=False):
                st.code(resp.context_sent, language="text")

        render_latency_breakdown(resp)
        render_retrieval_trace(resp)
    elif mode_name == "Grounded RAG":
        render_latency_breakdown(resp)
        render_retrieval_trace(resp)
    else:
        render_latency_breakdown(resp)


def _render_comparison(comp: ComparisonResponse) -> None:
    """Render side-by-side comparison of Grounded RAG and RAG + LLM."""
    st.markdown("### ⚖️ Side-by-Side Strategy Comparison")
    st.caption("Both modes evaluated using the **exact same query, model, and single retrieval pass**.")

    # Metadata metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🤖 Selected Model", comp.model)
    m2.metric("🧩 Retrieved Chunks", comp.retrieved_count)
    m3.metric("🎯 Max Similarity", f"{comp.highest_similarity:.4f}")
    m4.metric("⚙️ Threshold", f"{comp.trace.threshold:.2f}")

    # Telemetry card
    st.markdown("##### 📋 Strategy Telemetry")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        gen_time_disp = (
            f"{comp.grounded_generation_time_s:.2f}s"
            if comp.grounded_llm_used
            else "Not Used"
        )
        st.markdown(
            f"**Grounded RAG:**\n"
            f"- Retrieved candidates: `{comp.retrieved_count}`\n"
            f"- Relevant evidence: `{'Yes' if comp.has_relevant_evidence else 'No'}`\n"
            f"- LLM used: `{'Yes' if comp.grounded_llm_used else 'No'}`\n"
            f"- Generation time: `{gen_time_disp}`\n"
            f"- Result: `{comp.grounded_result_label}`"
        )
    with col_t2:
        st.markdown(
            f"**RAG + LLM:**\n"
            f"- Retrieved candidates: `{comp.retrieved_count}`\n"
            f"- Relevant evidence: `{'Yes' if comp.has_relevant_evidence else 'No'}`\n"
            f"- LLM used: `{'Yes' if comp.rag_llm_used else 'No'}`\n"
            f"- Generation time: `{comp.rag_llm_generation_time_s:.2f}s`\n"
            f"- Result: `{comp.rag_llm_result_label}`"
        )

    st.markdown("---")

    # Side-by-side response columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📚 Grounded RAG")
        st.caption("Strict document grounding (refuses if evidence missing)")
        if comp.grounded_response.is_low_confidence or not comp.has_relevant_evidence:
            st.warning("⚠️ **Information Not Found**")
            st.markdown("Information not found in the uploaded documents.")
            st.caption("🔒 **LLM:** Not Used *(Zero tokens requested)*")
        else:
            st.success("📚 Grounded Document Answer")
            st.markdown(comp.grounded_response.answer)

            if comp.grounded_response.sources:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for src in comp.grounded_response.sources:
                        st.markdown(
                            f"- **{src.filename}** (Page {src.page}) — Similarity: `{src.similarity_score:.4f}`"
                        )
                        st.caption(f"Snippet: {src.snippet}")

            if show_context and comp.grounded_response.context_sent:
                with st.expander("📄 Full Context Sent to LLM", expanded=False):
                    st.code(comp.grounded_response.context_sent, language="text")

        render_latency_breakdown(comp.grounded_response)
        render_retrieval_trace(comp.grounded_response)

    with col_right:
        st.markdown("#### 🤖 RAG + LLM")
        st.caption("Hybrid mode (document-grounded with general LLM fallback)")
        if comp.rag_llm_response.is_general_knowledge or not comp.has_relevant_evidence:
            st.info("🤖 **General LLM Answer**")
            st.caption(
                "ℹ️ *No relevant information was found in the uploaded documents. "
                "This answer was generated using the selected LLM's general knowledge.*"
            )
            st.markdown(comp.rag_llm_response.answer)
            st.caption("🔒 **RAG Context:** Not Used | **Reason:** No relevant evidence in retrieved documents")
        else:
            st.success("📚 Grounded Document Answer")
            st.markdown(comp.rag_llm_response.answer)

            if comp.rag_llm_response.sources:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for src in comp.rag_llm_response.sources:
                        st.markdown(
                            f"- **{src.filename}** (Page {src.page}) — Similarity: `{src.similarity_score:.4f}`"
                        )
                        st.caption(f"Snippet: {src.snippet}")

            if show_context and comp.rag_llm_response.context_sent:
                with st.expander("📄 Full Context Sent to LLM", expanded=False):
                    st.code(comp.rag_llm_response.context_sent, language="text")

        render_latency_breakdown(comp.rag_llm_response)
        if comp.has_relevant_evidence and not comp.rag_llm_response.is_general_knowledge:
            render_retrieval_trace(comp.rag_llm_response)


# Display conversation history
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            m_mode = msg.get("mode", "Grounded RAG")
            if m_mode == "Compare Both" and "compare_obj" in msg:
                _render_comparison(msg["compare_obj"])
            elif "response_obj" in msg:
                _render_single_response(msg["response_obj"], m_mode)
            else:
                st.markdown(msg["content"])

# User Input
if user_query := st.chat_input("Ask a question..."):
    st.session_state["chat_messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        pipeline = get_pipeline()

        if chat_mode == "Compare Both":
            with st.spinner(
                f"Retrieving vectors once & executing BOTH Grounded RAG + RAG+LLM with `{selected_model}`..."
            ):
                comparison = pipeline.compare(
                    query_text=user_query,
                    top_k=top_k,
                    threshold=threshold,
                    model=selected_model,
                    guardrails_enabled=guardrails_enabled,
                )
            _render_comparison(comparison)
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "mode": "Compare Both",
                    "content": "Compare Both execution completed.",
                    "compare_obj": comparison,
                }
            )
        else:
            internal_mode = "grounded" if chat_mode == "Grounded RAG" else "rag_plus_llm"
            with st.spinner(f"Processing query in {chat_mode} with `{selected_model}`..."):
                response = pipeline.query(
                    query_text=user_query,
                    top_k=top_k,
                    threshold=threshold,
                    model=selected_model,
                    chat_mode=internal_mode,
                    guardrails_enabled=guardrails_enabled,
                )
            _render_single_response(response, chat_mode)
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "mode": chat_mode,
                    "content": response.answer,
                    "response_obj": response,
                }
            )

