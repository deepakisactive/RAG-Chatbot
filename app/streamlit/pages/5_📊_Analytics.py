"""Streamlit System Analytics."""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
import pandas as pd
import streamlit as st
from app.database.database import get_db
from app.database.repository import DocumentRepository, QueryRepository
from app.streamlit.components.status import render_sidebar_status
st.set_page_config(page_title="RAG Analytics", page_icon="📊", layout="wide")
render_sidebar_status()
st.title("📊 RAG Analytics & Telemetry")
st.markdown("Monitor query history, latency distributions, per-model performance, and retrieval quality.")
doc_repo = DocumentRepository(db=get_db())
query_repo = QueryRepository(db=get_db())
doc_stats = doc_repo.get_stats()
query_stats = query_repo.get_stats()
st.markdown("### 📈 Operational Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📚 Indexed Documents", doc_stats["total_documents"])
c2.metric("📄 Extracted Pages", f"{doc_stats['total_pages']:,}")
c3.metric("🧩 Vector Chunks", f"{doc_stats['total_chunks']:,}")
c4.metric("💬 Total Queries", query_stats["total_queries"])
c5, c6, c7, c8 = st.columns(4)
c5.metric("⚡ Avg Retrieval", f"{query_stats['avg_retrieval_time']:.1f} ms")
c6.metric("🤖 Avg Generation", f"{query_stats['avg_generation_time']:.1f} ms")
c7.metric("⏱️ Avg Total", f"{query_stats['avg_total_time']:.1f} ms")
c8.metric("🎯 Avg Similarity", f"{query_stats['avg_similarity']:.4f}")
c9, c10 = st.columns(2)
c9.metric("✅ Grounded Queries", query_stats.get("successful_retrievals", 0))
c10.metric("⚠️ Low-Confidence", query_stats.get("low_confidence_queries", 0))
st.markdown("---")
recent_queries = query_repo.list_recent(limit=500)
if recent_queries:
    st.markdown("### 🤖 Per-Model Performance Breakdown")
    model_stats = defaultdict(lambda: {"count": 0, "gen_total": 0.0, "retr_total": 0.0, "total_total": 0.0, "sim_total": 0.0, "low_conf": 0})
    for q in recent_queries:
        m = q.model or "unknown"
        model_stats[m]["count"] += 1
        model_stats[m]["gen_total"] += q.generation_time or 0
        model_stats[m]["retr_total"] += q.retrieval_time or 0
        model_stats[m]["total_total"] += q.total_time or 0
        model_stats[m]["sim_total"] += q.max_similarity or 0
        if q.is_low_confidence:
            model_stats[m]["low_conf"] += 1
    model_rows = []
    for m_name, s in sorted(model_stats.items()):
        n = s["count"]
        model_rows.append({"Model": m_name, "Queries": n, "Avg Generation (ms)": round(s["gen_total"] / n, 1), "Avg Retrieval (ms)": round(s["retr_total"] / n, 1), "Avg Total (ms)": round(s["total_total"] / n, 1), "Avg Similarity": round(s["sim_total"] / n, 4), "Low-Confidence (%)": round((s["low_conf"] / n) * 100, 1)})
    df_models = pd.DataFrame(model_rows)
    st.dataframe(df_models, use_container_width=True)
    if len(model_rows) > 1:
        st.markdown("**Latency Comparison by Model**")
        chart_df = pd.DataFrame({"Retrieval (ms)": {r["Model"]: r["Avg Retrieval (ms)"] for r in model_rows}, "Generation (ms)": {r["Model"]: r["Avg Generation (ms)"] for r in model_rows}})
        st.bar_chart(chart_df)
    st.markdown("---")
st.markdown("### 📜 Recent Query History")
if not recent_queries:
    st.info("No queries yet. Ask questions on the Chat page to generate telemetry.")
else:
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        filter_low_conf = st.checkbox("Show only low-confidence queries", value=False)
    with col_f2:
        filter_guardrail = st.checkbox("Show only guardrail-triggered queries", value=False)
    with col_f3:
        all_models_seen = sorted(set(q.model or "unknown" for q in recent_queries))
        filter_model = st.selectbox("Filter by Model", options=["ALL"] + all_models_seen)

    display_data = []
    for q in recent_queries:
        if filter_low_conf and not q.is_low_confidence:
            continue
        if filter_guardrail and not getattr(q, "guardrail_triggered", False):
            continue
        q_model = q.model or "unknown"
        if filter_model != "ALL" and q_model != filter_model:
            continue

        gr_status = "🛡️ Triggered" if getattr(q, "guardrail_triggered", False) else "✓ Passed"
        llm_used_str = "Yes" if getattr(q, "llm_used", True) else "No"
        result_type = getattr(q, "final_result_type", "UNKNOWN") or "UNKNOWN"
        input_len = getattr(q, "input_length", len(q.query)) or len(q.query)

        display_data.append({
            "Timestamp": q.timestamp,
            "Query": q.query,
            "Model": q_model,
            "Length": input_len,
            "Result Type": result_type,
            "Guardrail": gr_status,
            "Guardrail Reason": getattr(q, "guardrail_reason", "") or "-",
            "LLM Used": llm_used_str,
            "Candidates": getattr(q, "retrieved_candidate_count", q.retrieved_count) or q.retrieved_count,
            "Similarity": round(q.max_similarity, 4) if q.max_similarity is not None else 0.0,
            "Retrieval (ms)": round(q.retrieval_time, 1) if q.retrieval_time is not None else 0.0,
            "Generation (ms)": round(q.generation_time, 1) if q.generation_time is not None else 0.0,
            "Total (ms)": round(q.total_time, 1) if q.total_time is not None else 0.0,
        })
    if display_data:
        st.dataframe(display_data, use_container_width=True)
        st.caption(f"Showing {len(display_data)} queries.")
    else:
        st.caption("No queries match the selected filters.")
