"""Streamlit System Architecture, Health Checks & Configuration Page."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.config import settings
from app.database.database import get_db
from app.database.repository import DocumentRepository
from app.ingestion.embedder import get_embedder
from app.llm.ollama_client import get_ollama_client
from app.streamlit.components.status import render_sidebar_status
from app.vectorstore.qdrant_client import get_vectorstore

st.set_page_config(page_title="System & Architecture", page_icon="🖥️", layout="wide")

render_sidebar_status()

st.title("🖥️ System Architecture & Connectivity Status")
st.markdown("Inspect component configurations, host networking parameters, and runtime health probes.")

ollama_client = get_ollama_client()
vectorstore = get_vectorstore()
embedder = get_embedder()
doc_repo = DocumentRepository(db=get_db())

# ── Live Health Checks ─────────────────────────────────────────────────────────
st.markdown("### 🩺 Live Health Probes")

ollama_health = ollama_client.health_check()
qdrant_health = vectorstore.health_check()
all_models = ollama_client.list_models()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🦙 Ollama Runtime")
    if ollama_health.get("status") == "connected":
        st.success("✅ **Status: Connected**")
        st.markdown(f"**Host:** `{settings.OLLAMA_BASE_URL}` (Windows via NAT 10.0.2.2)")
        st.markdown(f"**Models Available:** `{len(all_models)}`")
        for m in all_models:
            m_name = m.get("name")
            m_size = m.get("details", {}).get("parameter_size", "N/A")
            m_quant = m.get("details", {}).get("quantization_level", "N/A")
            m_mb = m.get("size", 0) // (1024 * 1024)
            st.caption(f"• `{m_name}` — {m_size}, {m_quant}, {m_mb:,} MB")
    else:
        st.error("❌ **Status: Disconnected**")
        st.caption(ollama_health.get("error", "Unable to connect"))
        st.warning(
            "👉 **Fix:** Set `OLLAMA_HOST=0.0.0.0:11434` in Windows Environment Variables "
            "and allow port 11434 through Windows Firewall."
        )

with col2:
    st.markdown("#### ⚡ Qdrant Vector DB")
    if qdrant_health.get("status") == "connected":
        st.success("✅ **Status: Connected**")
        st.markdown(f"**Instance:** Docker Container (`{settings.QDRANT_URL}`)")
        st.markdown(f"**Collection:** `{settings.QDRANT_COLLECTION}`")
        st.markdown(f"**Vector Count:** `{qdrant_health.get('vector_count', 0):,}`")
        st.markdown("**Storage:** Persistent Docker Volume")
    else:
        st.error("❌ **Status: Disconnected**")
        st.caption(qdrant_health.get("error", "Qdrant unavailable"))

with col3:
    st.markdown("#### 🧠 Embedding Engine")
    st.success("✅ **Status: Ready**")
    st.markdown(f"**Model:** `{embedder.model_name}`")
    st.markdown(f"**Dimensions:** `{embedder.dimension}` floats")
    st.markdown(f"**Distance Metric:** `{settings.DISTANCE_METRIC}`")
    st.markdown("**Hardware:** CPU (ONNX Runtime, FastEmbed)")

st.markdown("---")

# ── Live Model Ping Test ───────────────────────────────────────────────────────
st.markdown("### 🚀 Live Model Ping Test")
st.caption("Send a tiny test prompt to each LLM and measure first-token latency.")

if st.button("▶️ Ping All 3 Models Now", type="primary"):
    ping_cols = st.columns(len(all_models) if all_models else 1)
    for i, m in enumerate(all_models):
        m_name = m.get("name", "unknown")
        with ping_cols[i]:
            with st.spinner(f"Pinging `{m_name}`..."):
                try:
                    t0 = time.perf_counter()
                    ans, latency = ollama_client.generate(
                        prompt="Reply with exactly one word: OK",
                        model=m_name,
                    )
                    elapsed = round((time.perf_counter() - t0) * 1000, 1)
                    st.success(f"✅ **`{m_name}`**")
                    st.metric("Response Time", f"{elapsed:.0f} ms")
                    st.caption(f"Reply: `{ans[:80]}`")
                except Exception as e:
                    st.error(f"❌ `{m_name}` failed: {e}")

st.markdown("---")

# ── Architecture Diagram ───────────────────────────────────────────────────────
st.markdown("### 🏗️ Topology & Environment Mapping")
col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown(
        """
        ```
        ┌───────────────────────────────────────────────────────┐
        │                      WINDOWS HOST                     │
        │  Ollama Runtime (OLLAMA_HOST=0.0.0.0:11434)           │
        │  ├── llama3.2:1b       (1.2B params, Q8_0)            │
        │  ├── starcoder2:3b     (3B params,   Q4_0)            │
        │  └── qwen2.5-coder:1.5b(1.5B params, Q4_K_M)         │
        └─────────────────────────┬─────────────────────────────┘
                                  │ HTTP /api/generate  /api/tags
                           10.0.2.2:11434 (VirtualBox NAT)
        ┌─────────────────────────┴─────────────────────────────┐
        │              VIRTUALBOX UBUNTU VM                     │
        │  ┌─────────────────────────────────────────────────┐  │
        │  │  DOCKER COMPOSE STACK                           │  │
        │  │                                                 │  │
        │  │  [ rag-app  0.0.0.0:8501 ]                      │  │
        │  │   • Streamlit UI (7 pages)                      │  │
        │  │   • RAG Pipeline (Retrieve → Prompt → Generate) │  │
        │  │   • FastEmbed BAAI/bge-small-en-v1.5 (384D)     │  │
        │  │   • PyMuPDF Document Parser + Chunker           │  │
        │  │   • SQLite Metadata Store (metadata/rag.db)     │  │
        │  │   • Evaluation Suite (Exercises 1-6)            │  │
        │  │                                                 │  │
        │  │       ↓ Qdrant gRPC/HTTP :6333                  │  │
        │  │                                                 │  │
        │  │  [ rag-qdrant ]                                 │  │
        │  │   • Qdrant Vector Database Engine               │  │
        │  │   • Collection: rag_documents (241 chunks)      │  │
        │  │   • Persistent Volume: qdrant_data              │  │
        │  └─────────────────────────────────────────────────┘  │
        └───────────────────────────────────────────────────────┘
        ```
        """
    )

with col_b:
    st.markdown("#### 📊 Runtime Configuration")
    stats = doc_repo.get_stats()
    st.markdown(
        f"""
        | Setting | Value |
        |---------|-------|
        | Embedding Model | `BAAI/bge-small-en-v1.5` |
        | Vector Dimensions | `{embedder.dimension}` |
        | Distance Metric | `{settings.DISTANCE_METRIC}` |
        | Default Top-K | `{settings.TOP_K}` |
        | Similarity Threshold | `{settings.SIMILARITY_THRESHOLD}` |
        | Max Query Length | `{settings.MAX_QUERY_LENGTH} chars` |
        | Guardrails Enabled | `{settings.GUARDRAILS_ENABLED}` |
        | Ollama Timeout | `{settings.OLLAMA_TIMEOUT_SECONDS}s` |
        | Indexed Documents | `{stats['total_documents']}` |
        | Total Chunks | `{stats['total_chunks']:,}` |
        | Default Model | `{settings.OLLAMA_MODEL}` |
        """
    )

st.markdown("---")
st.markdown("### ⚙️ Active System Settings (Full)")
st.json(settings.to_dict())
