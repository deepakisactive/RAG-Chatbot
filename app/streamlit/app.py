"""Streamlit Main Application Entrypoint.

Home overview, system status telemetry, and navigation router.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.config import settings
from app.database.database import get_db
from app.database.repository import DocumentRepository, QueryRepository
from app.llm.ollama_client import get_ollama_client
from app.streamlit.components.status import render_sidebar_status

st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar_status()



# ──────────────────────────────────────────────────────────────────────────────
# Hero Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("Local RAG Chatbot")
st.markdown(
    "A fully transparent, observable Retrieval-Augmented Generation system with **multi-model evaluation**. "
    "Environment: Windows host runs 3 LLMs via Ollama · Ubuntu Docker runs the full RAG pipeline + Qdrant."
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Live System Health Banner
# ──────────────────────────────────────────────────────────────────────────────
ollama_client = get_ollama_client()
ollama_health = ollama_client.health_check()
all_models = ollama_client.list_models()

if ollama_health.get("status") == "connected":
    model_names = [m.get("name", "?") for m in all_models]
    st.success(
        f"✅ **All Systems Operational** — Ollama connected at `{settings.OLLAMA_BASE_URL}` "
        f"with **{len(all_models)} model(s)** ready: `{'` · `'.join(model_names)}`"
    )
else:
    st.error(
        f"❌ **Ollama Disconnected** — Cannot reach `{settings.OLLAMA_BASE_URL}`. "
        "Ensure `OLLAMA_HOST=0.0.0.0:11434` is set on the Windows host."
    )

# ──────────────────────────────────────────────────────────────────────────────
# KPI Metrics Row
# ──────────────────────────────────────────────────────────────────────────────
doc_repo = DocumentRepository(db=get_db())
query_repo = QueryRepository(db=get_db())
doc_stats = doc_repo.get_stats()
query_stats = query_repo.get_stats()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("📚 Indexed Docs", doc_stats["total_documents"])
c2.metric("🧩 Vector Chunks", f"{doc_stats['total_chunks']:,}")
c3.metric("💬 Total Queries", query_stats["total_queries"])
c4.metric("⚡ Avg Retrieval", f"{query_stats['avg_retrieval_time']:.0f} ms")
c5.metric("🤖 LLMs Available", len(all_models))
c6.metric("✅ Grounded", query_stats.get("successful_retrievals", 0))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Live Model Cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🤖 Available LLM Models (Windows Host Ollama)")
if all_models:
    model_icons = ["🦙", "⭐", "🧠", "🔮"]
    model_cols = st.columns(max(len(all_models), 1))
    for i, m in enumerate(all_models):
        with model_cols[i]:
            icon = model_icons[i % len(model_icons)]
            m_name = m.get("name", "unknown")
            m_size = m.get("details", {}).get("parameter_size", "?")
            m_quant = m.get("details", {}).get("quantization_level", "?")
            m_mb = m.get("size", 0) // (1024 * 1024)
            st.info(
                f"{icon} **`{m_name}`**\n\n"
                f"**Size:** {m_size}  \n"
                f"**Quant:** {m_quant}  \n"
                f"**Disk:** {m_mb:,} MB"
            )
else:
    st.warning("No models detected. Connect Ollama on the Windows host.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Navigation Cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Application Navigation")
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    st.info(
        "**💬 [Chat](Chat)**\n\n"
        "Unified conversational interface with 3 modes:\n"
        "• **Grounded RAG**: Strict document answers.\n"
        "• **RAG + LLM**: Hybrid mode with general fallback.\n"
        "• **Compare Both**: Side-by-side evaluation on identical retrieval."
    )
    st.info(
        "**📊 [Analytics](Analytics)**\n\n"
        "Query telemetry, latency breakdowns, and system utilization dashboards."
    )

with col_nav2:
    st.info(
        "**📄 [Documents](Documents)**\n\n"
        "Upload PDFs, Python code, and Markdown with SHA-256 deduplication. "
        "View chunk metadata and manage indexed corpus."
    )
    st.info(
        "**⚙️ [Pipeline](Pipeline)**\n\n"
        "Configure chunking strategy, chunk size, overlap, Top-K, and "
        "similarity threshold. Run incremental or full re-indexing."
    )

with col_nav3:
    st.info(
        "**🔍 [Retrieval Trace](Retrieval_Trace)**\n\n"
        "Inspect query embeddings, similarity score distributions, "
        "and analyze threshold cutoff behavior interactively."
    )
    st.info(
        "**🧪 [Model Evaluation](Model_Evaluation)**\n\n"
        "Full benchmark comparing all 3 LLMs on 25 representative tasks — "
        "accuracy, latency, hallucination, code test-pass, RAG traces, and automated 3-layer Guardrail & AI Output Testing."
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Architecture + Exercise Summary
# ──────────────────────────────────────────────────────────────────────────────
col_arch, col_ex = st.columns([1, 1])

with col_arch:
    st.markdown("### 🏗️ System Architecture")
    st.markdown(
        """
        ```
        ┌──────────────────────────────────────────┐
        │           WINDOWS HOST                   │
        │  Ollama Runtime  (0.0.0.0:11434)         │
        │  ├── llama3.2:1b      1.2B  Q8_0         │
        │  ├── starcoder2:3b    3B    Q4_0         │
        │  └── qwen2.5-coder    1.5B  Q4_K_M       │
        └─────────────┬────────────────────────────┘
                      │ HTTP /api/generate
        ┌─────────────▼────────────────────────────┐
        │    VIRTUALBOX UBUNTU — Docker Compose     │
        │  ┌──────────────────────────────────┐     │
        │  │   rag-app  (:8501)               │     │
        │  │   • Streamlit  • RAG Pipeline    │     │
        │  │   • FastEmbed  • SQLite          │     │
        │  └──────────────┬───────────────────┘     │
        │                 │ :6333                    │
        │  ┌──────────────▼───────────────────┐     │
        │  │   rag-qdrant (241 chunks / 5 docs)│     │
        │  └──────────────────────────────────┘     │
        └──────────────────────────────────────────┘
        ```
        """
    )

with col_ex:
    st.markdown("### 📋 Evaluation Suite Overview")
    st.markdown(
        """
        | Exercise | Focus |
        |:---:|---|
        | 1 | Multi-LLM comparison (3 models, identical conditions) |
        | 2 | 25-task representative evaluation dataset |
        | 3 | Token F1, Keyword Recall, Relevance metrics |
        | 4 | Hallucination detection & code test-pass rate |
        | 5 | RAG pipeline trace analysis (5 diagnostic cases) |
        | 6 | Cross-file repository understanding investigation |

        **Tech Stack**
        - **Embedder:** BAAI/bge-small-en-v1.5 (384 dim, CPU)
        - **Vector DB:** Qdrant (cosine similarity, Docker)
        - **Document Parser:** PyMuPDF (fitz)
        - **App Framework:** Streamlit
        - **LLM Runtime:** Ollama (Windows host via NAT 10.0.2.2)
        """
    )

st.markdown("---")
st.caption(
    "Local RAG System | LLMs: llama3.2:1b · starcoder2:3b · qwen2.5-coder:1.5b (Windows Host Ollama) "
    "| App + Qdrant: Dockerized Ubuntu VirtualBox"
)
