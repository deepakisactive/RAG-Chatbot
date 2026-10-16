"""System Status Component for Streamlit."""

from __future__ import annotations

import streamlit as st

from app.config import settings
from app.database.database import get_db
from app.database.repository import DocumentRepository
from app.ingestion.embedder import get_embedder
from app.llm.ollama_client import get_ollama_client
from app.vectorstore.qdrant_client import get_vectorstore


def render_sidebar_status() -> None:
    """Render a concise system connectivity status card in the sidebar."""
    st.sidebar.markdown("### 🔌 System Health")

    # 1. Ollama on Windows Host
    ollama_client = get_ollama_client()
    ollama_info = ollama_client.health_check()
    if ollama_info.get("status") == "connected":
        st.sidebar.success(f"**Ollama:** Connected\n*Model:* `{ollama_info.get('model')}`")
    else:
        st.sidebar.warning(f"**Ollama:** Disconnected\n*(Host: {settings.OLLAMA_BASE_URL})*")

    # 2. Qdrant Vector Store
    vectorstore = get_vectorstore()
    qdrant_info = vectorstore.health_check()
    if qdrant_info.get("status") == "connected":
        vec_count = qdrant_info.get("vector_count", 0)
        st.sidebar.success(f"**Qdrant:** Ready ({vec_count:,} vectors)")
    else:
        st.sidebar.error("**Qdrant:** Offline")

    # 3. Embedder
    embedder = get_embedder()
    st.sidebar.info(f"**Embedder:** `{embedder.model_name.split('/')[-1]}` ({embedder.dimension}d)")

    # 4. Document Index Status
    doc_repo = DocumentRepository(db=get_db())
    stats = doc_repo.get_stats()
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"📚 **Indexed:** {stats['total_documents']} docs | {stats['total_chunks']:,} chunks"
    )
