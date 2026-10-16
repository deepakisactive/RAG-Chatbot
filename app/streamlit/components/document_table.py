"""Document Table and Action Component."""

from __future__ import annotations

import streamlit as st

from app.database.models import DocumentRecord
from app.documents.manager import DocumentManager
from app.ingestion.indexer import DocumentIndexer
from app.vectorstore.qdrant_client import get_vectorstore


def render_document_card(doc: DocumentRecord, doc_mgr: DocumentManager, indexer: DocumentIndexer) -> None:
    """Render an individual document card with metadata and actions."""
    size_mb = doc.file_size / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB" if size_mb >= 0.01 else f"{doc.file_size / 1024:.1f} KB"

    status_color = {
        "INDEXED": "green",
        "PROCESSING": "blue",
        "FAILED": "red",
    }.get(doc.status, "gray")

    with st.container():
        col_meta, col_actions = st.columns([3, 1])

        with col_meta:
            st.markdown(f"#### 📄 `{doc.filename}`")
            st.markdown(
                f"**Pages:** {doc.page_count} &nbsp;|&nbsp; "
                f"**Size:** {size_str} &nbsp;|&nbsp; "
                f"**Chunks:** {doc.chunk_count} &nbsp;|&nbsp; "
                f"**Status:** :{status_color}[**{doc.status}**] &nbsp;|&nbsp; "
                f"**Pipeline:** `{doc.pipeline_version}`"
            )
            st.caption(
                f"ID: `{doc.document_id}` | Uploaded: {doc.upload_date} | Model: `{doc.embedding_model}`"
            )

        with col_actions:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Index", key=f"reindex_{doc.document_id}", help="Re-index this document"):
                    with st.spinner(f"Re-indexing {doc.filename}..."):
                        success, count, msg = indexer.index_document(doc.document_id, force_reindex=True)
                        if success:
                            st.success(f"Indexed {count} chunks!")
                            st.rerun()
                        else:
                            st.error(msg)

            with c2:
                if st.button("🗑️ Delete", key=f"del_{doc.document_id}", help="Delete document and purge vectors"):
                    vstore = get_vectorstore()
                    success, msg = doc_mgr.delete_document(doc.document_id, vectorstore_client=vstore)
                    if success:
                        st.success(f"Deleted {doc.filename}")
                        st.rerun()
                    else:
                        st.error(msg)

        with st.expander(f"Inspect Metadata for `{doc.filename}`", expanded=False):
            st.json(doc.to_dict())

        st.markdown("---")
