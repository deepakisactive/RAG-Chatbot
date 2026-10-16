"""Streamlit Documents Management Page."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.database.database import get_db
from app.documents.manager import DocumentManager
from app.ingestion.indexer import DocumentIndexer
from app.streamlit.components.document_table import render_document_card
from app.streamlit.components.status import render_sidebar_status

st.set_page_config(page_title="Document Management", page_icon="📄", layout="wide")

render_sidebar_status()

st.title("📄 Document Management")
st.markdown(
    "Upload PDF documents for ingestion. Files are checked against existing SHA-256 hashes "
    "to prevent duplicate indexing."
)

doc_mgr = DocumentManager(db_manager=get_db())
indexer = DocumentIndexer(db_manager=get_db())

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload PDF, Python, Markdown, or Text documents",
    type=["pdf", "py", "md", "txt"],
    accept_multiple_files=True,
    help="Supported formats: PDF, Python (.py), Markdown (.md), and plain text (.txt).",
)

if uploaded_files:
    if st.button("🚀 Process & Ingest Uploaded Files", type="primary"):
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for idx, file in enumerate(uploaded_files):
            file_bytes = file.read()
            with st.spinner(f"Ingesting '{file.name}'..."):
                try:
                    record, is_new, msg = doc_mgr.upload_document(
                        file_bytes=file_bytes,
                        filename=file.name,
                    )
                    if is_new:
                        if record.status == "FAILED":
                            st.error(f"❌ {file.name}: {msg}")
                        else:
                            # Automatically trigger indexing for newly uploaded document
                            success, count, index_msg = indexer.index_document(record.document_id)
                            if success:
                                st.success(f"✅ Successfully ingested & indexed '{file.name}' ({count} chunks).")
                            else:
                                st.error(f"❌ Indexing failed for '{file.name}': {index_msg}")
                    else:
                        st.info(f"ℹ️ {msg}")
                except Exception as exc:
                    st.error(f"Failed to process '{file.name}': {exc}")

            progress_bar.progress((idx + 1) / total_files)

        st.rerun()

st.markdown("---")
st.markdown("### 📚 Tracked Documents")

documents = doc_mgr.list_documents()

if not documents:
    st.info("No documents uploaded yet. Use the uploader above to add PDF documents.")
else:
    # Summary metrics
    stats = doc_mgr.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed Documents", stats["total_documents"])
    c2.metric("Total Pages", f"{stats['total_pages']:,}")
    c3.metric("Total Chunks", f"{stats['total_chunks']:,}")
    total_mb = stats["total_bytes"] / (1024 * 1024)
    c4.metric("Storage Used", f"{total_mb:.2f} MB")

    st.markdown("---")

    for doc in documents:
        render_document_card(doc=doc, doc_mgr=doc_mgr, indexer=indexer)
