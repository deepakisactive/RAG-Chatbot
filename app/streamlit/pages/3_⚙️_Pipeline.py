"""Streamlit Ingestion Pipeline Configuration & Execution Page."""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
import streamlit as st

from app.config import settings
from app.database.database import get_db
from app.database.models import PipelineRecord
from app.database.repository import PipelineRepository
from app.ingestion.embedder import get_embedder
from app.ingestion.indexer import DocumentIndexer
from app.streamlit.components.status import render_sidebar_status

st.set_page_config(page_title="RAG Pipeline Management", page_icon="⚙️", layout="wide")

render_sidebar_status()

st.title("⚙️ RAG Ingestion Pipeline")
st.markdown("Configure chunking strategies, embedding hyperparameters, and trigger pipeline execution.")

embedder = get_embedder()
indexer = DocumentIndexer(db_manager=get_db())
pipe_repo = PipelineRepository(db=get_db())

st.markdown("### 🔧 Pipeline Configuration")

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    st.markdown("#### 📑 Document Parsing & Chunking")
    st.text_input("Document Parser", value="PyMuPDF (fitz)", disabled=True)
    strategy = st.selectbox("Chunking Strategy", options=["recursive", "token_sliding_window"], index=0)
    chunk_size = st.slider("Chunk Size (Tokens)", min_value=128, max_value=1024, value=settings.CHUNK_SIZE, step=64)
    chunk_overlap = st.slider("Chunk Overlap (Tokens)", min_value=0, max_value=256, value=settings.CHUNK_OVERLAP, step=16)

with col_cfg2:
    st.markdown("#### 🧠 Embedding & Vector Retrieval")
    st.text_input("Embedding Model", value=embedder.model_name, disabled=True)
    c_dim, c_metric = st.columns(2)
    c_dim.text_input("Embedding Dimension", value=str(embedder.dimension), disabled=True)
    c_metric.text_input("Distance Metric", value=settings.DISTANCE_METRIC, disabled=True)
    top_k = st.slider("Default Top-K", min_value=1, max_value=20, value=settings.TOP_K, step=1)
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=float(settings.SIMILARITY_THRESHOLD), step=0.05)

st.markdown("---")

col_act1, col_act2 = st.columns(2)

with col_act1:
    st.markdown("#### ⚡ Incremental Ingestion")
    st.caption("Only indexes documents that have not yet been indexed. Preserves existing vector embeddings.")
    if st.button("▶️ Run Incremental Pipeline", type="primary"):
        with st.status("Executing Incremental Pipeline...", expanded=True) as status:
            st.write("🔍 Discovering unindexed documents...")
            time.sleep(0.2)
            results = indexer.index_all_unindexed(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.write(f"✓ Found {results['total_discovered']} total documents ({results['unindexed_found']} unindexed).")
            st.write(f"✓ Processed {results['processed']} new document(s).")
            st.write(f"✓ Generated and stored {results['total_chunks']} vector chunks in Qdrant.")
            status.update(label="Incremental Ingestion Complete!", state="complete", expanded=False)

        st.success(f"Ingested {results['total_chunks']} chunks in {results['duration_ms']:.1f} ms.")
        if results["errors"]:
            st.error(f"Encountered errors: {results['errors']}")

with col_act2:
    st.markdown("#### 🔄 Full Pipeline Re-indexing")
    st.caption("Purges existing vectors and re-indexes all documents under a new pipeline version.")
    new_version_tag = st.text_input("New Pipeline Version Tag", value=f"v{int(time.time())}")

    if st.button("⚠️ Execute Full Re-indexing", type="secondary"):
        with st.status("Executing Full Pipeline Re-indexing...", expanded=True) as status:
            st.write("🧹 Purging old vectors & parsing documents...")
            # Save pipeline configuration to database
            pipe_record = PipelineRecord(
                pipeline_id=new_version_tag,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=strategy,
                embedding_model=embedder.model_name,
                embedding_dimension=embedder.dimension,
                distance_metric=settings.DISTANCE_METRIC,
                top_k=top_k,
                similarity_threshold=threshold,
            )
            pipe_repo.insert(pipe_record)

            results = indexer.reindex_all(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pipeline_version=new_version_tag,
            )
            st.write(f"✓ Discovered {results['total_documents']} documents.")
            st.write(f"✓ Re-indexed {results['processed']} documents.")
            st.write(f"✓ Created {results['total_chunks']} chunks under version `{new_version_tag}`.")
            status.update(label="Full Re-indexing Complete!", state="complete", expanded=False)

        st.success(f"Completed re-indexing {results['processed']} docs ({results['total_chunks']} chunks) in {results['duration_ms']:.1f} ms.")

st.markdown("---")
st.markdown("### 📜 Registered Pipeline Configurations")
pipelines = pipe_repo.list_all()
if pipelines:
    st.dataframe([p.to_dict() for p in pipelines], use_container_width=True)
else:
    st.caption("No custom pipeline executions registered yet.")
