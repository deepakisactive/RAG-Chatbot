"""Index codebase and architectural documentation into Qdrant vector database.

Enables the RAG chatbot to answer repository-level questions, code explanations,
and architectural queries (Exercise 6) in addition to research papers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database.database import get_db
from app.documents.manager import DocumentManager
from app.ingestion.indexer import DocumentIndexer
from app.logging_config import logger

CODEBASE_FILES = [
    "README.md",
    "app/config.py",
    "app/rag/pipeline.py",
    "app/rag/retriever.py",
    "app/rag/prompt.py",
    "app/llm/ollama_client.py",
    "app/ingestion/parser.py",
    "app/ingestion/chunker.py",
    "app/ingestion/embedder.py",
    "app/ingestion/indexer.py",
    "app/vectorstore/qdrant_client.py",
    "app/database/database.py",
    "app/database/models.py",
    "app/database/repository.py",
    "app/documents/manager.py",
]


def index_codebase() -> None:
    doc_mgr = DocumentManager(db_manager=get_db())
    indexer = DocumentIndexer(db_manager=get_db())

    indexed_count = 0
    total_chunks = 0

    print("🚀 Starting codebase indexing into RAG vector database...")

    for rel_path in CODEBASE_FILES:
        file_path = _project_root / rel_path
        if not file_path.exists():
            print(f"⚠️ Skipping missing file: {rel_path}")
            continue

        filename = rel_path.replace("/", "_")  # e.g., app_rag_pipeline.py
        file_bytes = file_path.read_bytes()

        try:
            record, is_new, msg = doc_mgr.upload_document(
                file_bytes=file_bytes,
                filename=filename,
            )

            # Force re-indexing or index if new
            success, count, index_msg = indexer.index_document(
                document_id=record.document_id,
                force_reindex=True,
            )

            if success:
                indexed_count += 1
                total_chunks += count
                print(f"  ✅ Indexed: {rel_path} ({count} chunks)")
            else:
                print(f"  ❌ Indexing failed for {rel_path}: {index_msg}")

        except Exception as exc:
            print(f"  ❌ Error processing {rel_path}: {exc}")

    print(f"\n🎉 Codebase indexing complete! Indexed {indexed_count} files ({total_chunks} chunks).")


if __name__ == "__main__":
    index_codebase()
