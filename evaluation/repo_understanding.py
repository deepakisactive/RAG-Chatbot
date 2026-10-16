"""Exercise 6: Repository-Level Codebase Understanding Investigation.

Evaluates whether the current LLM + RAG system can answer questions that
require understanding MULTIPLE FILES, MODULES, OR COMPONENTS across the repository.

Investigates 5 representative repository-level architectural questions:
1. Multi-component query processing flow across files.
2. Component caller-callee relationships (embedder and vector store).
3. Lifecycle of user interactions (PDF upload to vector indexing).
4. Impact analysis: ripple effects of changing configuration (EMBEDDING_DIMENSION 384 -> 768).
5. Test traceability: mapping functionality to corresponding test cases.

Contrasts current chunk-based RAG with next week's Sourcegraph / Semantic Code Graph paradigms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.llm.ollama_client import get_ollama_client
from app.rag.pipeline import get_pipeline

REPO_UNDERSTANDING_RESULTS_FILE = (
    _project_root / "evaluation" / "results" / "repo_understanding_results.json"
)

REPO_QUESTIONS = [
    {
        "id": "REPO-Q1",
        "question": "Which files and modules in this repository are involved in user query handling, retrieval, generation, and telemetry logging?",
        "ground_truth_components": [
            "app/streamlit/pages/1_💬_Chat.py (UI entrypoint)",
            "app/rag/pipeline.py (orchestrator: RAGPipeline.query)",
            "app/rag/retriever.py (RetrievalEngine.retrieve)",
            "app/ingestion/embedder.py (EmbeddingEngine.embed_query)",
            "app/vectorstore/qdrant_client.py (VectorStoreClient.search_vectors)",
            "app/rag/prompt.py (PromptBuilder.build_prompt)",
            "app/llm/ollama_client.py (OllamaClient.generate)",
            "app/database/repository.py (QueryRepository.insert)",
            "app/database/models.py (QueryRecord dataclass)",
            "metadata/rag.db (queries SQLite table)",
        ],
        "multi_file_scope": 10,
        "category": "multi_file_interaction",
    },
    {
        "id": "REPO-Q2",
        "question": "Which function calls the embedding model and vector store client during the document ingestion and indexing pipeline?",
        "ground_truth_components": [
            "app/ingestion/indexer.py (DocumentIndexer.index_document)",
            "Calls EmbeddingEngine.embed_texts() in app/ingestion/embedder.py",
            "Calls VectorStoreClient.upsert_chunks() in app/vectorstore/qdrant_client.py",
            "Invoked by DocumentManager in app/documents/manager.py",
        ],
        "multi_file_scope": 4,
        "category": "call_graph_dependency",
    },
    {
        "id": "REPO-Q3",
        "question": "What happens after a user uploads a PDF in the Streamlit UI? Trace the complete lifecycle through all files.",
        "ground_truth_components": [
            "1. Upload triggered in app/streamlit/pages/2_📄_Documents.py",
            "2. Saved to data/documents/ via DocumentManager in app/documents/manager.py",
            "3. SHA-256 hash calculated in app/ingestion/parser.py (PDFParser.calculate_hash)",
            "4. Checked for duplicates against SQLite 'documents' table in app/database/repository.py",
            "5. PyMuPDF extracts page text into ParsedPDF",
            "6. RecursiveCharacterChunker in app/ingestion/chunker.py creates overlapping TextChunk objects",
            "7. FastEmbed ONNX generates 384D float vectors in app/ingestion/embedder.py",
            "8. VectorStoreClient upserts points into Qdrant in app/vectorstore/qdrant_client.py",
            "9. DocumentRecord saved with status='INDEXED' in metadata/rag.db",
        ],
        "multi_file_scope": 8,
        "category": "lifecycle_tracing",
    },
    {
        "id": "REPO-Q4",
        "question": "If EMBEDDING_DIMENSION is modified from 384 to 768 in app/config.py, which components, database tables, or vector collections will be affected or broken?",
        "ground_truth_components": [
            "1. app/config.py: Settings.EMBEDDING_DIMENSION",
            "2. app/ingestion/embedder.py: BAAI/bge-small-en-v1.5 produces 384D vectors, so changing to 768 requires switching models (e.g. bge-base or bge-large)",
            "3. app/vectorstore/qdrant_client.py: Qdrant collection vector size is fixed at creation time; existing collection must be dropped and recreated",
            "4. metadata/rag.db: 'pipelines' table records embedding_dimension",
            "5. Runtime failure: attempting to insert or query 768D vectors into a 384D Qdrant collection raises vector dimension mismatch error",
        ],
        "multi_file_scope": 5,
        "category": "impact_analysis",
    },
    {
        "id": "REPO-Q5",
        "question": "Which test files in the tests/ directory validate the vector similarity search, chunking, and RAG retrieval functionality?",
        "ground_truth_components": [
            "tests/test_retrieval.py (validates RetrievalEngine, threshold gating, cosine search)",
            "tests/test_chunker_and_embedder.py (validates RecursiveCharacterChunker and FastEmbed/MockEmbedder)",
            "tests/test_qdrant_and_indexer.py (validates in-memory Qdrant upsert and document indexer)",
            "tests/test_ollama_and_rag_pipeline.py (validates end-to-end RAGPipeline.query flow)",
            "tests/test_evaluation.py (validates Recall@K, Precision@K, and MRR evaluator)",
        ],
        "multi_file_scope": 5,
        "category": "test_traceability",
    },
]


def run_repo_understanding_investigation() -> Dict[str, Any]:
    """Execute repository understanding evaluation across the 3 models."""
    from app.llm.ollama_client import OllamaClient
    ollama_client = OllamaClient(timeout=300.0)  # 5 min — model swap is slow on CPU
    models = ["llama3.2:1b", "starcoder2:3b", "qwen2.5-coder:1.5b"]

    print("\n🏗️ Running Exercise 6: Repository-Level Codebase Understanding Investigation...")
    results: List[Dict[str, Any]] = []

    # Read relevant codebase files — trimmed to 400 chars each to minimize token count
    code_excerpts = []
    key_files = [
        "app/rag/pipeline.py",
        "app/rag/retriever.py",
        "app/ingestion/indexer.py",
        "app/documents/manager.py",
        "app/config.py",
    ]
    for rel_path in key_files:
        full_p = _project_root / rel_path
        if full_p.exists():
            code_excerpts.append(f"# FILE: {rel_path}\n" + full_p.read_text(encoding="utf-8")[:400] + "\n...")

    codebase_context = "\n\n".join(code_excerpts)

    for item in REPO_QUESTIONS:
        print(f"\n--- Investigating: {item['id']} ---")
        q_text = item["question"]
        model_answers: Dict[str, Any] = {}

        for m in models:
            print(f"  Model: {m}")
            prompt = (
                f"Python repo context:\n{codebase_context}\n\n"
                f"Q: {q_text}\n"
                f"A (name exact files/functions concisely):"
            )
            try:
                ans, latency = ollama_client.generate(prompt=prompt, model=m)
            except Exception as exc:
                print(f"  ⚠️ {m} timed out or failed: {exc}")
                ans = f"[TIMEOUT/ERROR — model did not respond within 300s: {exc}]"
                latency = 300000.0

            # Measure how many ground truth components were identified
            found_count = 0
            for comp in item["ground_truth_components"]:
                # Check for key tokens in component string
                tokens = [t.strip().lower() for t in comp.split() if len(t) > 4 and not t.startswith("(")]
                if any(tok in ans.lower() for tok in tokens):
                    found_count += 1
            component_recall = round(found_count / len(item["ground_truth_components"]), 4)

            model_answers[m] = {
                "answer": ans,
                "latency_ms": latency,
                "components_identified": found_count,
                "total_components": len(item["ground_truth_components"]),
                "component_recall": component_recall,
            }

        results.append({
            "id": item["id"],
            "question": q_text,
            "category": item["category"],
            "ground_truth_components": item["ground_truth_components"],
            "multi_file_scope": item["multi_file_scope"],
            "model_evaluations": model_answers,
        })

    # Structural Limitations Analysis
    limitations_analysis = {
        "limitations_of_naive_chunk_rag": [
            {
                "limitation": "Chunk Boundary Fragmentation",
                "description": "Standard fixed-size chunking (e.g. 512 tokens with 64 overlap) slices through class definitions, functions, and cross-file import statements, breaking syntactic integrity.",
            },
            {
                "limitation": "Absence of Abstract Syntax Tree (AST) Graphs",
                "description": "Text embedding models evaluate lexical and surface semantic similarity rather than formal AST hierarchies (parent classes, method declarations, lexical scopes).",
            },
            {
                "limitation": "Inability to Traverse Cross-File Call Graphs",
                "description": "Questions like 'Which function calls X?' or 'Trace user upload through all files' require following directional call edges across 8-10 distinct files. Basic vector retrieval retrieves disjoint snippets without edge connectivity.",
            },
            {
                "limitation": "Global Ripple Effect Blindness",
                "description": "Configuration changes (e.g. EMBEDDING_DIMENSION 384 -> 768) propagate across multiple layers (DB schemas, model weights, vector index constraints). Text embeddings cannot infer database-to-vector-index type constraints.",
            },
        ],
        "sourcegraph_and_code_graph_advantages": [
            {
                "advantage": "Precise Symbol Navigation (SCIP / LSIF)",
                "description": "Sourcegraph uses compiler-backed indexing (SCIP/LSIF) to provide 100% deterministic 'Find Definitions' and 'Find References' across thousands of files.",
            },
            {
                "advantage": "Cross-Repository Semantic Search",
                "description": "Combines structural pattern matching (AST regex) with semantic embeddings for syntax-aware code search.",
            },
            {
                "advantage": "Repository-Scale Dependency Graphing",
                "description": "Exposes explicit call graphs and dependency graphs to LLMs, replacing random top-K vector search with deterministic code path traversal.",
            },
        ],
    }

    full_output = {
        "meta": {
            "title": "Exercise 6: Repository-Level Codebase Understanding Investigation",
            "models_compared": models,
        },
        "investigations": results,
        "architectural_analysis": limitations_analysis,
    }

    REPO_UNDERSTANDING_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPO_UNDERSTANDING_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"✅ Repository understanding investigation complete! Saved to {REPO_UNDERSTANDING_RESULTS_FILE}")
    return full_output


if __name__ == "__main__":
    run_repo_understanding_investigation()

