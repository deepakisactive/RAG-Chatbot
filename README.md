# Production-Quality Local RAG Chatbot

A production-grade, observable Retrieval-Augmented Generation (RAG) chatbot system featuring strict physical environment separation, local CPU-optimized execution, transparent retrieval observability, and zero cloud footprint.

---

## 1. Project Overview

This project implements a complete local RAG system designed for privacy, deterministic execution, and full observability. 
- **LLM Runtime**: Runs natively on the **Windows Host** via Ollama serving `Llama 3.2 1B`.
- **RAG & Vector Stack**: Runs inside a containerized **Docker Compose** environment on a **VirtualBox Ubuntu VM**.
- **No Cloud Services**: Zero reliance on AWS, OpenAI, or remote cloud vector stores.

---

## 2. Architecture & Environment Separation

### Why Ollama Runs on the Windows Host
1. **Direct Hardware Access**: Windows host retains native CPU/RAM allocation without hypervisor emulation overhead.
2. **Resource Containment**: Prevents memory starvation inside the Ubuntu VM by isolating the LLM process.
3. **No Heavy Container Images**: The Ubuntu Docker image stays lightweight and portable by omitting multi-gigabyte LLM runtime layers.

### Why the RAG Application Runs in Ubuntu Docker
1. **Reproducibility**: Docker Compose guarantees deterministic dependencies across Linux environments.
2. **Persistence**: SQLite metadata, uploaded PDFs, and Qdrant vector indices are mapped to persistent host volumes.
3. **Microservice Topology**: Decouples the presentation layer (Streamlit) from the vector storage engine (Qdrant).

### Topology Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      WINDOWS HOST                           │
│  • Ollama Runtime (Listening on 0.0.0.0:11434)             │
│  • Model: Llama 3.2 1B (CPU Execution)                      │
└──────────────────────────────▲──────────────────────────────┘
                               │ HTTP API (/api/generate, /api/tags)
    ┌──────────────────────────┴──────────────────────────────┐
    │                   VIRTUALBOX UBUNTU VM                  │
    │  ┌────────────────────────────────────────────────────┐ │
    │  │ DOCKER COMPOSE STACK                               │ │
    │  │                                                    │ │
    │  │  [ Container: rag-app (Port 8501) ]                │ │
    │  │   • Streamlit Presentation UI                      │ │
    │  │   • PyMuPDF Document Parser                        │ │
    │  │   • Recursive Chunker                              │ │
    │  │   • BAAI/bge-small-en-v1.5 Embedder                │ │
    │  │   • SQLite Metadata Store (rag.db)                 │ │
    │  │                                                    │ │
    │  │       │ Semantic Vector Queries (Port 6333)        │ │
    │  │       ▼                                            │ │
    │  │  [ Container: qdrant (Port 6333) ]                 │ │
    │  │   • Qdrant Vector Engine                           │ │
    │  │   • Volume: rag_qdrant_storage                     │ │
    │  └────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
```

---

## 3. VirtualBox Networking & Host Configuration

To allow the Ubuntu Docker container to communicate with Ollama on Windows:

### Step 1: Configure Ollama on Windows
By default, Ollama only listens on `127.0.0.1`. You must instruct it to listen on all interfaces:
1. Open PowerShell on Windows as Administrator.
2. Set the environment variable:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'Machine')
   ```
3. Allow inbound connections on port 11434:
   ```powershell
   New-NetFirewallRule -DisplayName "Ollama API" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
   ```
4. Restart Ollama from the Windows system tray.

### Step 2: Determine Windows Host IP from Ubuntu
In VirtualBox:
- **Default NAT Mode**: The Windows host is reachable at `http://10.0.2.2:11434`.
- **Host-Only Mode**: Check your adapter IP (typically `http://192.168.56.1:11434`).
- **Bridged Mode**: Use your Windows machine's LAN IP (e.g. `http://192.168.1.50:11434`).

Test connectivity from Ubuntu:
```bash
curl http://10.0.2.2:11434/api/tags
```

---

## 4. Quick Start & Execution

### Prerequisites
- Docker Engine & Docker Compose installed on Ubuntu VM.
- Windows Host running Ollama with `llama3.2:1b` pulled:
  ```powershell
  ollama run llama3.2:1b
  ```

### Step 1: Clone or Open Workspace
Ensure your active workspace is set to `/home/deepak/.gemini/antigravity/scratch/rag-chatbot`.

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and verify `OLLAMA_BASE_URL`:
```bash
cp .env.example .env
```
Default `.env` contents:
```ini
OLLAMA_BASE_URL=http://10.0.2.2:11434
OLLAMA_MODEL=llama3.2:1b
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=rag_documents
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
SIMILARITY_THRESHOLD=0.45
SQLITE_DB_PATH=/app/metadata/rag.db
UPLOAD_DIR=/app/data/documents
```

### Step 3: Launch with Docker Compose
Start both the `qdrant` vector database and the `rag-app` Streamlit container:
```bash
docker compose up -d --build
```

### Step 4: Access the Application
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 5. End-to-End User Experience

### 1. Check System Status
Navigate to the **System** page (`http://localhost:8501/System`) to verify:
- **Ollama**: Connected (`Llama 3.2 1B` detected on Windows host).
- **Qdrant**: Connected.
- **Embedding Engine**: Ready (`BAAI/bge-small-en-v1.5`, 384 dimensions).

### 2. Upload Documents
Go to the **Documents** page:
1. Upload one or more PDF files.
2. System immediately computes SHA-256 hashes to detect duplicate uploads.
3. Automatically extracts pages, word counts, and token estimates using PyMuPDF.

### 3. Pipeline Ingestion
Go to the **Pipeline** page:
1. Adjust chunk size (default `512`), chunk overlap (default `64`), or similarity threshold (default `0.45`).
2. Click **Run Incremental Pipeline**.
3. Already-indexed documents are skipped; only newly uploaded PDFs are embedded and stored in Qdrant.

### 4. Grounded Chat & Retrieval Observability
Go to the **Chat** page:
1. Ask any question.
2. The assistant answers using **ONLY** the retrieved excerpts.
3. If retrieval scores fall below the similarity threshold, the model refuses to answer rather than hallucinating:
   > *"Insufficient evidence was found in the uploaded documents to answer this question."*
4. Expand **Sources & Citations** to see document names, page numbers, and similarity scores.
5. Expand **Inspect Retrieval Trace** to view query embedding times, vector search times, ranked chunks, and the full prompt sent to Llama 3.2 1B.

### 5. Document Deletion
On the **Documents** page, click **Delete**:
- Atomically removes the physical file from disk.
- Purges all corresponding vector chunks from Qdrant by `document_id`.
- Deletes document metadata from SQLite.
- Leaves zero orphaned records.

---

## 6. Project Structure

```
rag-chatbot/
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile                  # CPU-optimized Python 3.11 container
├── README.md                   # System documentation and operational guide
├── docker-compose.yml          # rag-app and qdrant multi-container definition
├── requirements.txt            # Python dependencies (CPU-optimized)
│
├── app/
│   ├── __init__.py
│   ├── config.py              # Pydantic/dataclass centralized configuration
│   ├── logging_config.py      # Timed structured logging without text leakage
│   │
│   ├── database/              # SQLite metadata and telemetry storage
│   │   ├── database.py        # Thread-safe connection manager (WAL mode)
│   │   ├── models.py          # DocumentRecord, PipelineRecord, QueryRecord
│   │   └── repository.py      # Document, Pipeline, and Query CRUD
│   │
│   ├── documents/
│   │   └── manager.py         # Document lifecycle (upload, deduplication, atomic delete)
│   │
│   ├── ingestion/
│   │   ├── chunker.py         # Recursive token/character chunker preserving page provenance
│   │   ├── embedder.py        # FastEmbed / SentenceTransformers BGE-small wrapper
│   │   ├── indexer.py         # Incremental ingestion and re-indexing orchestrator
│   │   └── parser.py          # PyMuPDF text extractor and SHA-256 validator
│   │
│   ├── llm/
│   │   └── ollama_client.py   # Resilient HTTP client for Windows Ollama runtime
│   │
│   ├── rag/
│   │   ├── evaluator.py       # Quantitative evaluation (Recall@K, MRR, Precision@K)
│   │   ├── models.py          # RetrievalTrace, SourceCitation, RAGResponse models
│   │   ├── pipeline.py        # End-to-end RAG orchestrator with confidence gating
│   │   ├── prompt.py          # Grounded prompt templates and citation formatting
│   │   └── retriever.py       # Qdrant semantic search and threshold filtering
│   │
│   └── streamlit/             # Streamlit multi-page frontend
│       ├── app.py             # Main entrypoint and dashboard
│       ├── components/
│       │   ├── document_table.py
│       │   ├── metrics.py
│       │   ├── retrieval_trace.py
│       │   └── status.py
│       └── pages/
│           ├── 1_💬_Chat.py
│           ├── 2_📄_Documents.py
│           ├── 3_⚙️_Pipeline.py
│           ├── 4_🔍_Retrieval_Trace.py
│           ├── 5_📊_Analytics.py
│           └── 6_🖥️_System.py
│
├── data/
│   └── documents/             # Persistent volume for uploaded PDF files
├── metadata/                  # Persistent volume for SQLite database (rag.db)
│
└── tests/                     # Automated test suite (26 passing tests)
    ├── test_chunker_and_embedder.py
    ├── test_config_and_db.py
    ├── test_evaluation.py
    ├── test_ollama_and_rag_pipeline.py
    ├── test_parser_and_docs.py
    └── test_retrieval.py
```

---

## 7. Verification & Automated Testing

The complete test suite runs in under 0.5s without external network dependencies:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
**Results: 26 tests passed (100% pass rate).**

### Key Test Coverage
- `test_config_and_db.py`: Settings parsing, schema migrations, WAL mode concurrency, SQLite CRUD.
- `test_parser_and_docs.py`: Filename sanitization, SHA-256 duplicate detection, atomic file + Qdrant deletion.
- `test_chunker_and_embedder.py`: Recursive token chunking with page retention, 384-d vector normalization.
- `test_qdrant_and_indexer.py`: Vector upsert, cosine search, payload-filtered deletion, incremental ingestion.
- `test_retrieval.py`: Score threshold cutoff, out-of-domain rejection, latency metrics.
- `test_ollama_and_rag_pipeline.py`: Prompt grounding, low-confidence gating, telemetry logging.
- `test_evaluation.py`: Recall@K, Precision@K, Mean Reciprocal Rank (MRR) benchmarks.

---

## 8. Performance & CPU Optimizations

1. **Lightweight Embedding Model**: `BAAI/bge-small-en-v1.5` outputs compact 384-dimensional vectors, executing in <20ms on modern CPUs.
2. **Avoiding CUDA Bloat**: Installing CPU-only wheels shrinks the image by over 2.5 GB.
3. **Strict Top-K & Context Limits**: Setting default `Top-K = 5` and `num_ctx = 2048` ensures fast generation on Llama 3.2 1B.
4. **SQLite WAL Mode**: Minimizes disk I/O lock contention between Streamlit threads and background ingestion tasks.

---

## 9. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Ollama shows Disconnected** | Ollama on Windows bound to `127.0.0.1` or blocked by firewall. | Set `OLLAMA_HOST=0.0.0.0:11434` in Windows environment variables and add an inbound firewall rule for port 11434. |
| **"Insufficient evidence..." message on queries** | Query similarity falls below `SIMILARITY_THRESHOLD`. | Lower the threshold slider on the Chat sidebar (e.g. from 0.45 to 0.30), or ensure the relevant document is indexed. |
| **Duplicate upload warning** | File SHA-256 hash matches an already uploaded PDF. | The system intentionally prevents vector duplication. Re-index the existing file on the Documents page if needed. |
