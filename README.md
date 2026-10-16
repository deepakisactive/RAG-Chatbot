# 🛡️Local RAG Chatbot & Guardrail Verification Suite

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant%20v1.11-red.svg?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/LLM%20Runtime-Ollama%20(Local)-white.svg?logo=ollama&logoColor=black)](https://ollama.ai/)
[![Tests](https://img.shields.io/badge/Pytest-64%2F64%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![Guardrails](https://img.shields.io/badge/Guardrails-22%2F22%20Verified%20(100%25)-success.svg)](evaluation/)
[![Privacy](https://img.shields.io/badge/Data%20Privacy-100%25%20Local%20%2F%20Zero%20Cloud-blueviolet.svg)]()

A production-grade, observable, and fully local **Retrieval-Augmented Generation (RAG)** chatbot and model evaluation system. Built with strict physical environment separation, local CPU-optimized vector search, transparent step-by-step retrieval observability, a **3-layer defense-in-depth guardrail framework (13 active rules)**, multi-model evaluation benchmarks across **Llama 3.2 1B**, **Qwen 2.5 Coder 1.5B**, and **StarCoder2 3B**, and zero cloud footprint.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture & Topology](#-system-architecture--topology)
- [Multi-Model Support](#-multi-model-support)
- [Unified Chat Page](#-unified-chat-page)
- [3-Layer Defense-in-Depth Guardrail System](#-3-layer-defense-in-depth-guardrail-system)
  - [Active Rules Engine](#active-rules-engine-13-rules)
  - [Before vs. After Guardrails Demonstration](#before-vs-after-guardrails-demonstration)
- [Model Evaluation & Benchmarking Workbench](#-model-evaluation--benchmarking-workbench)
  - [Unified Multi-Model Comparison Chart](#unified-multi-model-comparison-chart)
  - [Evaluation Dataset & Benchmarks](#evaluation-dataset--benchmarks)
- [Multi-Page Application Walkthrough](#-multi-page-application-walkthrough)
- [Quick Start Guide](#-quick-start-guide)
  - [Step 1: Configure Ollama on Windows Host](#step-1-configure-ollama-on-windows-host)
  - [Step 2: Start Application with Docker Compose](#step-2-start-application-with-docker-compose)
  - [Step 3: Access the Application](#step-3-access-the-application)
- [Running Automated Tests & Benchmarks](#-running-automated-tests--benchmarks)
- [Project Directory Structure](#-project-directory-structure)
- [Configuration Reference](#-configuration-reference)

---

## 🌟 Key Features

- **🔒 100% Local & Air-Gapped Privacy**: Zero reliance on external cloud APIs (OpenAI, Anthropic, AWS). All embeddings, vector indexing, retrieval, and LLM inferences execute strictly on local hardware.
- **🖥️ Physical Environment Separation**:
  - **Windows Host**: Direct-to-hardware execution of **Ollama** (`Llama 3.2 1B`, `Qwen 2.5 Coder 1.5B`, `StarCoder2 3B`) with zero virtualization overhead.
  - **Ubuntu VM (Docker Compose)**: Containerized microservice running **Streamlit**, **FastEmbed** (`BAAI/bge-small-en-v1.5`), and **Qdrant Vector Database**.
- **💬 Single Unified Chat Interface**: One intuitive interface supporting **Grounded RAG**, **RAG + LLM**, and side-by-side **Compare Both** modes with single-retrieval parity enforcement.
- **🛡️ 3-Layer Defense-in-Depth Guardrails**:
  1. *Input Guardrail*: Enforces length constraints ($\le 500$ chars), rejects empty/whitespace input, sanitizes null bytes and control characters.
  2. *Evidence / RAG Guardrail*: Goes beyond raw cosine similarity—evaluates true factual sufficiency, detects cross-domain topic mismatches (e.g. C/C++ query against biomedical documents), verifies focus attributes, and enforces strict Grounded RAG refusal without invoking the LLM (0 tokens/latency).
  3. *Untrusted LLM Output Guardrail*: Treats all model generations as untrusted—detects and cleans prompt leaks (`System:`, `Context:`) and automatically strips raw tokenizer special tokens (`<end_of_text>`, `<|im_end|>`).
- **🧪 Comprehensive Model Evaluation Workbench (`7_🧪_Model_Evaluation.py`)**:
  - Consolidated **Unified Multi-Model Comparison Chart** on a normalized 0–100% scale (Token F1 Accuracy, Keyword Recall, Relevance, Hallucination Safety, Code Test-Pass, Speed Efficiency).
  - 25-task evaluation dataset across 7 distinct categories (QA, Codebase, Code Generation, Hallucination probes).
  - Automated 22-test Guardrail & AI Output Benchmark suite with **100% pass rate**.
  - Interactive Live Guardrail Rule Playground with one-click presets.
- **🔍 Deep Pipeline Observability**: Real-time breakdown of embedding latency, vector search time, similarity scores, retrieved text payloads, prompt templates, and token accounting stored persistently in SQLite (WAL mode).

---

## 🏗️ System Architecture & Topology

### Physical Environment Separation

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          WINDOWS 10/11 HOST                             │
│  • Ollama Runtime (Listening on 0.0.0.0:11434)                          │
│  • Available Models:                                                    │
│    ├── llama3.2:1b         (Meta Llama 3.2 1B Instruct)                 │
│    ├── qwen2.5-coder:1.5b  (Alibaba Qwen 2.5 Coder 1.5B)               │
│    └── starcoder2:3b       (BigCode StarCoder2 3B)                      │
│  • Execution Parameters: temperature=0.2, num_predict=256               │
└────────────────────────────────────▲────────────────────────────────────┘
                                     │ HTTP API (/api/chat, /api/tags)
    ┌────────────────────────────────┴────────────────────────────────────┐
    │                      VIRTUALBOX UBUNTU VM (DOCKER)                  │
    │  ┌───────────────────────────────────────────────────────────────┐  │
    │  │ DOCKER COMPOSE NETWORK                                        │  │
    │  │                                                               │  │
    │  │  [ Container: rag-app (Port 8501) ]                           │  │
    │  │   • Streamlit Multi-Page UI (7 pages)                         │  │
    │  │   • Input Guardrail (Length, empty check, null bytes)         │  │
    │  │   • FastEmbed BAAI/bge-small-en-v1.5 (384-d normalized)       │  │
    │  │   • Evidence Guardrail (Deep relevance, topic mismatch check) │  │
    │  │   • Output Guardrail (Token stripper & prompt leak cleanse)   │  │
    │  │   • PyMuPDF Document Parser & Recursive Chunker (512 / 64)    │  │
    │  │   • SQLite Metadata & Telemetry Store (metadata/rag.db)       │  │
    │  │                                                               │  │
    │  │       │ gRPC / REST Vector Search (Port 6333)                 │  │
    │  │       ▼                                                       │  │
    │  │  [ Container: rag-qdrant (Port 6333) ]                        │  │
    │  │   • Qdrant Vector Engine v1.11.0                              │  │
    │  │   • HNSW Index with Cosine Distance                           │  │
    │  │   • Persistent Volume: rag_qdrant_storage                     │  │
    │  └───────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Multi-Model Support

The system dynamically connects to Ollama on the host and natively supports switching between 3 specialized local LLMs:

| Model | Parameters | Context Window | Primary Specialization | Strengths in System |
| :--- | :--- | :--- | :--- | :--- |
| **`llama3.2:1b`** | 1.23 Billion | 2,048 tokens | General Reasoning & Synthesis | Fast general synthesis, low memory footprint, excellent conversational clarity. |
| **`qwen2.5-coder:1.5b`** | 1.54 Billion | 2,048 tokens | Codebase QA & Code Synthesis | Strong syntax adherence, high accuracy on technical domain tasks. |
| **`starcoder2:3b`** | 3.03 Billion | 2,048 tokens | Code Generation & Technical Extraction | Highest parameter capacity, strictly cleaned via Output Guardrail to prevent token leaks. |

> **Generation Settings**: All models operate under deterministic, reproducible parameters: `temperature = 0.2`, `num_predict = 256`.

---

## 💬 Unified Chat Page

The **Chat** page (`1_💬_Chat.py`) unifies all interaction workflows into a single interface:

```text
                                  User Query
                                      ↓
                            [ Layer 1: Input Guardrail ]
                             (Length <= 500, empty check)
                                      ↓
                               Query Embedding
                            (FastEmbed bge-small)
                                      ↓
                             Single Qdrant Search
                             (Top-K = 5, Cosine)
                                      ↓
                          [ Layer 2: Evidence Guardrail ]
                           (Threshold + Topic Mismatch)
                                      ↓
                          Sufficient Evidence Found?
                                ↙            ↘
                              YES             NO
                              ↓                ↓
       ┌──────────────────────┴───────┐   ┌────┴──────────────────────────┐
       │ Grounded RAG:                │   │ Grounded RAG:                 │
       │ Calls LLM with context.      │   │ LLM NOT called (0 ms / 0 tok).│
       │ "Grounded Document Answer"   │   │ Displays: "Information not    │
       │                              │   │ found in uploaded documents." │
       ├──────────────────────────────┤   ├───────────────────────────────┤
       │ RAG + LLM:                   │   │ RAG + LLM:                    │
       │ Calls LLM with context.      │   │ Calls LLM without context.    │
       │ "Grounded Document Answer"   │   │ "General LLM Answer" + discl. │
       └──────────────────────────────┘   └───────────────────────────────┘
                                      ↓
                        [ Layer 3: Output Guardrail ]
                      (Special token & prompt cleaner)
                                      ↓
                              Rendered to User
```

### Modes
1. **Grounded RAG**: Strictly grounded in uploaded documents. If the Evidence Guardrail determines insufficient evidence, **the LLM is never called** (0 tokens, 0ms generation time), returning:
   > *"Information not found in the uploaded documents."*
2. **RAG + LLM**: Uses document context when evidence is sufficient; cleanly falls back to the LLM's general knowledge when documents do not contain the answer, explicitly labeled:
   > *"General LLM Answer - No relevant information was found in the uploaded documents. This answer was generated using the selected LLM's general knowledge."*
3. **Compare Both**: Executes **one single retrieval operation** and gives the identical evidence decision to both branches side-by-side, guaranteeing absolute comparison consistency.

---

## 🛡️ 3-Layer Defense-in-Depth Guardrail System

The application implements three explicit guardrail layers, tested against 22 distinct test cases across 15 attack/edge-case categories.

### Active Rules Engine (13 Rules)

#### Layer 1: Input Guardrail (`app/guardrails/input_guardrail.py`)
- **Rule 1.1 — Empty Query Gating**: Rejects empty strings and whitespace-only queries. Prevents unnecessary database queries and LLM hallucinations.
  - *Message*: `"Query is empty. Please provide a valid question."`
- **Rule 1.2 — Maximum Length Enforcement**: Enforces a strict, configurable character ceiling ($\le 500$ chars, defined in `app/config.py`). Blocks embedding context overflow and token inflation attacks before vector search.
  - *Message*: `"Query too long. Please shorten your question to 500 characters or fewer."`
- **Rule 1.3 — Malformed Input & Null Byte Sanitization**: Scans for null bytes (`\x00`) and unprintable control characters, blocking injection and serialization attacks.
  - *Message*: `"Malformed query. Please provide valid text."`
- **Rule 1.4 — Whitespace Normalization**: Trims leading/trailing whitespace and normalizes consecutive spaces to avoid embedding vector noise.

#### Layer 2: Evidence / RAG Guardrail (`app/guardrails/evidence_guardrail.py` & `app/rag/evaluator.py`)
- **Rule 2.1 — Cosine Similarity Cutoff**: Gating at configurable threshold ($\ge 0.55$). If top chunk similarity falls below threshold, retrieval is marked as insufficient.
- **Rule 2.2 — Topical Entity Mismatch Gating**: Prevents out-of-domain queries (e.g. *"which is better c or c++?"*) from being treated as grounded just because raw mathematical similarity exceeded the threshold against unrelated documents (e.g. protein sequence papers).
- **Rule 2.3 — Target Focus Attribute Verification**: Detects queries demanding specific attributes (e.g. cost in dollars, financial pricing) not present in technical documents, overriding false-positive chunk matches.
- **Rule 2.4 — Lexical Token Coverage Verification**: Calculates keyword and entity overlap between query and retrieved chunks. Insufficient token coverage flags the evidence as inadequate.
- **Rule 2.5 — Single-Pass Parity & Strategy Routing**: In *Compare Both* mode, retrieval is executed once. If evidence fails, Grounded RAG blocks the LLM while RAG+LLM falls back to general knowledge.

#### Layer 3: Untrusted LLM Output Guardrail (`app/guardrails/output_guardrail.py`)
- **Rule 3.1 — Model Special Token Sanitization**: Detects and strips raw tokenizer artifacts (`<end_of_text>`, `<|endoftext|>`, `<|im_end|>`, `<s>`, `</s>`, `<unk>`, `<pad>`).
- **Rule 3.2 — Prompt Leakage & Instruction Artifact Cleansing**: Strips leaked prompt headers and system directives (`System:`, `Context:`, `Answer:`, `Provide a more detailed answer:`).
- **Rule 3.3 — Empty & Minimum Output Length Gating**: Intercepts empty or single-character gibberish responses and replaces them with a safe fallback notification.
- **Rule 3.4 — Strict Grounding Consistency Verification**: Guarantees that ungrounded general knowledge answers can never be mislabeled as "Grounded Document Answer".

---

### Before vs. After Guardrails Demonstration

| Test Scenario | Without Guardrail (Problematic) | With Guardrail (Controlled) |
| :--- | :--- | :--- |
| **Topic Mismatch:** *"which is better c or c++?"* on biomedical docs | Raw cosine similarity above cutoff ($0.5656 \ge 0.55$). Grounded RAG calls LLM with protein chunks and mislabels answer as *"Grounded Document Answer"*. | Evidence Guardrail detects entity mismatch, blocks LLM call (0 tokens), and returns: *"Information not found in the uploaded documents."* |
| **Excessive Input Query:** $> 500$ character prompt | 500+ character string is embedded and searched in Qdrant, wasting CPU cycles and context memory. | Input Guardrail rejects query instantly: *"Query too long. Please shorten your question to 500 characters or fewer."* |
| **Model Token Artifacts:** StarCoder emitting `<end_of_text>` | Raw tokenizer tokens and internal instruction prompts leak into user chat bubbles. | Output Guardrail detects and cleanly purges all special tokens and leaked prompt headers before rendering. |
| **General Knowledge in RAG+LLM:** *"What is the capital of France?"* | Passes irrelevant document chunks to LLM, confusing model and yielding corrupted citations. | Evidence Guardrail discards irrelevant chunks, falls back to LLM general knowledge, and badges response *"General LLM Answer"*. |

---

## 🧪 Model Evaluation & Benchmarking Workbench

The dedicated **Model Evaluation** page (`7_🧪_Model_Evaluation.py`) delivers quantitative benchmarks across all three local models.

### Unified Multi-Model Comparison Chart

Instead of fragmented graphs, the dashboard consolidates quality, performance, and hardware telemetry into **one unified multi-model visualization** with interactive perspective toggles:

1. **🎯 Comprehensive Quality & Safety Overview (0–100%)**:
   - **Accuracy (Token F1 %)**: Token overlap with ground truth answers.
   - **Keyword Recall (%)**: Capture rate of vital domain keywords.
   - **Semantic Relevance (%)**: Cosine similarity to reference response.
   - **Hallucination Safety (%)**: $100\% - \text{Hallucination Rate}$ (higher is better).
   - **Code Test-Pass (%)**: Unit test execution pass rate on coding benchmarks.
   - **Speed Efficiency (%)**: Throughput normalized against the fastest model.
2. **⏱️ End-to-End Latency Profile (seconds)**: Mean, Median, and 95th-percentile (P95) response latency per model.
3. **💾 Resource Footprint**: Peak memory (RAM RSS in MB) and CPU execution time (seconds).

### Evaluation Dataset & Benchmarks

- **25 Comprehensive Tasks across 7 Categories**:
  - `RAG_DOMAIN_KNOWLEDGE` (Biomedical & technical paper QA)
  - `CODEBASE_QA` (Internal architecture and function queries)
  - `CODE_GENERATION` (Python algorithms evaluated with live unit test runners)
  - `EDGE_CASES` (Empty inputs, out-of-domain prompts, unanswerable queries)
  - `HALLUCINATION_PROBES` (Prompts designed to provoke ungrounded assertions)
  - `MULTI_HOP_REASONING` (Multi-page evidence synthesis)
  - `SUMMARIZATION` (Document-level summarization)

---

## 📱 Multi-Page Application Walkthrough

The Streamlit interface includes 7 dedicated pages:

| Page | Title | Capabilities |
| :---: | :--- | :--- |
| **—** | **Home (`app.py`)** | Dashboard status banner, active model indicators, vector count, quick-start shortcuts. |
| **1** | **💬 Chat** | Single unified chat page. Model selector, Mode selector (Grounded RAG / RAG+LLM / Compare Both), dynamic threshold slider, expandable retrieval trace, source citations. |
| **2** | **📄 Documents** | Multi-PDF uploader, SHA-256 duplicate detection, PyMuPDF page counter, token estimator, atomic cascade deletion (file + Qdrant vectors + SQLite records). |
| **3** | **⚙️ Pipeline** | Ingestion tuning (Chunk Size 128–1024, Chunk Overlap 0–256, Similarity Threshold), incremental indexing runner, full re-indexing purge. |
| **4** | **🔍 Retrieval Trace** | Deep diagnostic inspector for recent queries. Shows cosine similarity ranking, payload text, FastEmbed latency, Qdrant latency, and full LLM prompt preview. |
| **5** | **📊 Analytics** | Query latency distribution, mode selection breakdown, token usage analytics, and guardrail trigger frequency. |
| **6** | **🖥️ System** | Real-time connectivity matrix for Ollama, Qdrant, SQLite, and FastEmbed; disk space, CPU, and RAM telemetry. |
| **7** | **🧪 Model Evaluation** | Multi-model quantitative benchmark (25 tasks $\times$ 3 models), unified comparison chart, live Guardrail Rule Playground, and automated 22-case AI output test suite. |

---

## 🚀 Quick Start Guide

### Prerequisites

- **Host Machine (Windows)**:
  - [Ollama installed](https://ollama.ai/)
  - Models pulled:
    ```powershell
    ollama pull llama3.2:1b
    ollama pull qwen2.5-coder:1.5b
    ollama pull starcoder2:3b
    ```
- **Virtual Machine / Linux Host (Ubuntu)**:
  - [Docker Engine & Docker Compose installed](https://docs.docker.com/engine/install/)

---

### Step 1: Configure Ollama on Windows Host

By default, Ollama only listens on `127.0.0.1`. Enable cross-network access for Docker/VirtualBox:

1. Open PowerShell on Windows as **Administrator**.
2. Configure Ollama to bind to `0.0.0.0`:
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0:11434', 'Machine')
   ```
3. Allow inbound firewall connections on port 11434:
   ```powershell
   New-NetFirewallRule -DisplayName "Ollama API" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
   ```
4. Restart Ollama from the Windows taskbar system tray.
5. In VirtualBox NAT mode, the Windows host is reachable from Ubuntu at `http://10.0.2.2:11434`.

---

### Step 2: Start Application with Docker Compose

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/your-username/rag-chatbot.git
   cd rag-chatbot
   ```

2. Verify or create your `.env` file (copied from `.env.example`):
   ```bash
   cp .env.example .env
   ```

3. Launch both containers (`rag-app` and `rag-qdrant`):
   ```bash
   docker compose up -d --build
   ```

---

### Step 3: Access the Application

Open your browser and navigate to:
```text
http://localhost:8501
```

- Navigate to **2_📄_Documents** to upload your PDFs.
- Navigate to **3_⚙️_Pipeline** to run incremental indexing.
- Navigate to **1_💬_Chat** to begin chatting with grounded sources!

---

## 🧪 Running Automated Tests & Benchmarks

### 1. Pytest Unit & Integration Suite (64 Tests)

Run the full pytest suite inside the running container:
```bash
docker exec rag-app pytest -v
```
*Output: `64 passed in ~2.5s (100% pass rate)`.*

### 2. AI Output & Guardrail Verification Suite (22 Tests)

Run the automated 22-test guardrail verification suite:
```bash
docker exec rag-app python evaluation/guardrail_runner.py
```
*Output: `22/22 passed (100.0% pass rate)` across Input, Evidence, and Output layers.*

### 3. Full Multi-Model Benchmark (25 Tasks $\times$ 3 Models)

To run the full multi-model benchmark headlessly:
```bash
docker exec rag-app python evaluation/runner.py
```
*Results are written to `evaluation/results/evaluation_results.json` and immediately rendered in Page 7 of the UI.*

---

## 📁 Project Directory Structure

```text
rag-chatbot/
├── .dockerignore
├── .env.example                # Configuration template with verified defaults
├── .gitignore
├── Dockerfile                  # CPU-optimized Python 3.11 container
├── README.md                   # System documentation and operational guide
├── docker-compose.yml          # rag-app (Port 8501) and rag-qdrant (Port 6333) definitions
├── requirements.txt            # Python dependencies (Streamlit, FastEmbed, PyMuPDF, Qdrant)
│
├── app/
│   ├── config.py              # Centralized configuration with env-var fallbacks
│   ├── logging_config.py      # Timed structured logging without plaintext leakage
│   │
│   ├── database/              # SQLite metadata & telemetry storage
│   │   ├── database.py        # Thread-safe connection manager (WAL mode)
│   │   ├── models.py          # DocumentRecord, QueryRecord, PipelineRecord schemas
│   │   └── repository.py      # CRUD repository operations
│   │
│   ├── documents/             # Document lifecycle management
│   │   └── manager.py         # Upload, SHA-256 deduplication, atomic cascade delete
│   │
│   ├── guardrails/            # 3-Layer Defense-in-Depth Guardrail System
│   │   ├── __init__.py
│   │   ├── input_guardrail.py    # Layer 1: Length, empty, control char checks
│   │   ├── evidence_guardrail.py # Layer 2: Deep relevance, topic mismatch checks
│   │   ├── output_guardrail.py   # Layer 3: Token stripper & prompt leak cleaner
│   │   └── models.py             # Guardrail dataclass result models
│   │
│   ├── ingestion/             # Ingestion & document processing
│   │   ├── chunker.py         # Recursive character/token chunker with page provenance
│   │   ├── embedder.py        # FastEmbed BAAI/bge-small-en-v1.5 wrapper
│   │   ├── indexer.py         # Incremental ingestion orchestrator
│   │   └── parser.py          # PyMuPDF text and metadata extractor
│   │
│   ├── llm/                   # Host LLM connection
│   │   └── ollama_client.py   # HTTP client with special token & leak sanitization
│   │
│   ├── rag/                   # Core RAG pipeline
│   │   ├── evaluator.py       # EvidenceEvaluator for topic mismatch & token coverage
│   │   ├── models.py          # RAGResponse, RetrievalTrace, SourceCitation models
│   │   ├── pipeline.py        # Unified orchestrator (Grounded RAG / RAG+LLM / Compare)
│   │   ├── prompt.py          # Grounded prompts & citation formatters
│   │   └── retriever.py       # Qdrant vector search & similarity threshold gating
│   │
│   ├── vectorstore/           # Qdrant client wrapper
│   │   └── qdrant_client.py   # Collection lifecycle, upserts, and payload filtering
│   │
│   └── streamlit/             # Multi-page presentation frontend
│       ├── app.py             # Home overview page
│       ├── components/        # Reusable UI components (status, trace, metrics)
│       └── pages/
│           ├── 1_💬_Chat.py              # Single unified chat page
│           ├── 2_📄_Documents.py         # Upload, inspect, and delete PDFs
│           ├── 3_⚙️_Pipeline.py          # Incremental indexing and parameter tuning
│           ├── 4_🔍_Retrieval_Trace.py   # Diagnostic trace inspector
│           ├── 5_📊_Analytics.py         # Query metrics & guardrail telemetry
│           ├── 6_🖥️_System.py            # Service health & hardware statistics
│           └── 7_🧪_Model_Evaluation.py  # Multi-model benchmarks & guardrails playground
│
├── data/
│   └── documents/             # Persistent storage volume for uploaded PDF files
├── metadata/                  # Persistent storage volume for SQLite database (rag.db)
│
├── evaluation/                # Benchmarking and guardrails evaluation suite
│   ├── dataset.json           # 25 curated tasks across 7 categories
│   ├── guardrails_dataset.json# 22 test cases across 15 attack/edge categories
│   ├── guardrail_runner.py    # Automated test runner for guardrails
│   ├── metrics.py             # Token F1, Keyword Recall, Semantic Relevance formulas
│   ├── repo_understanding.py  # Codebase structure analyzer
│   ├── runner.py              # Full 25-task multi-model evaluation runner
│   └── results/               # Persistent benchmark results and logs
│
└── tests/                     # 64 automated pytest tests
    ├── test_chunker_and_embedder.py
    ├── test_config_and_db.py
    ├── test_evaluation.py
    ├── test_evaluation_metrics.py
    ├── test_guardrails.py
    ├── test_ollama_and_rag_pipeline.py
    ├── test_parser_and_docs.py
    ├── test_qdrant_and_indexer.py
    └── test_retrieval.py
```

---

## ⚙️ Configuration Reference

All settings can be specified via environment variables or in `.env`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://10.0.2.2:11434` | Ollama runtime address on Windows host. |
| `OLLAMA_MODEL` | `llama3.2:1b` | Default LLM model (`llama3.2:1b`, `qwen2.5-coder:1.5b`, `starcoder2:3b`). |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature for deterministic, factual outputs. |
| `LLM_NUM_PREDICT` | `256` | Maximum token limit generated per response. |
| `OLLAMA_TIMEOUT_SECONDS` | `60.0` | Timeout threshold for Ollama HTTP requests. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant vector database URL. |
| `QDRANT_COLLECTION` | `rag_documents` | Target collection name in Qdrant. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model (384 dimensions). |
| `CHUNK_SIZE` | `512` | Token chunk size for recursive splitting. |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks. |
| `TOP_K` | `5` | Number of candidate chunks retrieved from Qdrant. |
| `SIMILARITY_THRESHOLD` | `0.55` | Cosine similarity cutoff for evidence filtering. |
| `MAX_QUERY_LENGTH` | `500` | Input guardrail maximum character threshold. |
| `GUARDRAILS_ENABLED` | `true` | Master flag to toggle guardrail enforcement. |
| `SQLITE_DB_PATH` | `metadata/rag.db` | Path to persistent SQLite database. |
| `UPLOAD_DIR` | `data/documents` | Storage directory for uploaded PDF documents. |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum allowable upload file size per PDF. |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`). |

---