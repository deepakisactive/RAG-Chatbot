"""Exercise 5: RAG Pipeline Trace Analysis Engine.

Traces and documents:
  QUESTION -> RETRIEVED CONTEXT -> LLM RESPONSE

Systematically analyzes 5 distinct diagnostic case studies:
1. RELEVANT_INFO_RETRIEVED: Accurate context retrieval leading to grounded, factually correct response.
2. IRRELEVANT_INFO_RETRIEVED: Semantic drift or distractor chunks retrieved, testing noise robustness.
3. IMPORTANT_INFO_MISSED: Chunk boundary cutoff or sub-optimal ranking causing incomplete answers.
4. CORRECT_ANSWER_PRODUCED: Successful end-to-end synthesis and source citation.
5. HALLUCINATION_DESPITE_CONTEXT: LLM prioritizes internal parametric bias over retrieved factual evidence.

Formalizes the causal relationship:
  RETRIEVAL QUALITY (Recall@K, MRR, Semantic Distance)
       ↓
  CONTEXT QUALITY (Relevance Density, Noise Ratio, Completeness)
       ↓
  LLM RESPONSE QUALITY (Factual Grounding, Hallucination Rate, Precision)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.rag.pipeline import get_pipeline

TRACE_RESULTS_FILE = _project_root / "evaluation" / "results" / "rag_pipeline_trace_cases.json"

CASE_STUDY_QUERIES = [
    {
        "case_id": "CASE_1_RELEVANT_RETRIEVED",
        "title": "Case 1: Relevant Information Retrieved & Correctly Grounded",
        "query": "What is the sequence recovery rate of ProteinMPNN compared to Rosetta?",
        "phenomenon": "High semantic similarity between query and document chunk (#1 rank, score > 0.65) supplies exact quantitative facts. Model faithfully synthesizes percentages (52.4% vs 32.9%) with source citation.",
        "expected_behavior": "Faithful factual extraction with high precision.",
    },
    {
        "case_id": "CASE_2_IRRELEVANT_RETRIEVED",
        "title": "Case 2: Irrelevant / Distractor Information Retrieved (Noise Injection)",
        "query": "What are the specific hyperparameter settings for AdamW optimizer in the cloudburst prediction neural network?",
        "phenomenon": "Query matches general terminology ('neural network', 'sequence model') across cloudburst and patient embedding chunks, but specific optimizer hyperparameter values (beta1, beta2, weight decay) are absent from the retrieved excerpts.",
        "expected_behavior": "Evaluates whether LLM detects incomplete context or hallucinates plausible learning rates / decay values.",
    },
    {
        "case_id": "CASE_3_IMPORTANT_INFO_MISSED",
        "title": "Case 3: Important Information Missed (Chunk Boundary Cutoff)",
        "query": "Compare the clinical validation cohorts used for sepsis prediction versus generative disease history modeling.",
        "phenomenon": "Answering requires aggregating information across two distinct papers (MIMIC-III / eICU for sepsis vs UK Biobank / Danish registries for disease history). Fixed Top-K (5) biased towards one paper causes partial omission of the other.",
        "expected_behavior": "Highlights multi-document recall limitations of basic single-vector retrieval.",
    },
    {
        "case_id": "CASE_4_CORRECT_ANSWER_PRODUCED",
        "title": "Case 4: End-to-End Correct Grounded Answer with Strict Factuality",
        "query": "What clinical outcomes or lead times are predicted by early sepsis deep learning models?",
        "phenomenon": "Query returns chunks detailing lead times (4-12 hours before onset) and AUROC metrics (>0.85). LLM produces clear, well-formatted bullet points directly derived from text.",
        "expected_behavior": "Ideal RAG behavior where retrieval precision translates directly to answer accuracy.",
    },
    {
        "case_id": "CASE_5_HALLUCINATION_DESPITE_CONTEXT",
        "title": "Case 5: Parametric Bias Override / Hallucination Despite Retrieved Context",
        "query": "According to the ProteinMPNN paper, can ProteinMPNN design small molecule ligands and inorganic crystals?",
        "phenomenon": "Retrieved context clearly states ProteinMPNN designs protein amino acid sequences for fixed backbones. However, weak or smaller models occasionally generalize parametric knowledge to claim it designs 'any molecular structure'.",
        "expected_behavior": "Demonstrates parametric prior override where LLM expands beyond context boundaries.",
    },
]


def run_trace_analysis() -> Dict[str, Any]:
    """Execute trace analysis on the 5 diagnostic queries across the models."""
    pipeline = get_pipeline()
    pipeline.ollama_client.timeout = 300.0
    trace_records = []

    print("\n🔍 Running Exercise 5: RAG Pipeline Trace Analysis...")

    for case in CASE_STUDY_QUERIES:
        query_text = case["query"]
        print(f"\n--- Tracing: {case['case_id']} ---")
        print(f"Query: {query_text}")

        # Execute query through RAG pipeline
        response = pipeline.query(query_text=query_text, top_k=5, threshold=0.35)

        retrieved_excerpts = []
        for idx, chunk in enumerate(response.trace.filtered_chunks, start=1):
            retrieved_excerpts.append({
                "rank": idx,
                "document": chunk.filename,
                "page": chunk.page,
                "similarity_score": round(chunk.similarity_score, 4),
                "text_snippet": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
            })

        record = {
            "case_id": case["case_id"],
            "title": case["title"],
            "phenomenon": case["phenomenon"],
            "expected_behavior": case["expected_behavior"],
            "query": query_text,
            "is_low_confidence": response.is_low_confidence,
            "highest_similarity": response.trace.highest_similarity,
            "retrieved_count": len(response.trace.filtered_chunks),
            "retrieved_excerpts": retrieved_excerpts,
            "llm_response": response.answer,
            "model_used": response.model,
            "timings": {
                "embedding_time_ms": response.embedding_time_ms,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
            },
        }
        trace_records.append(record)

    output_data = {
        "analysis_meta": {
            "purpose": "Exercise 5 - Pipeline Trace & Retrieval-Context-Response Relationship",
            "pipeline_config": {
                "top_k": 5,
                "threshold": 0.35,
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "distance_metric": "Cosine",
            },
        },
        "case_studies": trace_records,
    }

    TRACE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Trace analysis complete! Output saved to {TRACE_RESULTS_FILE}")
    return output_data


if __name__ == "__main__":
    run_trace_analysis()

