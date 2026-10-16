"""Automated Evaluation Runner for Multi-Model RAG & Code LLM Benchmarking.

Evaluates the application under strictly identical conditions across:
- llama3.2:1b
- starcoder2:3b
- qwen2.5-coder:1.5b

Saves complete structured output to evaluation/results/evaluation_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import settings
from app.llm.ollama_client import get_ollama_client
from app.logging_config import logger
from app.rag.pipeline import RAGPipeline, get_pipeline
from evaluation.metrics import (
    compute_keyword_recall,
    compute_relevance,
    compute_token_f1,
    detect_hallucination,
    measure_resource_usage,
    run_code_test,
)

EVAL_DATASET_PATH = _project_root / "evaluation" / "dataset.json"
RESULTS_DIR = _project_root / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

MODELS_TO_EVALUATE = [
    "llama3.2:1b",
    "starcoder2:3b",
    "qwen2.5-coder:1.5b",
]


class BenchmarkRunner:
    """Orchestrates multi-model quantitative evaluation across the benchmark suite."""

    def __init__(
        self,
        dataset_path: Path = EVAL_DATASET_PATH,
        models: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> None:
        self.dataset_path = dataset_path
        self.models = models or MODELS_TO_EVALUATE
        self.top_k = top_k
        self.threshold = threshold
        self.pipeline = get_pipeline()
        self.pipeline.ollama_client.timeout = 300.0
        self.ollama_client = get_ollama_client()
        self.ollama_client.timeout = 300.0

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(self) -> Dict[str, Any]:
        """Execute complete benchmark across all models and tasks."""
        dataset = self.load_dataset()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Multi-Model Evaluation: {len(dataset)} tasks across {len(self.models)} models.")
        benchmark_results: Dict[str, Any] = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "top_k": self.top_k,
                "similarity_threshold": self.threshold,
                "total_tasks": len(dataset),
                "models": self.models,
            },
            "model_summaries": {},
            "detailed_task_results": [],
        }

        for model_name in self.models:
            logger.info(f"--- Evaluating Model: {model_name} ---")
            model_summary = self._evaluate_model(model_name, dataset)
            benchmark_results["model_summaries"][model_name] = model_summary

        # Build detailed task comparison table
        for task in dataset:
            task_entry = {
                "id": task["id"],
                "category": task["category"],
                "task_type": task["task_type"],
                "question": task["question"],
                "ground_truth": task["ground_truth"],
                "is_unanswerable": task.get("is_unanswerable", False),
                "model_outputs": {},
            }
            for model_name in self.models:
                # Find task result in model summary detailed records
                records = benchmark_results["model_summaries"][model_name]["task_records"]
                matching = next((r for r in records if r["id"] == task["id"]), None)
                if matching:
                    task_entry["model_outputs"][model_name] = {
                        "answer": matching["answer"],
                        "generation_time_ms": matching["generation_time_ms"],
                        "total_time_ms": matching["total_time_ms"],
                        "token_f1": matching["token_f1"],
                        "keyword_recall": matching["keyword_recall"],
                        "relevance": matching["relevance"],
                        "is_hallucination": matching["is_hallucination"],
                        "test_passed": matching.get("test_passed"),
                        "retrieved_count": matching["retrieved_count"],
                        "highest_similarity": matching["highest_similarity"],
                    }
            benchmark_results["detailed_task_results"].append(task_entry)

        # Write results to file
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)

        logger.info(f"Benchmark completed successfully! Results written to {RESULTS_FILE}")
        return benchmark_results

    def _evaluate_model(self, model_name: str, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single model against all tasks in the dataset."""
        task_records: List[Dict[str, Any]] = []

        total_gen_time = 0.0
        total_pipeline_time = 0.0
        total_f1 = 0.0
        total_kw_recall = 0.0
        total_relevance = 0.0
        hallucination_count = 0
        code_tasks_count = 0
        code_tests_passed = 0
        retrieval_hits = 0

        cpu_start, mem_start = measure_resource_usage()
        t_model_start = time.perf_counter()

        for idx, task in enumerate(dataset, start=1):
            logger.info(f"[{model_name}] Running Task {idx}/{len(dataset)}: {task['id']} ({task['category']})")

            # Execute RAG query under identical conditions
            response = self.pipeline.query(
                query_text=task["question"],
                top_k=self.top_k,
                threshold=self.threshold,
                model=model_name,
            )

            # Evaluate quality metrics
            f1 = compute_token_f1(response.answer, task["ground_truth"])
            kw_recall = compute_keyword_recall(response.answer, task.get("expected_keywords", []))
            relevance = compute_relevance(task["question"], response.answer, task["ground_truth"])
            is_hallucinating = detect_hallucination(
                is_unanswerable=task.get("is_unanswerable", False),
                prediction=response.answer,
                retrieved_context=response.context_sent,
                low_confidence_flag=response.is_low_confidence,
            )

            # Check code generation tests if applicable
            test_passed: Optional[bool] = None
            test_message: Optional[str] = None
            if task.get("is_code_task"):
                code_tasks_count += 1
                fn_name = task.get("test_function_name", "")
                test_code = task.get("test_suite_code", "")
                test_passed, test_message = run_code_test(response.answer, test_code, fn_name)
                if test_passed:
                    code_tests_passed += 1

            # Check retrieval hit if ground truth document specified
            hit_at_k = False
            expected_doc = task.get("expected_document")
            if expected_doc and response.trace.retrieved_chunks:
                hit_at_k = any(expected_doc.lower() in c.filename.lower() for c in response.trace.retrieved_chunks)
                if hit_at_k:
                    retrieval_hits += 1

            # Accumulate metrics
            total_gen_time += response.generation_time_ms
            total_pipeline_time += response.total_time_ms
            total_f1 += f1
            total_kw_recall += kw_recall
            total_relevance += relevance
            if is_hallucinating:
                hallucination_count += 1

            record = {
                "id": task["id"],
                "category": task["category"],
                "task_type": task["task_type"],
                "answer": response.answer,
                "is_low_confidence": response.is_low_confidence,
                "retrieved_count": len(response.trace.filtered_chunks),
                "highest_similarity": response.trace.highest_similarity,
                "embedding_time_ms": response.embedding_time_ms,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
                "token_f1": f1,
                "keyword_recall": kw_recall,
                "relevance": relevance,
                "is_hallucination": is_hallucinating,
                "test_passed": test_passed,
                "test_message": test_message,
                "hit_at_k": hit_at_k if expected_doc else None,
            }
            task_records.append(record)

        cpu_end, mem_end = measure_resource_usage()
        elapsed_sec = time.perf_counter() - t_model_start
        n_tasks = len(dataset)

        gen_times = [r["generation_time_ms"] for r in task_records if r["generation_time_ms"] > 0]
        gen_times.sort()
        mean_gen_time = round(total_gen_time / n_tasks, 2)
        median_gen_time = round(gen_times[len(gen_times) // 2], 2) if gen_times else 0.0
        p95_idx = int(0.95 * len(gen_times)) if gen_times else 0
        p95_gen_time = round(gen_times[p95_idx], 2) if gen_times else 0.0

        test_pass_rate = round((code_tests_passed / code_tasks_count) * 100.0, 2) if code_tasks_count > 0 else 0.0
        hallucination_rate = round((hallucination_count / n_tasks) * 100.0, 2)

        return {
            "model_name": model_name,
            "total_tasks_evaluated": n_tasks,
            "mean_token_f1_accuracy": round(total_f1 / n_tasks, 4),
            "mean_keyword_recall": round(total_kw_recall / n_tasks, 4),
            "mean_relevance": round(total_relevance / n_tasks, 4),
            "hallucination_rate_pct": hallucination_rate,
            "hallucination_count": hallucination_count,
            "code_tasks_evaluated": code_tasks_count,
            "code_test_pass_rate_pct": test_pass_rate,
            "code_tests_passed": code_tests_passed,
            "mean_generation_latency_ms": mean_gen_time,
            "median_generation_latency_ms": median_gen_time,
            "p95_generation_latency_ms": p95_gen_time,
            "mean_total_pipeline_ms": round(total_pipeline_time / n_tasks, 2),
            "total_benchmark_elapsed_sec": round(elapsed_sec, 2),
            "cpu_time_sec_delta": round(cpu_end - cpu_start, 3),
            "peak_memory_rss_mb": round(mem_end, 2),
            "task_records": task_records,
        }


def main() -> None:
    runner = BenchmarkRunner()
    results = runner.run()

    print("\n" + "=" * 80)
    print("🎯 MULTI-MODEL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<32} | {'llama3.2:1b':<14} | {'starcoder2:3b':<14} | {'qwen2.5-coder:1.5b':<14}")
    print("-" * 80)

    summaries = results["model_summaries"]
    l_sum = summaries.get("llama3.2:1b", {})
    s_sum = summaries.get("starcoder2:3b", {})
    q_sum = summaries.get("qwen2.5-coder:1.5b", {})

    print(f"{'Accuracy (Token F1)':<32} | {l_sum.get('mean_token_f1_accuracy', 0):<14.4f} | {s_sum.get('mean_token_f1_accuracy', 0):<14.4f} | {q_sum.get('mean_token_f1_accuracy', 0):<14.4f}")
    print(f"{'Keyword Recall':<32} | {l_sum.get('mean_keyword_recall', 0):<14.4f} | {s_sum.get('mean_keyword_recall', 0):<14.4f} | {q_sum.get('mean_keyword_recall', 0):<14.4f}")
    print(f"{'Relevance Score':<32} | {l_sum.get('mean_relevance', 0):<14.4f} | {s_sum.get('mean_relevance', 0):<14.4f} | {q_sum.get('mean_relevance', 0):<14.4f}")
    print(f"{'Hallucination Rate (%)':<32} | {l_sum.get('hallucination_rate_pct', 0):<14.1f}% | {s_sum.get('hallucination_rate_pct', 0):<14.1f}% | {q_sum.get('hallucination_rate_pct', 0):<14.1f}%")
    print(f"{'Code Test-Pass Rate (%)':<32} | {l_sum.get('code_test_pass_rate_pct', 0):<14.1f}% | {s_sum.get('code_test_pass_rate_pct', 0):<14.1f}% | {q_sum.get('code_test_pass_rate_pct', 0):<14.1f}%")
    print(f"{'Mean Generation Latency':<32} | {l_sum.get('mean_generation_latency_ms', 0):<11.1f} ms | {s_sum.get('mean_generation_latency_ms', 0):<11.1f} ms | {q_sum.get('mean_generation_latency_ms', 0):<11.1f} ms")
    print(f"{'Median Generation Latency':<32} | {l_sum.get('median_generation_latency_ms', 0):<11.1f} ms | {s_sum.get('median_generation_latency_ms', 0):<11.1f} ms | {q_sum.get('median_generation_latency_ms', 0):<11.1f} ms")
    print(f"{'P95 Generation Latency':<32} | {l_sum.get('p95_generation_latency_ms', 0):<11.1f} ms | {s_sum.get('p95_generation_latency_ms', 0):<11.1f} ms | {q_sum.get('p95_generation_latency_ms', 0):<11.1f} ms")
    print(f"{'Peak Memory RSS':<32} | {l_sum.get('peak_memory_rss_mb', 0):<11.1f} MB | {s_sum.get('peak_memory_rss_mb', 0):<11.1f} MB | {q_sum.get('peak_memory_rss_mb', 0):<11.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()

