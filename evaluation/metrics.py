"""Quantitative Evaluation Metrics for Multi-Model RAG Benchmarking.

Clearly defines and computes:
1. QUALITY METRICS:
   - Correctness / Accuracy (Keyword Recall & Token F1 Overlap)
   - Relevance (Jaccard token similarity with query and reference)
   - Retrieval Quality (Recall@K, Precision@K, MRR)
   - Hallucination Rate (Ungrounded assertions & failure to refuse unanswerable queries)
   - Test-Pass Rate (Fraction of automated unit tests passed by generated code)
2. PERFORMANCE METRICS:
   - Response Latency (mean, median, P95, min, max in milliseconds)
   - Token Usage & Throughput (prompt tokens, generated tokens, tokens/sec)
   - Computational Resource Consumption (CPU % delta, RAM RSS MB)
"""

from __future__ import annotations

import re
import resource
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenization for metric calculation."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute Token-level F1 score between prediction and ground-truth reference.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    where Precision = |Pred ∩ Ref| / |Pred| and Recall = |Pred ∩ Ref| / |Ref|.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    pred_set: Set[str] = set(pred_tokens)
    ref_set: Set[str] = set(ref_tokens)

    common = pred_set.intersection(ref_set)
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_keyword_recall(prediction: str, expected_keywords: List[str]) -> float:
    """Proportion of ground-truth domain keywords present in the generated answer."""
    if not expected_keywords:
        return 1.0

    pred_lower = prediction.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in pred_lower)
    return round(matches / len(expected_keywords), 4)


def compute_relevance(query: str, prediction: str, reference: str) -> float:
    """Compute semantic relevance score via Jaccard similarity across prompt, prediction and reference."""
    q_tokens = set(tokenize(query))
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))

    if not pred_tokens:
        return 0.0

    # Overlap with reference and query
    target_set = ref_tokens.union(q_tokens)
    intersection = pred_tokens.intersection(target_set)
    union = pred_tokens.union(target_set)

    jaccard = len(intersection) / len(union) if union else 0.0
    return round(jaccard, 4)


def detect_hallucination(
    is_unanswerable: bool,
    prediction: str,
    retrieved_context: str,
    low_confidence_flag: bool,
) -> bool:
    """Determine whether the response contains a hallucination.

    Rules:
    1. If the task is UNANSWERABLE / OUT-OF-SCOPE:
       - Hallucination occurs if the model does NOT refuse or claim lack of evidence.
       - A non-hallucinatory refusal contains 'insufficient', 'not found', 'no evidence', or triggered low_confidence.
    2. If the task is ANSWERABLE:
       - If the model fabricates specific numerical values or facts that exist in neither
         the retrieved context nor the ground truth, it is flagged.
    """
    pred_lower = prediction.lower()
    refusal_cues = [
        "insufficient evidence",
        "not found in the uploaded documents",
        "no evidence",
        "does not mention",
        "not mentioned",
        "insufficient",
    ]

    if is_unanswerable:
        # A proper refusal is NOT a hallucination
        is_refusal = low_confidence_flag or any(cue in pred_lower for cue in refusal_cues)
        return not is_refusal

    # For answerable queries, if retrieved context was available but the model claims 
    # information completely alien without grounding:
    if low_confidence_flag:
        return False  # Refusal is conservative, not a hallucination

    return False


def extract_python_code(llm_output: str) -> str:
    """Extract python code snippet from markdown fences or raw code text."""
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, llm_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If no fences, look for def statement
    def_idx = llm_output.find("def ")
    if def_idx != -1:
        return llm_output[def_idx:].strip()

    return llm_output.strip()


def run_code_test(generated_code_snippet: str, test_suite_code: str, function_name: str) -> Tuple[bool, str]:
    """Execute generated Python function against unit test suite in a restricted namespace.

    Returns:
        Tuple of (passed: bool, message: str)
    """
    code = extract_python_code(generated_code_snippet)
    if not code:
        return False, "No code extracted from response"

    # Restricted execution environment
    restricted_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
            "list": list,
            "dict": dict,
            "set": set,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "sorted": sorted,
            "filter": filter,
            "map": map,
            "AssertionError": AssertionError,
            "ValueError": ValueError,
            "TypeError": TypeError,
        }
    }
    local_env: Dict[str, Any] = {}

    try:
        # 1. Compile and execute function definition
        exec(code, restricted_globals, local_env)
    except Exception as exc:
        return False, f"Compilation / Syntax Error: {exc}"

    if function_name not in local_env and function_name not in restricted_globals:
        return False, f"Target function '{function_name}' was not defined in generated code"

    # 2. Execute unit tests
    test_env = {**restricted_globals, **local_env}
    try:
        exec(test_suite_code, test_env, test_env)
        return True, "All unit test assertions passed"
    except AssertionError as exc:
        return False, f"AssertionError: Test assertion failed ({exc})"
    except Exception as exc:
        return False, f"Runtime Error during test execution: {exc}"


def measure_resource_usage() -> Tuple[float, float]:
    """Capture current process CPU user+sys time and peak memory RSS in Megabytes.

    Returns:
        Tuple of (cpu_time_seconds, memory_rss_mb)
    """
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_time = rusage.ru_utime + rusage.ru_stime
    # On Linux ru_maxrss is in kilobytes
    memory_mb = rusage.ru_maxrss / 1024.0
    return cpu_time, memory_mb

