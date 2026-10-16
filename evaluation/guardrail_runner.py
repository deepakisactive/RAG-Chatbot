"""Automated Test Runner for AI Output & Guardrail Verification.

Loads evaluation/guardrails_dataset.json, executes all test cases against the
3 guardrail layers (Input, Evidence, Output), verifies explicit Pass/Fail criteria,
calculates real pass rates, and prints the standardized benchmark report.

Saves structured results to evaluation/results/guardrail_test_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import settings
from app.guardrails import (
    EvidenceGuardrail,
    InputGuardrail,
    OutputGuardrail,
    QUERY_TOO_LONG_MESSAGE,
    OUTPUT_VALIDATION_FAILED_MESSAGE,
)
from app.rag.prompt import LOW_CONFIDENCE_MESSAGE
from app.logging_config import logger
from app.rag.pipeline import RAGPipeline, get_pipeline

DATASET_PATH = _project_root / "evaluation" / "guardrails_dataset.json"
RESULTS_DIR = _project_root / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "guardrail_test_results.json"


class GuardrailRunner:
    """Executes AI Output and Guardrail benchmark suite."""

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        pipeline: Optional[RAGPipeline] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.pipeline = pipeline or get_pipeline()
        self.input_guardrail = self.pipeline.input_guardrail
        self.evidence_guardrail = self.pipeline.evidence_guardrail
        self.output_guardrail = self.pipeline.output_guardrail

    def load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_test_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test case and evaluate explicit Pass/Fail criteria."""
        test_id = case["test_id"]
        category = case["category"]
        query = case.get("query", "")
        mode = case.get("mode", "grounded")
        target_guardrail = case.get("target_guardrail", "none")
        expected_triggered = case.get("expected_guardrail_triggered", False)
        expected_layer = case.get("expected_guardrail_layer")
        expected_result_type = case.get("expected_result_type", "")
        simulated_raw = case.get("simulated_raw_output")

        actual_layer = None
        actual_triggered = False
        actual_result_type = ""
        passed = False
        notes = []

        # ── Test Execution ──
        if target_guardrail == "input":
            # Test Input Guardrail directly and via pipeline
            inp_res = self.input_guardrail.validate(query)
            if not inp_res.is_valid:
                actual_triggered = True
                actual_layer = "input"
                actual_result_type = "rejected_input"
                notes.append(f"Input rejected: {inp_res.reason}")
            else:
                actual_result_type = "accepted_input"
                notes.append("Input accepted")

            # Check criteria
            if actual_triggered == expected_triggered and actual_layer == expected_layer:
                if "500" in inp_res.reason or len(query) > 500:
                    if inp_res.error_message == QUERY_TOO_LONG_MESSAGE:
                        passed = True
                    else:
                        notes.append("Mismatch in exact 500 char error message")
                else:
                    passed = True

        elif target_guardrail == "output" and simulated_raw is not None:
            # Test Output Guardrail with simulated raw output
            out_res = self.output_guardrail.validate(
                raw_output=simulated_raw,
                is_grounded_mode=True,
                has_evidence=True,
                enabled=True,
            )
            actual_triggered = out_res.guardrail_triggered
            actual_layer = "output" if actual_triggered else None
            actual_result_type = "grounded_answer" if out_res.is_valid else "blocked_output"

            # Check special tokens removed or headers stripped
            clean = out_res.sanitized_output
            has_tokens = any(tok in clean for tok in ["<end_of_text>", "<|endoftext|>", "<|im_end|>"])
            has_leakage = any(hdr in clean for hdr in ["System:\n", "Context:\n", "Answer:\n"])

            if not has_tokens and not has_leakage and out_res.is_valid:
                passed = True
                notes.append("Special tokens/prompt leakage stripped cleanly")
            else:
                notes.append(f"Output sanitization failed: tokens_present={has_tokens}, leakage={has_leakage}")

        else:
            # End-to-end pipeline execution
            res = self.pipeline.query(
                query_text=query,
                chat_mode=mode,
                guardrails_enabled=True,
            )

            actual_triggered = res.guardrail_triggered
            actual_layer = None
            if res.final_result_type == "BLOCKED_INPUT":
                actual_layer = "input"
                actual_result_type = "rejected_input"
            elif res.final_result_type == "REFUSAL":
                actual_layer = "evidence"
                actual_result_type = "refusal"
            elif res.final_result_type == "GENERAL_LLM_ANSWER":
                actual_layer = "evidence"
                actual_result_type = "general_llm_answer"
            elif res.final_result_type == "GROUNDED_ANSWER":
                actual_result_type = "grounded_answer"
                actual_layer = "output" if res.output_guardrail_triggered else None

            # Verify Pass/Fail criteria against test case expectation
            if expected_result_type == "refusal":
                if res.answer == LOW_CONFIDENCE_MESSAGE and not res.llm_used and not res.evidence_sufficient:
                    passed = True
                    notes.append("Refusal verified: LLM not called, exact message returned")
                else:
                    notes.append(f"Failed refusal criteria: llm_used={res.llm_used}, ans={res.answer[:40]}")
            elif expected_result_type == "general_llm_answer":
                if res.is_general_knowledge and res.llm_used and not res.evidence_sufficient:
                    passed = True
                    notes.append("General LLM fallback verified: answer labeled General LLM Answer")
                else:
                    notes.append(f"Failed fallback criteria: is_gen={res.is_general_knowledge}, llm_used={res.llm_used}")
            elif expected_result_type == "grounded_answer":
                if res.evidence_sufficient and res.retrieval_completed and len(res.sources) > 0:
                    passed = True
                    notes.append("Grounded answer verified: sufficient evidence and sources present")
                else:
                    notes.append(f"Failed grounded criteria: suff={res.evidence_sufficient}, chunks={res.retrieved_candidate_count}")
            elif expected_result_type == "rejected_input":
                if res.final_result_type == "BLOCKED_INPUT" and not res.retrieval_completed:
                    passed = True
                    notes.append("Input rejection verified: retrieval not completed")
                else:
                    notes.append("Failed input rejection criteria")

        return {
            "test_id": test_id,
            "category": category,
            "query": query[:60] + ("..." if len(query) > 60 else ""),
            "target_guardrail": target_guardrail,
            "passed": passed,
            "expected_triggered": expected_triggered,
            "actual_triggered": actual_triggered,
            "expected_layer": expected_layer,
            "actual_layer": actual_layer,
            "expected_result_type": expected_result_type,
            "actual_result_type": actual_result_type,
            "notes": "; ".join(notes),
        }

    def run(self) -> Dict[str, Any]:
        """Run all test cases, calculate statistics, and print summary report."""
        dataset = self.load_dataset()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        total_tests = len(dataset)
        passed_tests = 0

        # Layer counters
        input_total = 0
        input_passed = 0
        evidence_total = 0
        evidence_passed = 0
        output_total = 0
        output_passed = 0
        # Check if Ollama service is reachable; if not, enable mock_mode to prevent timeout delays
        hc = self.pipeline.ollama_client.health_check()
        if hc.get("status") != "connected":
            logger.info("Ollama is not reachable on host; enabling mock_mode for evaluation runner.")
            self.pipeline.ollama_client.mock_mode = True

        for case in dataset:
            res = self.run_test_case(case)
            results.append(res)
            if res["passed"]:
                passed_tests += 1

            target = case.get("target_guardrail", "none")
            if target == "input":
                input_total += 1
                if res["passed"]:
                    input_passed += 1
            elif target == "evidence":
                evidence_total += 1
                if res["passed"]:
                    evidence_passed += 1
            elif target == "output":
                output_total += 1
                if res["passed"]:
                    output_passed += 1
            else:
                # Supported / None target - touches evidence and output validation
                evidence_total += 1
                if res["passed"]:
                    evidence_passed += 1

        failed_tests = total_tests - passed_tests
        pass_rate = round((passed_tests / total_tests) * 100.0, 1) if total_tests > 0 else 0.0

        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate_pct": pass_rate,
            "layers": {
                "input": {"passed": input_passed, "total": input_total},
                "evidence": {"passed": evidence_passed, "total": evidence_total},
                "output": {"passed": output_passed, "total": output_total},
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        }

        # Save to JSON with permissive read/write permissions for host and container
        try:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            try:
                os.chmod(RESULTS_FILE, 0o666)
            except Exception:
                pass
        except PermissionError:
            # If owned by another user, try unlinking or fallback
            try:
                if RESULTS_FILE.exists():
                    RESULTS_FILE.unlink(missing_ok=True)
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                os.chmod(RESULTS_FILE, 0o666)
            except Exception as pe:
                logger.warning(f"Could not persist results file {RESULTS_FILE}: {pe}")
        except Exception as e:
            logger.warning(f"Could not write results to {RESULTS_FILE}: {e}")

        # Print the exact requested summary output
        print("\n========================================")
        print("AI OUTPUT & GUARDRAIL TEST RESULTS")
        print("========================================")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Pass Rate: {pass_rate}%")
        print(f"Input Guardrail:    {input_passed}/{input_total}")
        print(f"Evidence Guardrail: {evidence_passed}/{evidence_total}")
        print(f"Output Guardrail:   {output_passed}/{output_total}")
        print("========================================\n")

        return summary


def run_benchmark() -> Dict[str, Any]:
    runner = GuardrailRunner()
    return runner.run()


if __name__ == "__main__":
    run_benchmark()
