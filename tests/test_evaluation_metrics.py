"""Unit tests for quantitative evaluation metrics calculation."""

import unittest

from evaluation.metrics import (
    compute_keyword_recall,
    compute_relevance,
    compute_token_f1,
    detect_hallucination,
    extract_python_code,
    run_code_test,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Verify quality and performance metric calculation algorithms."""

    def test_token_f1_exact_and_partial(self) -> None:
        self.assertEqual(compute_token_f1("hello world", "hello world"), 1.0)
        self.assertEqual(compute_token_f1("hello", "world"), 0.0)
        f1 = compute_token_f1("ProteinMPNN achieves 52.4% recovery", "ProteinMPNN achieves high sequence recovery")
        self.assertTrue(0.0 < f1 < 1.0)

    def test_keyword_recall(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        self.assertEqual(compute_keyword_recall(text, ["fox", "dog"]), 1.0)
        self.assertEqual(compute_keyword_recall(text, ["fox", "cat"]), 0.5)
        self.assertEqual(compute_keyword_recall(text, ["elephant"]), 0.0)

    def test_relevance(self) -> None:
        rel = compute_relevance("What is sepsis?", "Sepsis is a medical condition.", "Sepsis is an organ dysfunction.")
        self.assertTrue(0.0 < rel <= 1.0)

    def test_hallucination_detection_unanswerable(self) -> None:
        # Proper refusal should NOT be flagged as hallucination
        is_hallucinating = detect_hallucination(
            is_unanswerable=True,
            prediction="Insufficient evidence was found in the uploaded documents to answer this question.",
            retrieved_context="",
            low_confidence_flag=True,
        )
        self.assertFalse(is_hallucinating)

        # Fabricated answer to unanswerable question SHOULD be flagged
        is_hallucinating = detect_hallucination(
            is_unanswerable=True,
            prediction="Kubernetes cluster autoscaling is configured with HPA and minimum 3 replicas.",
            retrieved_context="",
            low_confidence_flag=False,
        )
        self.assertTrue(is_hallucinating)

    def test_code_execution_test_runner(self) -> None:
        code = (
            "def add_numbers(a, b):\n"
            "    return a + b\n"
        )
        test_suite = "assert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0"
        passed, msg = run_code_test(code, test_suite, "add_numbers")
        self.assertTrue(passed, msg)

        # Failing code
        bad_code = "def add_numbers(a, b):\n    return a - b"
        passed, msg = run_code_test(bad_code, test_suite, "add_numbers")
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()

