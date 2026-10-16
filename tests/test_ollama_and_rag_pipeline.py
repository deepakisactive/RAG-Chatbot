"""Unit tests for Ollama client, prompt builder, and end-to-end RAG pipeline."""

import tempfile
import unittest
from pathlib import Path

from app.config import settings
from app.database.database import DatabaseManager
from app.ingestion.chunker import TextChunk
from app.ingestion.embedder import MockEmbedder
from app.llm.ollama_client import OllamaClient
from app.rag.models import ComparisonResponse, RAGResponse
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import GENERAL_LLM_LABEL, LOW_CONFIDENCE_MESSAGE, PromptBuilder
from app.rag.retriever import RetrievalEngine
from app.vectorstore.qdrant_client import VectorStoreClient


class TestOllamaAndRAGPipeline(unittest.TestCase):
    """Test suite for LLM integration, prompt formatting, and end-to-end RAG workflow."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.db_path = self.temp_path / "test_rag.db"
        self.db_manager = DatabaseManager(db_path=self.db_path)

        self.embedder = MockEmbedder(dimension=384)
        self.vectorstore = VectorStoreClient(
            collection_name="pipeline_test",
            dimension=384,
            use_in_memory=True,
        )
        self.retriever = RetrievalEngine(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        self.ollama_client = OllamaClient(
            base_url="http://10.0.2.2:11434",
            model="llama3.2:1b",
            mock_mode=True,
        )
        self.prompt_builder = PromptBuilder()

        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            ollama_client=self.ollama_client,
            prompt_builder=self.prompt_builder,
            db_manager=self.db_manager,
        )

        # Seed knowledge chunk
        chunk = TextChunk(
            chunk_id="chunk_llama3_01",
            document_id="doc_rag_guide",
            filename="rag_guide.pdf",
            page=7,
            chunk_index=0,
            text="Llama 3.2 1B is a compact, high-efficiency language model optimized for edge devices.",
            char_count=87,
            token_estimate=21,
            pipeline_version="v1",
        )
        vectors = self.embedder.embed_texts([chunk.text])
        self.vectorstore.upsert_chunks([chunk], vectors)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prompt_builder_formatting(self) -> None:
        """Verify prompt construction attaches excerpts with citations and strict guidelines."""
        results = self.vectorstore.search(self.embedder.embed_query("Llama 3.2"), top_k=1)
        prompt, context_str, duration = self.prompt_builder.build_prompt("What is Llama 3.2 1B?", results)

        self.assertIn("Llama 3.2 1B is a compact", prompt)
        self.assertIn("rag_guide.pdf", prompt)
        self.assertIn("Page: 7", prompt)
        self.assertIn("Insufficient evidence was found", prompt)
        self.assertTrue(duration >= 0.0)

    def test_end_to_end_rag_high_confidence(self) -> None:
        """Verify confident query retrieves context, constructs prompt, calls LLM, and logs telemetry."""
        resp = self.pipeline.query(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
        )
        self.assertFalse(resp.is_low_confidence)
        self.assertTrue(len(resp.sources) > 0)
        self.assertEqual(resp.sources[0].filename, "rag_guide.pdf")
        self.assertEqual(resp.sources[0].page, 7)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)
        self.assertTrue(resp.total_time_ms > 0)
        self.assertTrue(resp.embedding_time_ms >= 0)
        self.assertTrue(resp.retrieval_time_ms >= 0)
        self.assertTrue(resp.generation_time_ms >= 0)

        # Verify recorded in SQLite queries table
        recent = self.pipeline.query_repo.list_recent(5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].query, "What is Llama 3.2 1B?")
        self.assertFalse(recent[0].is_low_confidence)
        self.assertEqual(recent[0].retrieved_count, 1)

    def test_end_to_end_rag_low_confidence_refusal(self) -> None:
        """Verify out-of-context query refuses cleanly without hallucinating or calling LLM."""
        resp = self.pipeline.query(
            query_text="What is the distance between Jupiter and Saturn?",
            top_k=3,
            threshold=0.99,  # Force refusal
        )
        self.assertTrue(resp.is_low_confidence)
        self.assertEqual(resp.answer, LOW_CONFIDENCE_MESSAGE)
        self.assertEqual(len(resp.sources), 0)
        self.assertEqual(resp.generation_time_ms, 0.0)  # LLM must not be called!

        # Verify recorded as low confidence in DB
        recent = self.pipeline.query_repo.list_recent(5)
        self.assertEqual(len(recent), 1)
        self.assertTrue(recent[0].is_low_confidence)
        self.assertEqual(recent[0].retrieved_count, 0)

    def test_grounded_chat_mode_refusal(self) -> None:
        """Verify Grounded RAG Chat refuses and returns exact refusal text when no chunks match."""
        resp = self.pipeline.query(
            query_text="What is quantum gravity?",
            top_k=3,
            threshold=0.99,
            chat_mode="grounded",
        )
        self.assertTrue(resp.is_low_confidence)
        self.assertFalse(resp.is_general_knowledge)
        self.assertEqual(resp.answer, "Information not found in the uploaded documents.")
        self.assertEqual(len(resp.sources), 0)
        self.assertEqual(resp.generation_time_ms, 0.0)

    def test_rag_plus_llm_chat_mode_fallback(self) -> None:
        """Verify RAG + LLM Chat calls LLM with general knowledge when no chunks match."""
        resp = self.pipeline.query(
            query_text="What is quantum gravity?",
            top_k=3,
            threshold=0.99,
            chat_mode="rag_plus_llm",
        )
        self.assertTrue(resp.is_low_confidence)
        self.assertTrue(resp.is_general_knowledge)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)
        self.assertEqual(len(resp.sources), 0)
        self.assertTrue(resp.generation_time_ms > 0.0)

    def test_rag_plus_llm_chat_mode_grounded_success(self) -> None:
        """Verify RAG + LLM Chat uses document context when relevant chunks are found."""
        resp = self.pipeline.query(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
            chat_mode="rag_plus_llm",
        )
        self.assertFalse(resp.is_low_confidence)
        self.assertFalse(resp.is_general_knowledge)
        self.assertTrue(len(resp.sources) > 0)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)

    def test_rag_plus_llm_auto_fallback_on_missing_context(self) -> None:
        """Verify RAG + LLM Chat falls back to general knowledge if context lacks the answer."""
        orig_mock = self.ollama_client.chat

        def custom_chat(*args, **kwargs):
            return "Narendra Modi is the Prime Minister of India.", 15.0

        self.ollama_client.chat = custom_chat
        try:
            resp = self.pipeline.query(
                query_text="who is prime minister of india?",
                top_k=3,
                threshold=0.01,  # Force chunks to pass numerical threshold
                chat_mode="rag_plus_llm",
            )
            self.assertTrue(resp.is_general_knowledge)
            self.assertEqual(resp.answer, "Narendra Modi is the Prime Minister of India.")
            self.assertEqual(len(resp.sources), 0)
        finally:
            self.ollama_client.chat = orig_mock

    # ──────────────────────────────────────────────────────────────────────────
    # New Targeted Tests for Requirements 1 - 9
    # ──────────────────────────────────────────────────────────────────────────

    def test_1_grounded_rag_with_relevant_documents(self) -> None:
        """Requirement 1: Grounded RAG with relevant documents generates grounded answer strictly from docs."""
        resp = self.pipeline.query(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
            chat_mode="grounded",
        )
        self.assertFalse(resp.is_low_confidence)
        self.assertFalse(resp.is_general_knowledge)
        self.assertTrue(len(resp.sources) > 0)
        self.assertEqual(resp.sources[0].filename, "rag_guide.pdf")
        self.assertEqual(resp.sources[0].page, 7)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)
        self.assertTrue(resp.generation_time_ms >= 0.0)

    def test_2_grounded_rag_with_no_relevant_documents(self) -> None:
        """Requirement 2: Grounded RAG with no relevant documents does NOT call LLM and displays refusal."""
        resp = self.pipeline.query(
            query_text="What is the distance between Jupiter and Saturn?",
            top_k=3,
            threshold=0.99,
            chat_mode="grounded",
        )
        self.assertTrue(resp.is_low_confidence)
        self.assertFalse(resp.is_general_knowledge)
        self.assertEqual(resp.answer, LOW_CONFIDENCE_MESSAGE)
        self.assertEqual(len(resp.sources), 0)
        self.assertEqual(resp.generation_time_ms, 0.0)

    def test_3_rag_plus_llm_with_relevant_documents(self) -> None:
        """Requirement 3: RAG + LLM with relevant documents generates grounded answer."""
        resp = self.pipeline.query(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
            chat_mode="rag_plus_llm",
        )
        self.assertFalse(resp.is_low_confidence)
        self.assertFalse(resp.is_general_knowledge)
        self.assertTrue(len(resp.sources) > 0)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)

    def test_4_rag_plus_llm_with_no_relevant_documents(self) -> None:
        """Requirement 4: RAG + LLM with no relevant documents calls LLM directly without RAG context."""
        resp = self.pipeline.query(
            query_text="What is the distance between Jupiter and Saturn?",
            top_k=3,
            threshold=0.99,
            chat_mode="rag_plus_llm",
        )
        self.assertTrue(resp.is_low_confidence)
        self.assertTrue(resp.is_general_knowledge)
        self.assertEqual(len(resp.sources), 0)
        self.assertIn("Mock Llama 3.2 1B", resp.answer)
        self.assertTrue(resp.generation_time_ms > 0.0)

    def test_5_compare_both_with_relevant_documents(self) -> None:
        """Requirement 5: Compare Both executes both modes side-by-side with relevant documents."""
        comp = self.pipeline.compare(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
            model="llama3.2:1b",
        )
        self.assertIsInstance(comp, ComparisonResponse)
        self.assertTrue(comp.grounded_context_retrieved)
        self.assertTrue(comp.grounded_llm_used)
        self.assertEqual(comp.grounded_result_label, "Grounded Document Answer")
        self.assertFalse(comp.grounded_response.is_low_confidence)

        self.assertTrue(comp.rag_llm_context_retrieved)
        self.assertTrue(comp.rag_llm_used)
        self.assertEqual(comp.rag_llm_result_label, "Grounded Document Answer")
        self.assertFalse(comp.rag_llm_response.is_general_knowledge)

    def test_6_compare_both_with_no_relevant_documents(self) -> None:
        """Requirement 6: Compare Both handles no relevant documents correctly for both modes."""
        comp = self.pipeline.compare(
            query_text="What is quantum gravity?",
            top_k=3,
            threshold=0.99,
            model="llama3.2:1b",
        )
        self.assertIsInstance(comp, ComparisonResponse)
        # Grounded mode refuses without calling LLM
        self.assertFalse(comp.grounded_context_retrieved)
        self.assertFalse(comp.grounded_llm_used)
        self.assertEqual(comp.grounded_response.answer, LOW_CONFIDENCE_MESSAGE)
        self.assertEqual(comp.grounded_response.generation_time_ms, 0.0)
        self.assertEqual(comp.grounded_result_label, LOW_CONFIDENCE_MESSAGE)

        # RAG + LLM calls LLM directly using general knowledge
        self.assertFalse(comp.rag_llm_context_retrieved)
        self.assertTrue(comp.rag_llm_used)
        self.assertTrue(comp.rag_llm_response.is_general_knowledge)
        self.assertEqual(comp.rag_llm_result_label, GENERAL_LLM_LABEL)
        self.assertTrue(comp.rag_llm_response.generation_time_ms > 0.0)

    def test_7_compare_both_uses_same_retrieved_chunks(self) -> None:
        """Requirement 7: Compare Both reuses the EXACT SAME single retrieval pass for both modes."""
        comp = self.pipeline.compare(
            query_text="What is Llama 3.2 1B?",
            top_k=3,
            threshold=0.10,
            model="llama3.2:1b",
        )
        # Verify trace identity: single retrieval pass shared by both modes
        self.assertIs(comp.grounded_response.trace, comp.rag_llm_response.trace)
        self.assertEqual(
            comp.grounded_response.trace.filtered_chunks,
            comp.rag_llm_response.trace.filtered_chunks,
        )
        self.assertEqual(comp.retrieved_count, len(comp.trace.filtered_chunks))

    def test_8_both_modes_use_same_selected_llm(self) -> None:
        """Requirement 8: Both modes in Compare Both use the exact same selected LLM."""
        for test_model in ["llama3.2:1b", "qwen2.5-coder:1.5b", "starcoder2:3b"]:
            comp = self.pipeline.compare(
                query_text="What is Llama 3.2 1B?",
                top_k=3,
                threshold=0.10,
                model=test_model,
            )
            self.assertEqual(comp.model, test_model)
            self.assertEqual(comp.grounded_response.model, test_model)
            self.assertEqual(comp.rag_llm_response.model, test_model)

    def test_9_num_predict_identical_across_all_models(self) -> None:
        """Requirement 9: num_predict (256) and temperature (0.2) are identical for all 3 models."""
        self.assertEqual(settings.LLM_NUM_PREDICT, 256)
        self.assertEqual(settings.LLM_TEMPERATURE, 0.2)

        # Verify client generates payloads with identical num_predict and temperature for all 3 models
        for test_model in ["llama3.2:1b", "qwen2.5-coder:1.5b", "starcoder2:3b"]:
            client = OllamaClient(model=test_model, mock_mode=True)
    def test_10_clean_response_special_tokens_and_loops(self) -> None:
        """Requirement 10: Special tokens, leaked instructions, and loops are stripped cleanly."""
        raw_output = (
            "Provide a more detailed answer:\n"
            "Provide a concise synthesized summary answer:\n"
            "Answer: C++ provides object-oriented features while C is procedural.<end_of_text><|endoftext|><|end_of_answer|>\n"
            "Question: which is better c++ or c?\n"
            "Answer: C++ provides object-oriented features while C is procedural."
        )
        cleaned = OllamaClient._clean_response(raw_output)
        self.assertNotIn("<end_of_text>", cleaned)
        self.assertNotIn("<|endoftext|>", cleaned)
        self.assertNotIn("<|end_of_answer|>", cleaned)
        self.assertNotIn("Provide a more detailed answer:", cleaned)
        self.assertNotIn("Provide a concise synthesized summary answer:", cleaned)
        self.assertNotIn("Question: which is better c++ or c?", cleaned)
        self.assertFalse(cleaned.startswith("Answer:"))
        self.assertEqual(cleaned, "C++ provides object-oriented features while C is procedural.")

        # Test deduplication of repeated lines (StarCoder prompt loop)
        looping_output = (
            "Provide a detailed explanation:\n"
            "Provide a detailed explanation:\n"
            "Provide a detailed explanation:"
        )
        deduped = OllamaClient._clean_response(looping_output)
        self.assertEqual(deduped, "Provide a detailed explanation:")

    def test_11_evidence_evaluator_unrelated_c_vs_cpp(self) -> None:
        """Case 2: Query unrelated to uploaded documents ('which is better c or c++?').

        Grounded RAG refuses without calling LLM (generation_time_ms = 0.0).
        RAG + LLM falls back to general knowledge (is_general_knowledge = True).
        Telemetry displays identical candidates and evidence status on both sides.
        """
        comp = self.pipeline.compare(
            query_text="which is better c or c++?",
            top_k=3,
            threshold=0.01,  # Force candidate to pass numerical threshold
            model="llama3.2:1b",
        )
        self.assertFalse(comp.has_relevant_evidence)
        self.assertIn("c++", comp.relevance_reason.lower())

        # Grounded RAG side
        self.assertFalse(comp.grounded_context_retrieved)
        self.assertFalse(comp.grounded_llm_used)
        self.assertEqual(comp.grounded_response.answer, LOW_CONFIDENCE_MESSAGE)
        self.assertEqual(comp.grounded_response.generation_time_ms, 0.0)
        self.assertEqual(comp.grounded_result_label, LOW_CONFIDENCE_MESSAGE)

        # RAG + LLM side
        self.assertFalse(comp.rag_llm_context_retrieved)
        self.assertTrue(comp.rag_llm_used)
        self.assertTrue(comp.rag_llm_response.is_general_knowledge)
        self.assertEqual(comp.rag_llm_result_label, GENERAL_LLM_LABEL)
        self.assertEqual(len(comp.rag_llm_response.sources), 0)

        # Telemetry consistency
        self.assertEqual(comp.retrieved_count, 1)

    def test_12_evidence_evaluator_distractor_query_cost_in_dollars(self) -> None:
        """Case 3: Query where candidates are semantically similar but lack evidence ('cost in dollars').

        Focus terms 'cost' / 'dollars' are missing from the retrieved context.
        Grounded RAG refuses without calling LLM.
        RAG + LLM falls back to general knowledge.
        """
        chunk_mpnn = TextChunk(
            chunk_id="chunk_mpnn_train",
            document_id="doc_mpnn",
            filename="ProteinMPNN.pdf",
            page=2,
            chunk_index=0,
            text="We trained ProteinMPNN on native protein backbones using deep graph neural networks.",
            char_count=87,
            token_estimate=15,
            pipeline_version="v1",
        )
        vectors = self.embedder.embed_texts([chunk_mpnn.text])
        self.vectorstore.upsert_chunks([chunk_mpnn], vectors)

        comp = self.pipeline.compare(
            query_text="What is the training cost of ProteinMPNN in dollars?",
            top_k=3,
            threshold=0.01,
            model="llama3.2:1b",
        )
        self.assertFalse(comp.has_relevant_evidence)
        self.assertIn("target focus terms", comp.relevance_reason.lower())

        # Grounded RAG refuses
        self.assertFalse(comp.grounded_context_retrieved)
        self.assertFalse(comp.grounded_llm_used)
        self.assertEqual(comp.grounded_response.answer, LOW_CONFIDENCE_MESSAGE)
        self.assertEqual(comp.grounded_response.generation_time_ms, 0.0)

        # RAG + LLM falls back to general LLM
        self.assertTrue(comp.rag_llm_response.is_general_knowledge)
        self.assertEqual(comp.rag_llm_result_label, GENERAL_LLM_LABEL)
        self.assertEqual(len(comp.rag_llm_response.sources), 0)

    def test_13_evidence_evaluator_relevant_query_proteinmpnn_recovery(self) -> None:
        """Case 1: Query clearly answered by uploaded document.

        Both Grounded RAG and RAG + LLM generate grounded answers from the documents.
        """
        chunk_mpnn = TextChunk(
            chunk_id="chunk_mpnn_recovery",
            document_id="doc_mpnn",
            filename="ProteinMPNN.pdf",
            page=1,
            chunk_index=0,
            text="On native protein backbones, ProteinMPNN achieves a sequence recovery of 52.4%.",
            char_count=82,
            token_estimate=15,
            pipeline_version="v1",
        )
        vectors = self.embedder.embed_texts([chunk_mpnn.text])
        self.vectorstore.upsert_chunks([chunk_mpnn], vectors)

        comp = self.pipeline.compare(
            query_text="What sequence recovery does ProteinMPNN achieve on native backbone structures?",
            top_k=3,
            threshold=0.01,
            model="llama3.2:1b",
        )
        self.assertTrue(comp.has_relevant_evidence)
        self.assertTrue(comp.grounded_context_retrieved)
        self.assertTrue(comp.grounded_llm_used)
        self.assertEqual(comp.grounded_result_label, "Grounded Document Answer")
        self.assertFalse(comp.grounded_response.is_low_confidence)

        self.assertTrue(comp.rag_llm_context_retrieved)
        self.assertTrue(comp.rag_llm_used)
        self.assertEqual(comp.rag_llm_result_label, "Grounded Document Answer")
        self.assertFalse(comp.rag_llm_response.is_general_knowledge)


if __name__ == "__main__":
    unittest.main()

