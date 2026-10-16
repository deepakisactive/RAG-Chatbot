"""Retrieval Engine Module.

Executes semantic search against the vector database, tracks latency metrics,
enforces similarity threshold cutoffs, and produces transparent retrieval traces.
"""

from __future__ import annotations

import time
from typing import Optional

from app.config import settings
from app.ingestion.embedder import BaseEmbedder, get_embedder
from app.logging_config import logger, timed_operation
from app.rag.models import RetrievalTrace
from app.vectorstore.qdrant_client import VectorStoreClient, get_vectorstore


class RetrievalEngine:
    """Coordinates query embedding, vector search, and confidence thresholding."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vectorstore: Optional[VectorStoreClient] = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.vectorstore = vectorstore or get_vectorstore()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalTrace:
        """Execute semantic retrieval with latency tracking and threshold filtering."""
        clean_query = query.strip()
        active_top_k = top_k if top_k is not None else settings.TOP_K
        active_threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        if not clean_query:
            return RetrievalTrace(
                query=query,
                top_k=active_top_k,
                threshold=active_threshold,
                distance_metric=settings.DISTANCE_METRIC,
                is_low_confidence=True,
            )

        # 1. Embed query
        t_embed_start = time.perf_counter()
        query_vector = self.embedder.embed_query(clean_query)
        embedding_time_ms = round((time.perf_counter() - t_embed_start) * 1000.0, 2)

        # 2. Vector search in Qdrant
        t_search_start = time.perf_counter()
        retrieved_chunks = self.vectorstore.search(
            query_vector=query_vector,
            top_k=active_top_k,
            document_id=document_id,
        )
        retrieval_time_ms = round((time.perf_counter() - t_search_start) * 1000.0, 2)

        # 3. Calculate metrics and threshold filtering
        highest_similarity = (
            max((c.similarity_score for c in retrieved_chunks), default=0.0)
            if retrieved_chunks
            else 0.0
        )

        filtered_chunks = [
            c for c in retrieved_chunks if c.similarity_score >= active_threshold
        ]

        is_low_confidence = (
            len(retrieved_chunks) == 0 or highest_similarity < active_threshold
        )

        logger.info(
            f"Retrieval for '{clean_query[:40]}...': "
            f"retrieved={len(retrieved_chunks)}, filtered={len(filtered_chunks)}, "
            f"max_sim={highest_similarity:.4f}, threshold={active_threshold}, "
            f"embed={embedding_time_ms}ms, search={retrieval_time_ms}ms, "
            f"low_conf={is_low_confidence}"
        )

        return RetrievalTrace(
            query=clean_query,
            top_k=active_top_k,
            threshold=active_threshold,
            distance_metric=settings.DISTANCE_METRIC,
            retrieved_chunks=retrieved_chunks,
            filtered_chunks=filtered_chunks,
            highest_similarity=round(highest_similarity, 4),
            is_low_confidence=is_low_confidence,
            embedding_time_ms=embedding_time_ms,
            retrieval_time_ms=retrieval_time_ms,
        )
