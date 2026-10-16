"""RAG Retrieval Evaluation Engine.

Measures quantitative retrieval metrics including:
- Recall@K: proportion of test cases where relevant document/page appears in Top-K
- Precision@K: proportion of retrieved chunks that are relevant
- Mean Reciprocal Rank (MRR): reciprocal rank of the first relevant chunk
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from app.rag.retriever import RetrievalEngine
from app.vectorstore.qdrant_client import ScoredChunk


STOPWORDS = {
    "what", "is", "the", "of", "in", "does", "which", "better", "or", "and", "a", "an",
    "on", "for", "to", "from", "with", "by", "at", "how", "why", "when", "where", "who",
    "whom", "can", "could", "would", "should", "do", "did", "are", "were", "was", "be",
    "been", "being", "have", "has", "had", "achieve", "give", "tell", "explain", "describe",
    "about", "between", "using", "used", "use", "than", "this", "that", "these", "those",
    "compare", "difference", "versus", "vs", "much", "many", "some", "any", "all",
}

SPECIFIC_TARGETS = {
    "cost", "costs", "dollar", "dollars", "price", "prices", "pricing", "fee", "fees",
    "phone", "telephone", "mobile", "email", "contact", "address", "salary",
    "revenue", "profit", "net worth", "birthday", "birth", "died", "death",
    "license", "founder", "ceo", "headquarters", "location",
}

SPECIAL_ENTITIES = ["c++", "c#", ".net", "objective-c", "golang"]


@dataclass
class EvidenceDecision:
    """Represents the factual grounding verification decision."""

    has_relevant_evidence: bool
    reason: str
    token_coverage: float
    max_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceEvaluator:
    """Determines whether retrieved chunks contain sufficient evidence to answer a query.

    Distinguishes between retrieved candidates and actual relevant evidence.
    Prevents hallucination and strict grounding violations when similarity scores
    pass numeric thresholds on topically related but non-answering context.
    """

    @classmethod
    def evaluate(
        cls,
        query: str,
        chunks: Union[List[ScoredChunk], List[str]],
        max_similarity: float,
        threshold: float,
    ) -> EvidenceDecision:
        """Evaluate evidence presence in retrieved candidates.

        Args:
            query: The user's input query.
            chunks: List of ScoredChunk objects or text strings.
            max_similarity: Highest cosine similarity among candidates.
            threshold: Current similarity threshold setting.

        Returns:
            EvidenceDecision indicating whether evidence is present and reasoning.
        """
        if max_similarity < threshold or not chunks:
            return EvidenceDecision(
                has_relevant_evidence=False,
                reason="Similarity below threshold or no candidates retrieved",
                token_coverage=0.0,
                max_similarity=max_similarity,
            )

        # Extract combined text
        if chunks and isinstance(chunks[0], str):
            combined_text = " ".join(chunks)  # type: ignore
        else:
            combined_text = " ".join(c.text for c in chunks)  # type: ignore

        lower_query = query.lower()
        lower_text = combined_text.lower()

        # 1. Check special entities with punctuation (e.g. c++, c#, .net)
        for ent in SPECIAL_ENTITIES:
            if ent in lower_query:
                pattern = r"(?:^|\W)" + re.escape(ent) + r"(?:$|\W)"
                if not re.search(pattern, lower_text):
                    return EvidenceDecision(
                        has_relevant_evidence=False,
                        reason=f"Special entity '{ent}' not present in retrieved context",
                        token_coverage=0.0,
                        max_similarity=max_similarity,
                    )

        # 2. Check specific factual target attribute constraints (cost, dollars, phone, etc.)
        query_targets = [
            t for t in SPECIFIC_TARGETS
            if re.search(r"\b" + re.escape(t) + r"\b", lower_query)
        ]
        if query_targets:
            target_found = any(
                re.search(r"\b" + re.escape(t) + r"\b", lower_text)
                for t in query_targets
            )
            if not target_found:
                return EvidenceDecision(
                    has_relevant_evidence=False,
                    reason=f"Target focus terms {query_targets} not found in retrieved context",
                    token_coverage=0.0,
                    max_similarity=max_similarity,
                )

        # 3. Content tokens coverage check
        raw_tokens = re.findall(r"[a-zA-Z0-9_\+\#\.]+", lower_query)
        content_tokens = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]

        if not content_tokens:
            return EvidenceDecision(
                has_relevant_evidence=True,
                reason="General query without specific content terms",
                token_coverage=1.0,
                max_similarity=max_similarity,
            )

        matches = [
            t for t in content_tokens
            if re.search(r"\b" + re.escape(t) + r"\b", lower_text)
        ]
        coverage = len(matches) / len(content_tokens)

        # If query has multiple content tokens and fewer than 2 match
        if len(content_tokens) >= 2 and len(matches) < 2 and max_similarity < 0.85:
            return EvidenceDecision(
                has_relevant_evidence=False,
                reason=f"Insufficient query token matches ({len(matches)}/{len(content_tokens)}, sim={max_similarity:.4f})",
                token_coverage=round(coverage, 4),
                max_similarity=max_similarity,
            )

        # Token coverage cutoff
        if coverage < 0.60 and max_similarity < 0.85:
            return EvidenceDecision(
                has_relevant_evidence=False,
                reason=f"Insufficient query token coverage ({coverage:.2f} < 0.60, sim={max_similarity:.4f})",
                token_coverage=round(coverage, 4),
                max_similarity=max_similarity,
            )

        return EvidenceDecision(
            has_relevant_evidence=True,
            reason=f"Evidence verified (coverage={coverage:.2f}, sim={max_similarity:.4f})",
            token_coverage=round(coverage, 4),
            max_similarity=max_similarity,
        )


@dataclass
class EvalSample:
    """Represents a benchmark test case for RAG evaluation."""

    query: str
    expected_answer: str
    relevant_document: str
    relevant_page: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Individual query evaluation result."""

    query: str
    relevant_doc: str
    relevant_page: int
    retrieved_count: int
    hit_at_k: bool
    reciprocal_rank: float
    precision_at_k: float
    highest_similarity: float
    is_low_confidence: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    """Computes Recall@K, Precision@K, and MRR across benchmark datasets."""

    def __init__(self, retriever: Optional[RetrievalEngine] = None) -> None:
        self.retriever = retriever or RetrievalEngine()

    def evaluate_dataset(
        self,
        dataset: List[EvalSample],
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance across a collection of test samples."""
        if not dataset:
            return {"error": "Empty evaluation dataset"}

        sample_results: List[EvalResult] = []
        hits = 0
        total_rr = 0.0
        total_precision = 0.0

        for sample in dataset:
            trace = self.retriever.retrieve(
                query=sample.query,
                top_k=top_k,
                threshold=threshold,
            )

            # Check for matches
            first_hit_rank: Optional[int] = None
            relevant_chunks_count = 0

            for chunk in trace.retrieved_chunks:
                # Match on filename (or doc_id) and page
                doc_match = (
                    sample.relevant_document.lower() in chunk.filename.lower()
                    or chunk.filename.lower() in sample.relevant_document.lower()
                )
                page_match = sample.relevant_page == chunk.page

                if doc_match and page_match:
                    relevant_chunks_count += 1
                    if first_hit_rank is None:
                        first_hit_rank = chunk.rank

            is_hit = first_hit_rank is not None
            if is_hit:
                hits += 1
                rr = 1.0 / first_hit_rank
            else:
                rr = 0.0

            total_rr += rr
            prec = relevant_chunks_count / len(trace.retrieved_chunks) if trace.retrieved_chunks else 0.0
            total_precision += prec

            sample_results.append(
                EvalResult(
                    query=sample.query,
                    relevant_doc=sample.relevant_document,
                    relevant_page=sample.relevant_page,
                    retrieved_count=len(trace.retrieved_chunks),
                    hit_at_k=is_hit,
                    reciprocal_rank=round(rr, 4),
                    precision_at_k=round(prec, 4),
                    highest_similarity=trace.highest_similarity,
                    is_low_confidence=trace.is_low_confidence,
                )
            )

        n = len(dataset)
        recall_at_k = round(hits / n, 4)
        mean_reciprocal_rank = round(total_rr / n, 4)
        avg_precision = round(total_precision / n, 4)

        return {
            "sample_count": n,
            "top_k": top_k,
            "threshold": threshold,
            "recall_at_k": recall_at_k,
            "mean_reciprocal_rank": mean_reciprocal_rank,
            "avg_precision_at_k": avg_precision,
            "detailed_results": [r.to_dict() for r in sample_results],
        }
