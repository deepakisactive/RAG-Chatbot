"""Qdrant Vector Database Client.

Manages collection lifecycle, payload indexes, vector upserting,
cosine similarity search, and payload-filtered deletion.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ingestion.chunker import TextChunk
from app.logging_config import logger, timed_operation

# Attempt importing official qdrant_client
try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
    QDRANT_INSTALLED = True
except ImportError:
    QdrantClient = None
    qmodels = None
    QDRANT_INSTALLED = False
    logger.warning("qdrant_client library is not installed in the current environment.")


@dataclass
class ScoredChunk:
    """Represents a retrieved chunk with similarity score and rank."""

    chunk_id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    similarity_score: float
    rank: int
    pipeline_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VectorStoreClient:
    """High-level wrapper around Qdrant client."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        use_in_memory: bool = False,
    ) -> None:
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.use_in_memory = use_in_memory or not QDRANT_INSTALLED

        # In-memory storage fallback for offline testing or when Qdrant unreachable
        self._in_memory_points: Dict[str, Dict[str, Any]] = {}
        self.client: Optional[Any] = None

        if not self.use_in_memory and QDRANT_INSTALLED:
            candidates = [self.url]
            if "://qdrant:" in self.url:
                candidates.append(self.url.replace("://qdrant:", "://localhost:"))
            elif "://localhost:" in self.url:
                candidates.append(self.url.replace("://localhost:", "://qdrant:"))

            connected = False
            for target_url in candidates:
                try:
                    self.client = QdrantClient(url=target_url, timeout=5.0, check_compatibility=False)
                    self.ensure_collection_exists()
                    self.url = target_url
                    connected = True
                    break
                except Exception:
                    continue

            if not connected:
                logger.warning(
                    f"Could not connect to Qdrant at {self.url}. Using in-memory fallback store."
                )
                self.use_in_memory = True
                self.client = None

    def ensure_collection_exists(self) -> None:
        """Create the collection and required payload indices if not present."""
        if self.use_in_memory or not self.client:
            return

        try:
            collections_res = self.client.get_collections()
            existing_names = [c.name for c in collections_res.collections]

            if self.collection_name not in existing_names:
                logger.info(
                    f"Creating Qdrant collection '{self.collection_name}' with size {self.dimension}, Cosine distance."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                # Create payload indices for fast filtered queries and deletions
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="page",
                    field_schema=qmodels.PayloadSchemaType.INTEGER,
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="pipeline_version",
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
                logger.info(f"Qdrant collection '{self.collection_name}' created successfully.")
        except Exception as exc:
            logger.error(f"Error ensuring Qdrant collection: {exc}")
            raise

    def upsert_chunks(
        self,
        chunks: List[TextChunk],
        vectors: List[List[float]],
    ) -> int:
        """Upsert a batch of text chunks and their dense vector embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided with {len(vectors)} vectors."
            )
        if not chunks:
            return 0

        with timed_operation("qdrant_upsert", {"count": len(chunks)}):
            if self.use_in_memory or not self.client:
                for chunk, vector in zip(chunks, vectors):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                    self._in_memory_points[point_id] = {
                        "id": point_id,
                        "vector": vector,
                        "payload": chunk.to_dict(),
                    }
                return len(chunks)

            points = []
            for chunk, vector in zip(chunks, vectors):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=chunk.to_dict(),
                    )
                )

            # Upsert in batches of 100 points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )

            return len(chunks)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Execute cosine similarity search against Qdrant."""
        with timed_operation("qdrant_search", {"top_k": top_k, "doc_filter": document_id}):
            if self.use_in_memory or not self.client:
                # In-memory cosine search calculation
                scored: List[tuple[float, Dict[str, Any]]] = []
                for pt in self._in_memory_points.values():
                    if document_id and pt["payload"].get("document_id") != document_id:
                        continue
                    v = pt["vector"]
                    # Cosine similarity for normalized vectors: dot product
                    sim = sum(a * b for a, b in zip(query_vector, v))
                    # Bound to [-1.0, 1.0]
                    sim = max(-1.0, min(1.0, sim))
                    scored.append((sim, pt["payload"]))

                scored.sort(key=lambda x: x[0], reverse=True)
                top_matches = scored[:top_k]

                results: List[ScoredChunk] = []
                for rank, (score, p) in enumerate(top_matches, start=1):
                    results.append(
                        ScoredChunk(
                            chunk_id=p["chunk_id"],
                            document_id=p["document_id"],
                            filename=p["filename"],
                            page=p["page"],
                            chunk_index=p["chunk_index"],
                            text=p["text"],
                            similarity_score=round(score, 4),
                            rank=rank,
                            pipeline_version=p.get("pipeline_version", "v1"),
                        )
                    )
                return results

            # Official Qdrant search
            query_filter = None
            if document_id:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )

            if hasattr(self.client, "query_points"):
                query_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
                hits = query_res.points if hasattr(query_res, "points") else query_res
            else:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )

            results = []
            for rank, hit in enumerate(hits, start=1):
                p = hit.payload or {}
                results.append(
                    ScoredChunk(
                        chunk_id=p.get("chunk_id", ""),
                        document_id=p.get("document_id", ""),
                        filename=p.get("filename", ""),
                        page=p.get("page", 1),
                        chunk_index=p.get("chunk_index", 0),
                        text=p.get("text", ""),
                        similarity_score=round(hit.score, 4),
                        rank=rank,
                        pipeline_version=p.get("pipeline_version", "v1"),
                    )
                )
            return results

    def delete_document_vectors(self, document_id: str) -> bool:
        """Delete all vector points belonging to a specific document_id."""
        with timed_operation("qdrant_delete_vectors", {"document_id": document_id}):
            if self.use_in_memory or not self.client:
                initial_count = len(self._in_memory_points)
                self._in_memory_points = {
                    pid: pt
                    for pid, pt in self._in_memory_points.items()
                    if pt["payload"].get("document_id") != document_id
                }
                removed = initial_count - len(self._in_memory_points)
                logger.info(f"Removed {removed} points for doc '{document_id}' from in-memory store.")
                return True

            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="document_id",
                                    match=qmodels.MatchValue(value=document_id),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
                logger.info(f"Deleted vector points for '{document_id}' from Qdrant.")
                return True
            except Exception as exc:
                logger.error(f"Failed to delete points for '{document_id}' in Qdrant: {exc}")
                raise

    def count_vectors(self) -> int:
        """Count total vectors currently indexed in the collection."""
        if self.use_in_memory or not self.client:
            return len(self._in_memory_points)
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def count_document_vectors(self, document_id: str) -> int:
        """Count how many vectors exist in Qdrant for a specific document_id."""
        if self.use_in_memory or not self.client:
            return sum(
                1
                for pt in self._in_memory_points.values()
                if pt.get("payload", {}).get("document_id") == document_id
            )
        try:
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
                exact=True,
            )
            return res.count
        except Exception:
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant server health and connectivity status."""
        if self.use_in_memory or not self.client:
            return {
                "status": "connected" if self.use_in_memory else "disconnected",
                "mode": "in-memory-fallback" if self.use_in_memory else "offline",
                "collection": self.collection_name,
                "vector_count": len(self._in_memory_points),
            }
        try:
            collections = self.client.get_collections()
            return {
                "status": "connected",
                "mode": "qdrant-server",
                "url": self.url,
                "collection": self.collection_name,
                "vector_count": self.count_vectors(),
                "available_collections": [c.name for c in collections.collections],
            }
        except Exception as exc:
            return {
                "status": "disconnected",
                "url": self.url,
                "error": str(exc),
            }


_vectorstore_instance: Optional[VectorStoreClient] = None


def get_vectorstore(use_in_memory: bool = False) -> VectorStoreClient:
    """Retrieve or initialize singleton VectorStoreClient."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreClient(use_in_memory=use_in_memory)
    return _vectorstore_instance
