"""Vector storage for fact embeddings, backed by Chroma.

Stores each fact's embedding + metadata so we can do semantic similarity search
during consolidation (near-duplicate detection) and retrieval. We supply our
own embeddings (from the EmbeddingProvider) rather than letting Chroma pick a
model, so the LLM/embedding provider abstraction stays the single source of truth.

Chroma returns cosine *distance*; we convert to a [0, 1] similarity score so the
rest of the code can treat higher = more similar consistently.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.models import Fact
from app.embeddings.base import EmbeddingProvider
from app.storage.base import ScoredFact

logger = get_logger(__name__)


class ChromaVectorStore:
    """Thin wrapper over a persistent Chroma collection of fact embeddings."""

    def __init__(self, embeddings: EmbeddingProvider, persist_dir: str) -> None:
        import chromadb  # lazy import
        from chromadb.config import Settings as ChromaSettings

        self._embeddings = embeddings
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # cosine space so distance in [0, 2]; we map to similarity in [0, 1].
        self._collection = self._client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store: Chroma (persist_dir=%s)", persist_dir)

    def upsert(self, fact: Fact) -> None:
        fact = fact.ensure_id()
        vec = self._embeddings.embed_one(fact.text())
        self._collection.upsert(
            ids=[fact.fact_id],
            embeddings=[vec],
            documents=[fact.text()],
            metadatas=[
                {
                    "entity": fact.entity,
                    "attribute": fact.attribute,
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "session_id": fact.session_id,
                }
            ],
        )

    def delete(self, fact_id: str) -> None:
        self._collection.delete(ids=[fact_id])

    def search(self, query: str, *, top_k: int = 5) -> list[ScoredFact]:
        count = self._collection.count()
        if count == 0:
            return []
        qv = self._embeddings.embed_one(query)
        res = self._collection.query(
            query_embeddings=[qv],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        scored: list[ScoredFact] = []
        for fid, meta, dist in zip(ids, metas, dists):
            fact = Fact(
                entity=meta.get("entity", ""),
                attribute=meta.get("attribute", ""),
                value=meta.get("value", ""),
                confidence=meta.get("confidence", 1.0),
                fact_id=fid,
                session_id=meta.get("session_id", ""),
            )
            scored.append(ScoredFact(fact=fact, score=_distance_to_similarity(dist)))
        return scored

    def clear(self) -> None:
        # Delete and recreate the collection to wipe all vectors.
        self._client.delete_collection("facts")
        self._collection = self._client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"},
        )


def _distance_to_similarity(distance: float) -> float:
    """Map Chroma cosine distance ([0, 2]) to a similarity score ([0, 1])."""
    sim = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, sim))
