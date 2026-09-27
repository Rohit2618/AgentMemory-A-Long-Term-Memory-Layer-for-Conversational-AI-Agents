"""Storage factory — assembles the configured MemoryStore.

``get_memory_store()`` returns a process-wide composite store built from:
  * a graph backend chosen by GRAPH_PROVIDER (networkx | neo4j), and
  * a Chroma vector store using the configured embedding provider.

The consolidation engine, retrieval and agent all obtain their store from here,
so switching graph backends is a one-line config change.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.embeddings.factory import get_embeddings
from app.storage.base import MemoryStore
from app.storage.graph_store import GraphStore, NetworkXGraphStore, Neo4jGraphStore
from app.storage.memory_store import GraphVectorMemoryStore
from app.storage.vector_store import ChromaVectorStore

logger = get_logger(__name__)


def _build_graph(settings: Settings) -> GraphStore:
    provider = settings.graph_provider.lower().strip()
    if provider == "neo4j":
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
    if provider == "networkx":
        logger.info("Graph store: NetworkX (in-process)")
        return NetworkXGraphStore()
    raise ValueError(
        f"Unknown GRAPH_PROVIDER={settings.graph_provider!r}. Expected 'neo4j' or 'networkx'."
    )


def build_memory_store(settings: Settings) -> MemoryStore:
    """Construct a composite graph+vector MemoryStore (no caching)."""
    graph = _build_graph(settings)
    vectors = ChromaVectorStore(
        embeddings=get_embeddings(),
        persist_dir=settings.chroma_persist_dir,
    )
    return GraphVectorMemoryStore(graph=graph, vectors=vectors)


@lru_cache
def get_memory_store() -> MemoryStore:
    """Return a cached composite MemoryStore built from active settings."""
    return build_memory_store(get_settings())
