"""Graph storage for facts.

Facts are stored as a graph so we can do exact (entity, attribute) lookups and,
crucially, *traverse* from one entity to related entities during retrieval.

Model:
    (:Entity {name})  -[:ATTR {attribute, fact_id, value, confidence, ...}]->  (:Value {text})

Each fact becomes an edge from its entity node to a value node, labelled with
the attribute. Two implementations sit behind one small interface:

  * ``NetworkXGraphStore``  — in-process, zero infra, the default for local/tests.
  * ``Neo4jGraphStore``     — real graph database via docker-compose.

Only the composite store (memory_store.py) and retrieval talk to this; they use
the abstract ``GraphStore`` so swapping backends changes nothing upstream.
"""

from __future__ import annotations

import abc

from app.core.logging import get_logger
from app.core.models import Fact

logger = get_logger(__name__)


class GraphStore(abc.ABC):
    """Abstract graph backend storing facts as entity->value edges."""

    @abc.abstractmethod
    def upsert_fact(self, fact: Fact) -> None:
        """Insert or replace a fact edge keyed by fact_id."""

    @abc.abstractmethod
    def delete_fact(self, fact_id: str) -> None:
        ...

    @abc.abstractmethod
    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        """All facts for an exact (entity, attribute) pair."""

    @abc.abstractmethod
    def get_entity_facts(self, entity: str) -> list[Fact]:
        """All facts whose subject is ``entity``."""

    @abc.abstractmethod
    def related_entities(self, entity: str, *, max_hops: int = 1) -> list[str]:
        """Entities reachable from ``entity`` within ``max_hops`` (for traversal)."""

    @abc.abstractmethod
    def all_facts(self) -> list[Fact]:
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        ...

    @abc.abstractmethod
    def name(self) -> str:
        ...


# --------------------------------------------------------------------------- #
# NetworkX implementation (in-process, zero infra)
# --------------------------------------------------------------------------- #
class NetworkXGraphStore(GraphStore):
    """Directed multigraph in memory. Great for local dev, tests, and CI."""

    def __init__(self) -> None:
        import networkx as nx  # lazy import

        self._nx = nx
        # MultiDiGraph: multiple labelled edges allowed between two nodes.
        self._g = nx.MultiDiGraph()

    def _entity_node(self, name: str) -> str:
        return f"entity::{name.strip().lower()}"

    def _value_node(self, value: str) -> str:
        return f"value::{value.strip().lower()}"

    def upsert_fact(self, fact: Fact) -> None:
        fact = fact.ensure_id()
        # Remove any prior edge with the same fact_id (upsert semantics).
        self.delete_fact(fact.fact_id)

        en = self._entity_node(fact.entity)
        vn = self._value_node(fact.value)
        self._g.add_node(en, kind="entity", name=fact.entity)
        self._g.add_node(vn, kind="value", text=fact.value)
        self._g.add_edge(
            en,
            vn,
            key=fact.fact_id,
            fact_id=fact.fact_id,
            entity=fact.entity,
            attribute=fact.attribute,
            value=fact.value,
            confidence=fact.confidence,
            session_id=fact.session_id,
        )

    def delete_fact(self, fact_id: str) -> None:
        for u, v, k in list(self._g.edges(keys=True)):
            if k == fact_id:
                self._g.remove_edge(u, v, key=k)

    def _edge_to_fact(self, data: dict) -> Fact:
        return Fact(
            entity=data["entity"],
            attribute=data["attribute"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            fact_id=data.get("fact_id", ""),
            session_id=data.get("session_id", ""),
        )

    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        e, a = entity.strip().lower(), attribute.strip().lower()
        out: list[Fact] = []
        for _, _, data in self._g.edges(data=True):
            if data.get("entity", "").lower() == e and data.get("attribute", "").lower() == a:
                out.append(self._edge_to_fact(data))
        return out

    def get_entity_facts(self, entity: str) -> list[Fact]:
        en = self._entity_node(entity)
        if en not in self._g:
            return []
        return [self._edge_to_fact(data) for _, _, data in self._g.out_edges(en, data=True)]

    def related_entities(self, entity: str, *, max_hops: int = 1) -> list[str]:
        """Find other entities that share a value node with ``entity``.

        Two entities are "related" if they point at the same value (e.g. both
        live in Bangalore). We walk entity -> value -> other entities.
        """
        en = self._entity_node(entity)
        if en not in self._g:
            return []
        related: set[str] = set()
        frontier = {en}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                # entity -> value nodes
                for _, value_node in self._g.out_edges(node):
                    # value node -> back to other entities
                    for other_entity, _ in self._g.in_edges(value_node):
                        data = self._g.nodes[other_entity]
                        if data.get("kind") == "entity" and other_entity != en:
                            related.add(data.get("name", ""))
                            next_frontier.add(other_entity)
            frontier = next_frontier
        return sorted(r for r in related if r)

    def all_facts(self) -> list[Fact]:
        return [self._edge_to_fact(data) for _, _, data in self._g.edges(data=True)]

    def clear(self) -> None:
        self._g.clear()

    def name(self) -> str:
        return "networkx"


# --------------------------------------------------------------------------- #
# Neo4j implementation (real graph DB via docker-compose)
# --------------------------------------------------------------------------- #
class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store using the official bolt driver."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase  # lazy import

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()
        logger.info("Graph store: Neo4j (%s)", uri)

    def _ensure_constraints(self) -> None:
        with self._driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def upsert_fact(self, fact: Fact) -> None:
        fact = fact.ensure_id()
        with self._driver.session() as s:
            s.run(
                """
                MERGE (e:Entity {name: $entity})
                MERGE (v:Value {text: $value})
                MERGE (e)-[r:ATTR {fact_id: $fact_id}]->(v)
                SET r.attribute = $attribute,
                    r.value = $value,
                    r.confidence = $confidence,
                    r.session_id = $session_id
                """,
                entity=fact.entity.lower(),
                value=fact.value,
                fact_id=fact.fact_id,
                attribute=fact.attribute.lower(),
                confidence=fact.confidence,
                session_id=fact.session_id,
            )

    def delete_fact(self, fact_id: str) -> None:
        with self._driver.session() as s:
            s.run("MATCH ()-[r:ATTR {fact_id: $fid}]->() DELETE r", fid=fact_id)

    def _rows_to_facts(self, rows) -> list[Fact]:
        return [
            Fact(
                entity=r["entity"],
                attribute=r["attribute"],
                value=r["value"],
                confidence=r.get("confidence", 1.0),
                fact_id=r.get("fact_id", ""),
                session_id=r.get("session_id", ""),
            )
            for r in rows
        ]

    def get_by_key(self, entity: str, attribute: str) -> list[Fact]:
        with self._driver.session() as s:
            res = s.run(
                """
                MATCH (e:Entity {name: $entity})-[r:ATTR {attribute: $attribute}]->(v:Value)
                RETURN e.name AS entity, r.attribute AS attribute, r.value AS value,
                       r.confidence AS confidence, r.fact_id AS fact_id, r.session_id AS session_id
                """,
                entity=entity.lower(),
                attribute=attribute.lower(),
            )
            return self._rows_to_facts([dict(x) for x in res])

    def get_entity_facts(self, entity: str) -> list[Fact]:
        with self._driver.session() as s:
            res = s.run(
                """
                MATCH (e:Entity {name: $entity})-[r:ATTR]->(v:Value)
                RETURN e.name AS entity, r.attribute AS attribute, r.value AS value,
                       r.confidence AS confidence, r.fact_id AS fact_id, r.session_id AS session_id
                """,
                entity=entity.lower(),
            )
            return self._rows_to_facts([dict(x) for x in res])

    def related_entities(self, entity: str, *, max_hops: int = 1) -> list[str]:
        with self._driver.session() as s:
            res = s.run(
                """
                MATCH (e:Entity {name: $entity})-[:ATTR]->(v:Value)<-[:ATTR]-(other:Entity)
                WHERE other.name <> $entity
                RETURN DISTINCT other.name AS name
                """,
                entity=entity.lower(),
            )
            return sorted(x["name"] for x in res if x["name"])

    def all_facts(self) -> list[Fact]:
        with self._driver.session() as s:
            res = s.run(
                """
                MATCH (e:Entity)-[r:ATTR]->(v:Value)
                RETURN e.name AS entity, r.attribute AS attribute, r.value AS value,
                       r.confidence AS confidence, r.fact_id AS fact_id, r.session_id AS session_id
                """
            )
            return self._rows_to_facts([dict(x) for x in res])

    def clear(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def name(self) -> str:
        return "neo4j"
