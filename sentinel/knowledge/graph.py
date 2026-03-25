"""Structured knowledge graph — entity relationships using NetworkX."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    entity_type: str  # "function", "class", "module", "file", "concept", "person", etc.
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class Relationship(BaseModel):
    """An edge in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str
    target_id: str
    relation_type: str  # "calls", "imports", "depends_on", "contains", "related_to", etc.
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class GraphQueryResult(BaseModel):
    """Result from a graph query."""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)  # Lists of entity IDs


class KnowledgeGraph:
    """NetworkX-backed knowledge graph for entity relationships.

    Complements the vector store (which handles unstructured text similarity)
    with structured relationships like:
      - "function X calls function Y"
      - "module A depends on module B"
      - "class C implements interface D"
      - "concept E is related to concept F"

    Persists to a JSON file for simplicity.
    """

    def __init__(self, persist_path: str = "./data/knowledge_graph.json"):
        self._persist_path = Path(persist_path)
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph = None
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._load()

    def _get_graph(self):
        if self._graph is None:
            import networkx as nx
            self._graph = nx.DiGraph()
        return self._graph

    def _load(self) -> None:
        """Load graph state from disk."""
        if not self._persist_path.exists():
            return

        try:
            data = json.loads(self._persist_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        graph = self._get_graph()

        for e_data in data.get("entities", []):
            entity = Entity(**e_data)
            self._entities[entity.id] = entity
            graph.add_node(entity.id, **entity.model_dump())

        for r_data in data.get("relationships", []):
            rel = Relationship(**r_data)
            self._relationships[rel.id] = rel
            graph.add_edge(
                rel.source_id, rel.target_id,
                key=rel.id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                **rel.properties,
            )

    def _save(self) -> None:
        """Persist graph state to disk."""
        data = {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relationships": [r.model_dump() for r in self._relationships.values()],
        }
        self._persist_path.write_text(json.dumps(data, indent=2))

    def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
        entity_id: str | None = None,
    ) -> Entity:
        """Add an entity (node) to the graph. Returns existing if name+type match."""
        # Check for existing entity with same name and type
        for existing in self._entities.values():
            if existing.name == name and existing.entity_type == entity_type:
                if properties:
                    existing.properties.update(properties)
                    self._save()
                return existing

        entity = Entity(
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        if entity_id:
            entity.id = entity_id

        self._entities[entity.id] = entity
        graph = self._get_graph()
        graph.add_node(entity.id, **entity.model_dump())
        self._save()
        return entity

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> Relationship | None:
        """Add a relationship (edge) between two entities."""
        if source_id not in self._entities or target_id not in self._entities:
            return None

        # Check for existing identical relationship
        for existing in self._relationships.values():
            if (existing.source_id == source_id
                    and existing.target_id == target_id
                    and existing.relation_type == relation_type):
                existing.weight = weight
                if properties:
                    existing.properties.update(properties)
                self._save()
                return existing

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            properties=properties or {},
        )

        self._relationships[rel.id] = rel
        graph = self._get_graph()
        graph.add_edge(
            source_id, target_id,
            key=rel.id,
            relation_type=relation_type,
            weight=weight,
            **(properties or {}),
        )
        self._save()
        return rel

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Find entities by name (substring match) and/or type."""
        results = []
        for entity in self._entities.values():
            if name and name.lower() not in entity.name.lower():
                continue
            if entity_type and entity.entity_type != entity_type:
                continue
            results.append(entity)
            if len(results) >= limit:
                break
        return results

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> GraphQueryResult:
        """Get entities connected to a given entity."""
        if entity_id not in self._entities:
            return GraphQueryResult()

        graph = self._get_graph()
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        seen_ids: set[str] = set()

        # Outgoing edges
        if direction in ("outgoing", "both"):
            for _, target, data in graph.out_edges(entity_id, data=True):
                rel_type = data.get("relation_type", "")
                if relation_type and rel_type != relation_type:
                    continue
                if target not in seen_ids:
                    entity = self._entities.get(target)
                    if entity:
                        entities.append(entity)
                        seen_ids.add(target)
                # Find matching relationship
                for rel in self._relationships.values():
                    if (rel.source_id == entity_id and rel.target_id == target
                            and rel.relation_type == rel_type):
                        relationships.append(rel)
                        break

        # Incoming edges
        if direction in ("incoming", "both"):
            for source, _, data in graph.in_edges(entity_id, data=True):
                rel_type = data.get("relation_type", "")
                if relation_type and rel_type != relation_type:
                    continue
                if source not in seen_ids:
                    entity = self._entities.get(source)
                    if entity:
                        entities.append(entity)
                        seen_ids.add(source)
                for rel in self._relationships.values():
                    if (rel.source_id == source and rel.target_id == entity_id
                            and rel.relation_type == rel_type):
                        relationships.append(rel)
                        break

        return GraphQueryResult(entities=entities, relationships=relationships)

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find shortest path between two entities. Returns list of entity IDs."""
        if source_id not in self._entities or target_id not in self._entities:
            return None

        graph = self._get_graph()
        try:
            import networkx as nx
            path = nx.shortest_path(graph, source_id, target_id)
            if len(path) - 1 > max_depth:
                return None
            return path
        except Exception:
            return None

    def get_subgraph(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> GraphQueryResult:
        """Get a subgraph centered on an entity, up to a given depth."""
        if entity_id not in self._entities:
            return GraphQueryResult()

        graph = self._get_graph()
        import networkx as nx

        # BFS to find all nodes within depth
        visited: set[str] = set()
        frontier = {entity_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                next_frontier.update(graph.successors(node))
                next_frontier.update(graph.predecessors(node))
            frontier = next_frontier - visited

        visited.update(frontier)

        entities = [
            self._entities[nid]
            for nid in visited
            if nid in self._entities
        ]
        relationships = [
            rel for rel in self._relationships.values()
            if rel.source_id in visited and rel.target_id in visited
        ]

        return GraphQueryResult(entities=entities, relationships=relationships)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships."""
        if entity_id not in self._entities:
            return False

        del self._entities[entity_id]
        graph = self._get_graph()
        if graph.has_node(entity_id):
            graph.remove_node(entity_id)

        # Remove related relationships
        to_remove = [
            rid for rid, rel in self._relationships.items()
            if rel.source_id == entity_id or rel.target_id == entity_id
        ]
        for rid in to_remove:
            del self._relationships[rid]

        self._save()
        return True

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship by ID."""
        rel = self._relationships.pop(relationship_id, None)
        if not rel:
            return False

        graph = self._get_graph()
        if graph.has_edge(rel.source_id, rel.target_id):
            graph.remove_edge(rel.source_id, rel.target_id)

        self._save()
        return True

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        graph = self._get_graph()
        entity_types: dict[str, int] = {}
        for e in self._entities.values():
            entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1

        relation_types: dict[str, int] = {}
        for r in self._relationships.values():
            relation_types[r.relation_type] = relation_types.get(r.relation_type, 0) + 1

        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "is_dag": graph.number_of_nodes() > 0 and __import__("networkx").is_directed_acyclic_graph(graph),
        }
