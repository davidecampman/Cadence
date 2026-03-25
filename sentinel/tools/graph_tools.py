"""Tools for the structured knowledge graph."""

from __future__ import annotations

from typing import Any

from sentinel.core.types import PermissionTier
from sentinel.knowledge.graph import KnowledgeGraph
from sentinel.tools.base import Tool


class GraphAddEntityTool(Tool):
    """Add an entity to the knowledge graph."""

    name = "graph_add_entity"
    description = (
        "Add an entity (node) to the knowledge graph. "
        "Entities represent things like functions, classes, modules, concepts, or people. "
        "If an entity with the same name and type exists, it updates its properties."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the entity (e.g., 'UserService', 'auth.py')",
            },
            "entity_type": {
                "type": "string",
                "description": "Type: 'function', 'class', 'module', 'file', 'concept', 'person', etc.",
            },
            "properties": {
                "type": "object",
                "description": "Optional key-value properties for the entity",
            },
        },
        "required": ["name", "entity_type"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    async def execute(
        self,
        name: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        entity = self._graph.add_entity(
            name=name,
            entity_type=entity_type,
            properties=properties,
        )
        return f"Entity added: {entity.name} ({entity.entity_type}) [id: {entity.id}]"


class GraphAddRelationTool(Tool):
    """Add a relationship between two entities in the knowledge graph."""

    name = "graph_add_relation"
    description = (
        "Add a directed relationship (edge) between two entities. "
        "Examples: 'calls', 'imports', 'depends_on', 'contains', 'implements', 'related_to'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "ID of the source entity",
            },
            "target_id": {
                "type": "string",
                "description": "ID of the target entity",
            },
            "relation_type": {
                "type": "string",
                "description": "Type of relationship (e.g., 'calls', 'imports', 'depends_on')",
            },
            "weight": {
                "type": "number",
                "description": "Relationship strength (default: 1.0)",
            },
        },
        "required": ["source_id", "target_id", "relation_type"],
    }
    permission_tier = PermissionTier.STANDARD

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    async def execute(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> str:
        rel = self._graph.add_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )
        if not rel:
            return "Error: One or both entity IDs not found."
        source = self._graph.get_entity(source_id)
        target = self._graph.get_entity(target_id)
        s_name = source.name if source else source_id
        t_name = target.name if target else target_id
        return f"Relationship added: {s_name} --[{relation_type}]--> {t_name}"


class GraphQueryTool(Tool):
    """Query the knowledge graph for entities and relationships."""

    name = "graph_query"
    description = (
        "Query the knowledge graph. Search for entities by name/type, "
        "get neighbors of an entity, or find paths between entities."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["find_entities", "get_neighbors", "find_path", "get_subgraph", "stats"],
                "description": "Query action to perform",
            },
            "name": {
                "type": "string",
                "description": "Entity name to search for (substring match)",
            },
            "entity_type": {
                "type": "string",
                "description": "Filter by entity type",
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID for neighbor/path/subgraph queries",
            },
            "target_id": {
                "type": "string",
                "description": "Target entity ID for path queries",
            },
            "relation_type": {
                "type": "string",
                "description": "Filter relationships by type",
            },
            "depth": {
                "type": "integer",
                "description": "Max depth for subgraph/path queries (default: 2)",
            },
        },
        "required": ["action"],
    }
    permission_tier = PermissionTier.READ_ONLY

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph

    async def execute(
        self,
        action: str,
        name: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        target_id: str | None = None,
        relation_type: str | None = None,
        depth: int = 2,
    ) -> str:
        if action == "find_entities":
            entities = self._graph.find_entities(name=name, entity_type=entity_type)
            if not entities:
                return "No entities found."
            lines = [f"  [{e.id}] {e.name} ({e.entity_type})" for e in entities]
            return f"Found {len(entities)} entities:\n" + "\n".join(lines)

        elif action == "get_neighbors":
            if not entity_id:
                return "Error: entity_id required for get_neighbors"
            result = self._graph.get_neighbors(
                entity_id, relation_type=relation_type
            )
            if not result.entities:
                return "No neighbors found."
            lines = []
            for e in result.entities:
                lines.append(f"  [{e.id}] {e.name} ({e.entity_type})")
            for r in result.relationships:
                lines.append(f"  --[{r.relation_type}]--> (weight: {r.weight})")
            return f"Neighbors ({len(result.entities)}):\n" + "\n".join(lines)

        elif action == "find_path":
            if not entity_id or not target_id:
                return "Error: entity_id and target_id required for find_path"
            path = self._graph.find_path(entity_id, target_id, max_depth=depth)
            if not path:
                return "No path found between the given entities."
            names = []
            for nid in path:
                e = self._graph.get_entity(nid)
                names.append(e.name if e else nid)
            return f"Path ({len(path)} nodes): " + " -> ".join(names)

        elif action == "get_subgraph":
            if not entity_id:
                return "Error: entity_id required for get_subgraph"
            result = self._graph.get_subgraph(entity_id, depth=depth)
            lines = [f"Subgraph: {len(result.entities)} entities, {len(result.relationships)} relationships"]
            for e in result.entities:
                lines.append(f"  [{e.id}] {e.name} ({e.entity_type})")
            for r in result.relationships:
                s = self._graph.get_entity(r.source_id)
                t = self._graph.get_entity(r.target_id)
                lines.append(
                    f"  {s.name if s else r.source_id} --[{r.relation_type}]--> "
                    f"{t.name if t else r.target_id}"
                )
            return "\n".join(lines)

        elif action == "stats":
            s = self._graph.stats()
            return (
                f"Knowledge Graph Stats:\n"
                f"  Entities: {s['total_entities']}\n"
                f"  Relationships: {s['total_relationships']}\n"
                f"  Entity types: {s['entity_types']}\n"
                f"  Relation types: {s['relation_types']}\n"
                f"  Is DAG: {s['is_dag']}"
            )

        return f"Unknown action: {action}"
