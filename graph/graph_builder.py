import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.schemas import Entity, Relation
from api.sanitization import sanitize_entity_name, sanitize_relation_type, sanitize_description

logger = logging.getLogger(__name__)


def add_entities(graph: nx.DiGraph, entities: List[Entity]) -> None:
   
    added_count = 0
    for entity in entities:
        try:
            # Sanitize entity data
            name = sanitize_entity_name(entity.name, strict=False)
            etype = sanitize_entity_name(entity.type, strict=False)
            description = sanitize_description(entity.description)
            
            if not name or not etype:
                logger.warning(
                    f"Skipping entity {entity.id}: name or type became empty after sanitization"
                )
                continue
            
            graph.add_node(
                name,
                id=entity.id,
                type=etype,
                description=description if description else "",
            )
            added_count += 1
        except Exception as e:
            logger.error(f"Error adding entity {entity.id}: {e}")
            continue
    
    logger.info(f"Added {added_count} entities to graph")


def add_relations(graph: nx.DiGraph, relations: List[Relation]) -> None:
   
    added_count = 0
    for relation in relations:
        try:
            # Sanitize relation data
            source = sanitize_entity_name(relation.source, strict=False)
            target = sanitize_entity_name(relation.target, strict=False)
            rel_type = sanitize_relation_type(relation.relation, strict=False)
            
            if not source or not target or not rel_type:
                logger.warning(
                    f"Skipping relation {relation.source} -> {relation.target}: "
                    "source, target, or relation became empty after sanitization"
                )
                continue
            
            # Verify nodes exist before adding edge
            if source not in graph:
                logger.warning(f"Source entity '{source}' not in graph, skipping relation")
                continue
            if target not in graph:
                logger.warning(f"Target entity '{target}' not in graph, skipping relation")
                continue
            
            graph.add_edge(
                source,
                target,
                relation=rel_type,
                confidence=relation.confidence,
            )
            added_count += 1
        except Exception as e:
            logger.error(f"Error adding relation {relation.source} -> {relation.target}: {e}")
            continue
    
    logger.info(f"Added {added_count} relations to graph")


def get_neighbors(
    graph: nx.DiGraph, entity_name: str, depth: int = 1
) -> List[Dict[str, Any]]:
    
    if entity_name not in graph:
        logger.warning(f"Entity '{entity_name}' not found in graph")
        return []

    neighbors = []
    visited = {entity_name}

    # BFS to find all nodes within depth hops
    current_level = {entity_name}

    for hop in range(1, depth + 1):
        next_level = set()

        for node in current_level:
            # Get successors (outgoing edges)
            for successor in graph.successors(node):
                if successor not in visited:
                    next_level.add(successor)
                    visited.add(successor)

                    # Get node attributes
                    node_data = graph.nodes[successor]
                    neighbors.append(
                        {
                            "name": successor,
                            "type": node_data.get("type", "Unknown"),
                            "description": node_data.get("description", ""),
                            "distance": hop,
                        }
                    )

            # Get predecessors (incoming edges)
            for predecessor in graph.predecessors(node):
                if predecessor not in visited:
                    next_level.add(predecessor)
                    visited.add(predecessor)

                    # Get node attributes
                    node_data = graph.nodes[predecessor]
                    neighbors.append(
                        {
                            "name": predecessor,
                            "type": node_data.get("type", "Unknown"),
                            "description": node_data.get("description", ""),
                            "distance": hop,
                        }
                    )

        current_level = next_level
        if not current_level:
            break

    return neighbors


def find_path(graph: nx.DiGraph, source: str, target: str) -> List[str]:
   
    if source not in graph or target not in graph:
        logger.warning(f"Source or target not found in graph")
        return []

    try:
        path = nx.shortest_path(graph, source, target)
        logger.info(f"Found path from {source} to {target}: {' -> '.join(path)}")
        return path
    except nx.NetworkXNoPath:
        logger.info(f"No path found between {source} and {target}")
        return []
    except Exception as e:
        logger.error(f"Error finding path: {e}")
        return []


class GraphBuilder:
    

    def __init__(self):
       
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_entities(self, entities: List[Entity]) -> None:
       
        add_entities(self.graph, entities)

    def add_relations(self, relations: List[Relation]) -> None:
        
        add_relations(self.graph, relations)

    def get_neighbors(self, entity_name: str, depth: int = 1) -> List[Dict[str, Any]]:
       
        return get_neighbors(self.graph, entity_name, depth)

    def find_path(self, source: str, target: str) -> List[str]:
        
        return find_path(self.graph, source, target)

    def get_node(self, node_name: str) -> Optional[Dict[str, Any]]:
        
        if node_name in self.graph:
            return dict(self.graph.nodes[node_name])
        return None

    def get_graph_stats(self) -> Dict[str, Any]:
        
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_strongly_connected": nx.is_strongly_connected(self.graph),
            "num_weakly_connected_components": nx.number_weakly_connected_components(
                self.graph
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
       
        nodes = [
            {"name": node, **attrs}
            for node, attrs in self.graph.nodes(data=True)
        ]
        edges = [
            {
                "source": u,
                "target": v,
                **attrs
            }
            for u, v, attrs in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}


