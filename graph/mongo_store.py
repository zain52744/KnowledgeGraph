import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import networkx as nx
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MongoGraphStore:
   

    def __init__(self, mongodb_uri: Optional[str] = None, db_name: str = "knowledge_graph"):
        
        uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.snapshots_collection = self.db["graph_snapshots"]
        self.entities_collection = self.db["entities"]
        self.relations_collection = self.db["relations"]
        self.query_history_collection = self.db["query_history"]
        self._create_indexes()

    def _create_indexes(self) -> None:

        try:
            self.snapshots_collection.create_index("timestamp", unique=False)
            self.entities_collection.create_index("entity_id", unique=True)
            self.relations_collection.create_index([("source", 1), ("target", 1)])
            self.relations_collection.create_index("source_document", unique=False)
            self.query_history_collection.create_index("timestamp", unique=False)
            self.query_history_collection.create_index("document_sources", unique=False)
            logger.info("MongoDB indexes created")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")

    def save_graph_snapshot(self, graph: nx.DiGraph, collection_name: str = "graph_snapshots") -> str:
       
        try:
            snapshot = {
                "timestamp": datetime.utcnow(),
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "nodes": [
                    {"name": node, **attrs}
                    for node, attrs in graph.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, **attrs}
                    for u, v, attrs in graph.edges(data=True)
                ],
            }
            result = self.snapshots_collection.insert_one(snapshot)
            logger.info(f"Graph snapshot saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving graph snapshot: {e}")
            raise

    def load_latest_graph(self) -> nx.DiGraph:
       
        try:
            snapshot = self.snapshots_collection.find_one(
                sort=[("timestamp", -1)]
            )
            if not snapshot:
                logger.warning("No graph snapshots found")
                return nx.DiGraph()

            graph = nx.DiGraph()
            
            # Restore nodes
            for node_data in snapshot.get("nodes", []):
                name = node_data.pop("name")
                graph.add_node(name, **node_data)
            
            # Restore edges
            for edge_data in snapshot.get("edges", []):
                source = edge_data.pop("source")
                target = edge_data.pop("target")
                graph.add_edge(source, target, **edge_data)
            
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return nx.DiGraph()

    def save_entities(self, entities: List[Dict[str, Any]]) -> int:
        
        try:
            count = 0
            for entity in entities:
                self.entities_collection.update_one(
                    {"entity_id": entity.get("id")},
                    {"$set": entity},
                    upsert=True
                )
                count += 1
            logger.info(f"Saved {count} entities")
            return count
        except Exception as e:
            logger.error(f"Error saving entities: {e}")
            raise

    def save_relations(self, relations: List[Dict[str, Any]], source_document: str = None) -> int:
        try:
            count = 0
            for relation in relations:
                doc = {**relation}
                if source_document:
                    doc["source_document"] = source_document
                self.relations_collection.update_one(
                    {"source": relation.get("source"), "target": relation.get("target")},
                    {"$set": doc},
                    upsert=True
                )
                count += 1
            logger.info(f"Saved {count} relations")
            return count
        except Exception as e:
            logger.error(f"Error saving relations: {e}")
            raise

    def get_document_sources_for_entities(self, entity_names: List[str]) -> List[str]:
        try:
            docs = self.relations_collection.find(
                {"$or": [
                    {"source": {"$in": entity_names}},
                    {"target": {"$in": entity_names}},
                ]},
                {"source_document": 1, "_id": 0}
            )
            sources = {d["source_document"] for d in docs if d.get("source_document")}
            return list(sources)
        except Exception as e:
            logger.warning(f"Error fetching document sources for entities: {e}")
            return []

    def get_queries_by_document(self, document_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            cursor = self.query_history_collection.find(
                {"document_sources": document_name},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching queries by document: {e}")
            return []

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
       
        return self.entities_collection.find_one({"entity_id": entity_id})

    def get_relation(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        
        return self.relations_collection.find_one({"source": source, "target": target})

    def close(self) -> None:
        
        self.client.close()
        logger.info("MongoDB connection closed")

    def get_all_entities(self, limit: int = 1000) -> List[Dict[str, Any]]:
       
        return list(self.entities_collection.find().limit(limit))

    def get_all_relations(self, limit: int = 10000) -> List[Dict[str, Any]]:
       
        return list(self.relations_collection.find().limit(limit))

    def save_query(self, question: str, answer: str, sources: List[str], graph_context: List[Dict[str, Any]], document_sources: List[str] = None) -> str:
        try:
            doc = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "graph_context": graph_context,
                "document_sources": list(set(sources + (document_sources or []))),
                "timestamp": datetime.utcnow(),
            }
            result = self.query_history_collection.insert_one(doc)
            logger.info(f"Query saved to history: {question[:60]}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error saving query to history: {e}")
            raise

    def clear(self) -> None:
        self.entities_collection.delete_many({})
        self.relations_collection.delete_many({})
        self.snapshots_collection.delete_many({})
        self.query_history_collection.delete_many({})
        logger.info("MongoDB collections cleared")
