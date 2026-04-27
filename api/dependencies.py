import logging

from config import settings
from graph.graph_builder import GraphBuilder
from graph.mongo_store import MongoGraphStore
from vector_store.faiss_store import FAISSVectorStore
from orchestration.pipeline import KnowledgeGraphPipeline

logger = logging.getLogger(__name__)

_graph_store: GraphBuilder = None
_vector_store: FAISSVectorStore = None
_pipeline: KnowledgeGraphPipeline = None
_mongo_store: MongoGraphStore = None


def init_shared_resources() -> None:

    global _graph_store, _vector_store, _pipeline, _mongo_store

    _graph_store = GraphBuilder()
    _vector_store = FAISSVectorStore(
        index_path=settings.faiss_index_path
    )

    try:
        _mongo_store = MongoGraphStore(
            mongodb_uri=settings.mongodb_uri
        )
        logger.info("MongoGraphStore initialized")
    except Exception as e:
        logger.warning(f"MongoGraphStore init failed (graph won't persist to MongoDB): {e}")
        _mongo_store = None

    _pipeline = KnowledgeGraphPipeline(
        graph_store=_graph_store,
        vector_store=_vector_store,
        llm_model=settings.llm_model,
        mongo_store=_mongo_store,
    )
    logger.info("Shared resources (graph, vector store, pipeline) initialized")


def get_pipeline() -> KnowledgeGraphPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized — call init_shared_resources() first")
    return _pipeline


def get_graph_store() -> GraphBuilder:
    if _graph_store is None:
        raise RuntimeError("Graph store not initialized — call init_shared_resources() first")
    return _graph_store


def get_vector_store() -> FAISSVectorStore:
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized — call init_shared_resources() first")
    return _vector_store


def get_mongo_store() -> MongoGraphStore:
    if _mongo_store is None:
        raise RuntimeError("Mongo store not initialized — call init_shared_resources() first")
    return _mongo_store
