import logging
import os

from graph.graph_builder import GraphBuilder
from vector_store.faiss_store import FAISSVectorStore
from orchestration.pipeline import KnowledgeGraphPipeline

logger = logging.getLogger(__name__)

_graph_store: GraphBuilder = None
_vector_store: FAISSVectorStore = None
_pipeline: KnowledgeGraphPipeline = None


def init_shared_resources() -> None:
    
    global _graph_store, _vector_store, _pipeline

    _graph_store = GraphBuilder()
    _vector_store = FAISSVectorStore(
        index_path=os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
    )
    _pipeline = KnowledgeGraphPipeline(
        graph_store=_graph_store,
        vector_store=_vector_store,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
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
