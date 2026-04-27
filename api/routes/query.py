import logging
from typing import Any, Dict, List

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.auth import verify_api_key
from api.cache import cache
from api.dependencies import get_graph_store, get_mongo_store, get_pipeline
from api.limiter import limiter
from monitoring.langfuse_setup import langfuse_manager
from graph.graph_builder import GraphBuilder
from graph.mongo_store import MongoGraphStore
from orchestration.pipeline import KnowledgeGraphPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"], dependencies=[Depends(verify_api_key)])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    graph_context: List[Dict[str, Any]]


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: QueryRequest,
    pipeline: KnowledgeGraphPipeline = Depends(get_pipeline),
    mongo_store: MongoGraphStore = Depends(get_mongo_store),
) -> QueryResponse:

    try:
        cached = cache.get(body.question)
        if cached:
            logger.info(f"Cache hit for query: {body.question[:50]}...")
            return QueryResponse(**cached)

        result = pipeline.query(query=body.question)

        answer = result.get("answer", "")
        context = result.get("context", {})
        sources = [
            c.get("source", "")
            for c in context.get("vector_chunks", [])
            if c.get("source")
        ]
        graph_context = context.get("graph_context", [])

        # Collect source documents from graph context entities
        entity_names = [ctx.get("entity") for ctx in graph_context if ctx.get("entity")]
        graph_doc_sources = mongo_store.get_document_sources_for_entities(entity_names) if entity_names else []

        response_data = {"answer": answer, "sources": sources, "graph_context": graph_context}
        cache.set(body.question, response_data)

        mongo_store.save_query(body.question, answer, sources, graph_context, document_sources=graph_doc_sources)

        logger.info(f"Query processed and cached: {body.question[:50]}...")
        langfuse_manager.flush()
        return QueryResponse(**response_data)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")


@router.get("/graph/entity/{entity_name}", response_model=Dict[str, Any])
async def get_entity(
    entity_name: str,
    graph_store: GraphBuilder = Depends(get_graph_store),
) -> Dict[str, Any]:

    try:
        if entity_name not in graph_store.graph:
            raise HTTPException(
                status_code=404, detail=f"Entity '{entity_name}' not found"
            )

        entity_data = dict(graph_store.graph.nodes[entity_name])
        neighbors = list(graph_store.graph.successors(entity_name))

        logger.info(f"Retrieved entity: {entity_name}")
        return {
            "entity": entity_name,
            "data": entity_data,
            "neighbors": neighbors,
            "neighbor_count": len(neighbors),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving entity: {e}")
        raise HTTPException(status_code=500, detail="Entity retrieval failed")


@router.get("/graph/path", response_model=Dict[str, Any])
async def get_graph_path(
    source: str = Query(..., description="Source entity"),
    target: str = Query(..., description="Target entity"),
    graph_store: GraphBuilder = Depends(get_graph_store),
) -> Dict[str, Any]:
    try:
        if source not in graph_store.graph:
            raise HTTPException(
                status_code=404, detail=f"Source entity '{source}' not found"
            )
        if target not in graph_store.graph:
            raise HTTPException(
                status_code=404, detail=f"Target entity '{target}' not found"
            )

        try:
            path = nx.shortest_path(graph_store.graph, source, target)
            response: Dict[str, Any] = {
                "source": source,
                "target": target,
                "path": path,
                "path_length": len(path) - 1,
            }
        except nx.NetworkXNoPath:
            response = {
                "source": source,
                "target": target,
                "path": None,
                "path_length": -1,
                "message": "No path exists between entities",
            }

        logger.info(f"Retrieved path from {source} to {target}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving path: {e}")
        raise HTTPException(status_code=500, detail="Path retrieval failed")


@router.get("/query-history", response_model=List[Dict[str, Any]])
async def get_query_history(
    document: str = Query(..., description="Document filename to filter by"),
    limit: int = Query(100, ge=1, le=500),
    mongo_store: MongoGraphStore = Depends(get_mongo_store),
) -> List[Dict[str, Any]]:
    try:
        results = mongo_store.get_queries_by_document(document, limit=limit)
        logger.info(f"Fetched {len(results)} queries for document: {document}")
        return results
    except Exception as e:
        logger.error(f"Error fetching query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch query history")


