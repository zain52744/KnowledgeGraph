import logging
import time
from typing import List, Dict, Any, Optional
import networkx as nx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from vector_store.faiss_store import FAISSVectorStore
from graph.graph_builder import GraphBuilder
from extraction.extractor import KnowledgeExtractor

try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        return lambda f: f

logger = logging.getLogger(__name__)


@observe(name="hybrid_search")
def hybrid_search(query: str, faiss_store: FAISSVectorStore, graph: nx.DiGraph, llm: ChatOpenAI) -> Dict[str, Any]:
    
    try:
        # 1. Vector search for relevant text chunks
        vector_chunks = faiss_store.search_similar(query, n_results=5)
        logger.info(f"Found {len(vector_chunks)} similar chunks from vector search")

        # 2. Extract entity names from query
        entities_in_query = _extract_entities_from_query(query, llm)
        logger.info(f"Extracted {len(entities_in_query)} entities from query: {entities_in_query}")

        # 3. Graph context for found entities
        graph_context = []
        for entity_name in entities_in_query:
            if entity_name in graph:
                neighbors = _get_neighbors_from_graph(graph, entity_name, depth=1)
                graph_context.append({
                    "entity": entity_name,
                    "neighbors": neighbors,
                })
                logger.info(f"Found {len(neighbors)} neighbors for entity '{entity_name}'")
            else:
                logger.debug(f"Entity '{entity_name}' not found in graph")

        # 4. Combine results
        result = {
            "vector_chunks": vector_chunks,
            "graph_context": graph_context,
            "query": query,
        }

        return result

    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise


@observe(name="answer_question")
def answer_question(query: str, context: Dict[str, Any], llm: ChatOpenAI) -> str:
 
    try:
        # Build context string from vector chunks
        vector_context = _build_vector_context(context.get("vector_chunks", []))

        # Build context string from graph context
        graph_context_str = _build_graph_context(context.get("graph_context", []))

        # Combine all context
        full_context = f"{vector_context}\n{graph_context_str}" if vector_context and graph_context_str else (vector_context or graph_context_str)

        # Build prompt
        prompt = _build_answer_prompt(query, full_context)

        # Call LLM to generate answer
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided context. Be concise and grounded in the information provided."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        answer = response.content

        logger.info(f"Generated answer for query: {query[:50]}...")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise


def _extract_entities_from_query(query: str, llm: ChatOpenAI) -> List[str]:
   
    try:
        system_prompt = """Extract all named entities (person names, organization names, location names, concepts, etc.) from the query. 
Return only a JSON array of entity names, nothing else.
Example: ["Albert Einstein", "Nobel Prize", "Physics"]
If no entities found, return empty array: []"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}"),
        ]
        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON array
        import json
        try:
            entities = json.loads(response_text)
            return entities if isinstance(entities, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entities from LLM response: {response_text}")
            return []

    except Exception as e:
        logger.warning(f"Error extracting entities from query: {e}")
        return []


def _get_neighbors_from_graph(graph: nx.DiGraph, entity_name: str, depth: int = 1) -> List[Dict[str, Any]]:
   
    neighbors = []
    visited = {entity_name}
    current_level = {entity_name}

    for hop in range(1, depth + 1):
        next_level = set()
        for node in current_level:
            # Get successors and predecessors
            for successor in graph.successors(node):
                if successor not in visited:
                    next_level.add(successor)
                    visited.add(successor)
                    node_data = graph.nodes[successor]
                    neighbors.append({
                        "name": successor,
                        "type": node_data.get("type", "Unknown"),
                        "description": node_data.get("description", ""),
                        "distance": hop,
                    })

            for predecessor in graph.predecessors(node):
                if predecessor not in visited:
                    next_level.add(predecessor)
                    visited.add(predecessor)
                    node_data = graph.nodes[predecessor]
                    neighbors.append({
                        "name": predecessor,
                        "type": node_data.get("type", "Unknown"),
                        "description": node_data.get("description", ""),
                        "distance": hop,
                    })

        current_level = next_level
        if not current_level:
            break

    return neighbors


def _build_vector_context(vector_chunks: List[Dict[str, Any]]) -> str:
    if not vector_chunks:
        return ""

    parts = ["## Vector Search Results:"]
    for i, chunk in enumerate(vector_chunks, 1):
        text = chunk.get("text", "")[:2000]
        source = chunk.get("source", "Unknown")
        score = chunk.get("similarity_score", 0)
        parts.append(f"{i}. [{source} (score: {score:.2f})] {text}...")

    return "\n".join(parts)


def _build_graph_context(graph_context: List[Dict[str, Any]]) -> str:
    if not graph_context:
        return ""

    parts = ["## Graph Context:"]
    for ctx in graph_context:
        entity = ctx.get("entity", "Unknown")
        neighbors = ctx.get("neighbors", [])
        parts.append(f"\n### Entity: {entity}")
        if neighbors:
            parts.append("  Related Entities:")
            for neighbor in neighbors[:5]:  # Limit to 5 neighbors
                name = neighbor.get("name", "Unknown")
                distance = neighbor.get("distance", 0)
                parts.append(f"    - {name} (distance: {distance})")
        else:
            parts.append("  No related entities found")

    return "\n".join(parts)


def _build_answer_prompt(query: str, context: str) -> str:
    return f"""Based on the following context, answer the user's query concisely and accurately. 
If the context doesn't contain information to answer the query, say so clearly.

User Query: {query}

Context:
{context}

Answer:"""


class HybridRetriever:
   

    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        graph: GraphBuilder,
        llm_model: str = "gpt-4o-mini",
    ):
        
        self.faiss_store = faiss_store
        self.graph = graph
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        self.extractor = KnowledgeExtractor(model=llm_model)

    async def retrieve(
        self, query: str, vector_k: int = 5, graph_hops: int = 1
    ) -> Dict[str, Any]:
       
        try:
            # Vector search
            vector_results = self.faiss_store.similarity_search(query, k=vector_k)

            # Extract entities from query for graph search
            graph_results = await self._graph_search(query, hops=graph_hops)

            # Combine context
            context = self._build_context(vector_results, graph_results)

            # Generate answer
            answer = await self._generate_answer(query, context)

            return {
                "answer": answer,
                "vector_sources": vector_results,
                "graph_context": graph_results,
                "combined_context": context,
            }

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    async def _graph_search(self, query: str, hops: int = 1) -> List[Dict[str, Any]]:
     
        try:
            # Find entities in the query
            matching_entities = []
            for node in self.graph.graph.nodes():
                if any(word in node.lower() for word in query.lower().split()):
                    matching_entities.append(node)

            graph_context = []
            for entity in matching_entities:
                neighbors = self.graph.get_neighbors(entity, hops=hops)
                node_data = self.graph.get_node(entity)

                graph_context.append(
                    {
                        "entity": entity,
                        "type": node_data.get("type") if node_data else "Unknown",
                        "neighbors": neighbors,
                    }
                )

            return graph_context

        except Exception as e:
            logger.warning(f"Error during graph search: {e}")
            return []

    def _build_context(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
    ) -> str:
     
        context_parts = []

        if vector_results:
            context_parts.append("## Vector Search Results:")
            for i, result in enumerate(vector_results, 1):
                text = result.get("text", "")[:200]
                source = result.get("source", "Unknown")
                context_parts.append(f"{i}. [{source}] {text}...")

        if graph_results:
            context_parts.append("\n## Graph Context:")
            for entity_context in graph_results:
                entity = entity_context["entity"]
                entity_type = entity_context["type"]
                context_parts.append(f"\n### Entity: {entity} ({entity_type})")

                outgoing = entity_context["neighbors"].get("outgoing", [])
                if outgoing:
                    context_parts.append("  Outgoing:")
                    for target, relation in outgoing:
                        context_parts.append(f"    - {relation} -> {target}")

                incoming = entity_context["neighbors"].get("incoming", [])
                if incoming:
                    context_parts.append("  Incoming:")
                    for source, relation in incoming:
                        context_parts.append(f"    - {relation} <- {source}")

        return "\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
      
        try:
            prompt = f"""Based on the following context, answer the query concisely and accurately.

Query: {query}

Context:
{context}

Answer:"""

            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
