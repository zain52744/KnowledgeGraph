import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from ingestion.pdf_parser import PDFParser
from ingestion.csv_loader import CSVLoader
from extraction.extractor import KnowledgeExtractor
from graph.graph_builder import GraphBuilder
from graph.mongo_store import MongoGraphStore
from vector_store.faiss_store import FAISSVectorStore
from rag.retriever import hybrid_search, answer_question
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class KnowledgeGraphPipeline:
    

    def __init__(
        self,
        graph_store: GraphBuilder,
        vector_store: FAISSVectorStore,
        llm_model: str = "gpt-4o-mini",
        mongo_store: MongoGraphStore = None,
    ):

        self.graph_store = graph_store
        self.vector_store = vector_store
        self.mongo_store = mongo_store
        self.extractor = KnowledgeExtractor(model=llm_model)
        self.pdf_parser = PDFParser()
        self.csv_loader = CSVLoader()
        self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
        
        # Compiled pipelines
        self.ingestion_pipeline = None
        self.query_pipeline = None
        
        # Build pipelines on init
        self._build_ingestion_pipeline()
        self._build_query_pipeline()

    def _parse_pdf_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            return state

        logger.info(f"Parsing PDF: {pdf_path}")
        chunks = self.pdf_parser.extract_and_chunk(pdf_path)
        state["chunks"] = chunks
        state["source"] = pdf_path
        return state

    def _parse_csv_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        csv_path = state.get("csv_path")
        if not csv_path:
            return state

        logger.info(f"Loading CSV: {csv_path}")
        csv_data = self.csv_loader.load_csv(csv_path)
        state["triples"] = csv_data["triples"]
        state["source"] = csv_data["source"]
        return state

    def _extract_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        # extract_batch expects plain strings; chunks are dicts from extract_and_chunk
        texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        logger.info(f"Extracting entities/relations from {len(texts)} chunks")
        knowledge_graphs = self.extractor.extract_batch(texts)
        state["knowledge_graphs"] = knowledge_graphs
        return state

    def _store_graph_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Store in NetworkX graph node."""
        knowledge_graphs = state.get("knowledge_graphs", [])
        triples = state.get("triples", [])

        if knowledge_graphs:
            logger.info(f"Storing {len(knowledge_graphs)} KGs in graph")
            for kg in knowledge_graphs:
                self.graph_store.add_entities(kg.entities)
                self.graph_store.add_relations(kg.relations)

                if self.mongo_store:
                    try:
                        source_doc = state.get("source", "unknown")
                        entities_data = [e.model_dump() for e in kg.entities]
                        relations_data = [r.model_dump() for r in kg.relations]
                        self.mongo_store.save_entities(entities_data)
                        self.mongo_store.save_relations(relations_data, source_document=source_doc)
                    except Exception as e:
                        logger.warning(f"MongoDB persist failed (non-fatal): {e}")

        elif triples:
            logger.info(f"Storing {len(triples)} triples in graph")
            from extraction.schemas import Relation

            entities = set()
            relations = []

            for triple in triples:
                if isinstance(triple, dict):
                    source = triple.get("source")
                    relation_type = triple.get("relation")
                    target = triple.get("target")
                else:
                    source, relation_type, target = triple
                    
                entities.add(source)
                entities.add(target)
                relations.append(
                    Relation(
                        source=source,
                        relation=relation_type,
                        target=target,
                        confidence=0.8,
                    )
                )

            for entity_name in entities:
                self.graph_store.graph.add_node(entity_name, type="Entity", description="")

            self.graph_store.add_relations(relations)

            if self.mongo_store:
                try:
                    source_doc = state.get("source", "unknown")
                    entities_data = [{"entity_id": e, "name": e, "type": "Entity"} for e in entities]
                    relations_data = [r.model_dump() for r in relations]
                    self.mongo_store.save_entities(entities_data)
                    self.mongo_store.save_relations(relations_data, source_document=source_doc)
                except Exception as e:
                    logger.warning(f"MongoDB persist failed (non-fatal): {e}")

        state["graph_stats"] = self.graph_store.get_graph_stats()
        return state

    def _index_vectors_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        chunks = state.get("chunks", [])
        source = state.get("source", "unknown")

        if chunks:
            logger.info(f"Indexing {len(chunks)} chunks in FAISS")
            self.vector_store.add_chunks(chunks)

        state["vector_stats"] = self.vector_store.get_stats()
        return state

    def _retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        query = state.get("query")
        if not query:
            logger.warning("No query provided for retrieval")
            return state

        try:
            logger.info(f"Retrieving context for query: {query}")
            context = hybrid_search(
                query=query,
                faiss_store=self.vector_store,
                graph=self.graph_store.graph,
                llm=self.llm
            )
            state["context"] = context
            state["retrieve_status"] = "success"
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            state["retrieve_status"] = "failed"
            state["error"] = str(e)

        return state

    def _answer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        context = state.get("context")
        query = state.get("query")
        
        if not context or not query:
            logger.warning("Missing context or query for answer generation")
            state["answer"] = "Unable to generate answer - missing context or query"
            return state

        try:
            logger.info(f"Generating answer for query: {query}")
            answer = answer_question(
                query=query,
                context=context,
                llm=self.llm
            )
            state["answer"] = answer
            state["answer_status"] = "success"
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["answer"] = "Error generating answer"
            state["answer_status"] = "failed"
            state["error"] = str(e)

        return state

    def _build_ingestion_pipeline(self) -> None:
        
        workflow = StateGraph(dict)

        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("extract", self._extract_node)
        workflow.add_node("store_graph", self._store_graph_node)
        workflow.add_node("store_vectors", self._index_vectors_node)

        workflow.add_edge(START, "ingest")
        workflow.add_edge("ingest", "extract")
        workflow.add_edge("extract", "store_graph")
        workflow.add_edge("store_graph", "store_vectors")
        workflow.add_edge("store_vectors", END)

        self.ingestion_pipeline = workflow.compile()
        logger.info("Ingestion pipeline compiled")

    def _build_query_pipeline(self) -> None:
        
        workflow = StateGraph(dict)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("answer", self._answer_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        self.query_pipeline = workflow.compile()
        logger.info("Query pipeline compiled")

    def _ingest_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        pdf_path = state.get("pdf_path")
        csv_path = state.get("csv_path")

        if pdf_path:
            logger.info(f"Ingesting PDF: {pdf_path}")
            try:
                chunks = self.pdf_parser.extract_and_chunk(pdf_path)
                state["chunks"] = chunks
                state["source"] = pdf_path
                state["ingest_type"] = "pdf"
            except Exception as e:
                logger.error(f"Error parsing PDF: {e}")
                state["chunks"] = []
                state["error"] = str(e)
        elif csv_path:
            logger.info(f"Ingesting CSV: {csv_path}")
            try:
                csv_data = self.csv_loader.load_csv(csv_path)
                state["triples"] = csv_data
                state["source"] = csv_path
                state["ingest_type"] = "csv"
            except Exception as e:
                logger.error(f"Error loading CSV: {e}")
                state["triples"] = []
                state["error"] = str(e)
        else:
            logger.warning("No PDF or CSV path provided")
            state["error"] = "No file path provided"

        return state

    def ingest_text(self, text: str, source: str = "report") -> Dict[str, Any]:
        """Ingest plain text directly into the vector store, bypassing PDF parsing."""
        if not text.strip():
            return {"chunks": [], "vector_stats": self.vector_store.get_stats()}

        chunks = self.pdf_parser._create_token_chunks(text, [], source)
        if chunks:
            self.vector_store.add_chunks(chunks)

        return {
            "chunks": chunks,
            "vector_stats": self.vector_store.get_stats(),
        }

    def ingest(self, pdf_path: str = None, csv_path: str = None) -> Dict[str, Any]:
        
        if not self.ingestion_pipeline:
            raise RuntimeError("Ingestion pipeline not initialized")

        initial_state = {}
        if pdf_path:
            initial_state["pdf_path"] = pdf_path
        elif csv_path:
            initial_state["csv_path"] = csv_path
        else:
            raise ValueError("Either pdf_path or csv_path must be provided")

        logger.info(f"Running ingestion pipeline with state: {initial_state}")
        return self.ingestion_pipeline.invoke(initial_state)

    def query(self, query: str) -> Dict[str, Any]:
        
        if not self.query_pipeline:
            raise RuntimeError("Query pipeline not initialized")

        initial_state = {"query": query}
        logger.info(f"Running query pipeline with query: {query}")
        return self.query_pipeline.invoke(initial_state)
