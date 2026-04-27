import json
import logging
from typing import List, Dict, Any, Optional
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import settings

logger = logging.getLogger(__name__)


class FAISSVectorStore:


    def __init__(self, dimension: int = 1536, index_path: Optional[str] = None):

        self.dimension = dimension
        self.index = None
        self.index_path = index_path or "faiss_index.bin"
        self.metadata_path = self.index_path.replace(".bin", "_metadata.json")
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        self.chunks_metadata: List[Dict[str, Any]] = []

        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._create_index()

    def _create_index(self) -> None:
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created new FAISS index with dimension {self.dimension}")

    def _load_index(self) -> None:
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from {self.index_path}")
        except Exception as e:
            logger.warning(f"Failed to load index, creating new: {e}")
            self._create_index()
            return

        # Load matching metadata sidecar
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.chunks_metadata = json.load(f)
                logger.info(f"Loaded {len(self.chunks_metadata)} chunk metadata entries")
            except Exception as e:
                logger.warning(f"Failed to load metadata, starting fresh: {e}")
                self.chunks_metadata = []
        else:
            # Metadata out of sync with FAISS — reset both to stay consistent
            logger.warning("No metadata sidecar found; resetting FAISS index to maintain sync")
            self._create_index()
            self.chunks_metadata = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        
        try:
            texts = [chunk.get("text", "") for chunk in chunks]
            
            # Get embeddings from OpenAI
            embeddings = self.embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype(np.float32)

            # Add to FAISS index
            self.index.add(embeddings_array)

            # Store metadata
            for i, chunk in enumerate(chunks):
                self.chunks_metadata.append({
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", "unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", f"chunk_{len(self.chunks_metadata) + i}"),
                })

            logger.info(f"Added {len(chunks)} chunks to vector store")
            self._save_index()
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise

    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        
        try:
            if self.index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []

            # Get query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            query_array = np.array([query_embedding]).astype(np.float32)

            # Search FAISS index
            distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))

            # Build results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks_metadata):
                    chunk = self.chunks_metadata[idx]
                    results.append({
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_id": chunk["chunk_id"],
                        "similarity_score": float(1 / (1 + distance)),  # Convert L2 distance to similarity
                    })

            logger.info(f"Found {len(results)} similar chunks for query")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.search_similar(query, n_results=k)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_path": self.index_path,
        }

    def _save_index(self) -> None:
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, ensure_ascii=False)
            logger.info(f"Saved {len(self.chunks_metadata)} metadata entries to {self.metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def close(self) -> None:
        self._save_index()
        logger.info("Vector store closed")
