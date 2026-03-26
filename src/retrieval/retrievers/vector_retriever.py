import logging
from src.core.connection_pool import connection_pool
from src.ingestion.embedder import Embedder
from typing import List, Dict, Any

logger = logging.getLogger("hyperrag.vector_retriever")


class VectorRetriever:
    def __init__(self):
        self.collection_name = "hyper_rag_chunks"
        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Vector search with NVIDIA embeddings"""
        try:
            query_vector = self.embedder.embed_texts([query])[0]

            results = connection_pool.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            ).points

            retrieved = []
            for point in results:
                payload = point.payload or {}
                retrieved.append({
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "file_type": payload.get("file_type"),
                    "vector_score": float(point.score),
                    "metadata": {"retriever": "vector"}
                })
            
            logger.debug(f"Vector retriever: {len(retrieved)} results")
            return retrieved

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []