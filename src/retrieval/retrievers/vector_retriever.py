from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.config.settings import settings
from src.ingestion.embedder import Embedder
from typing import List, Dict, Any

class VectorRetriever:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_HOST, timeout=15)
        self.collection_name = "hyper_rag_chunks"
        self.embedder = Embedder()

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Vector search with NVIDIA embeddings - fixed for current Qdrant client"""
        try:
            query_vector = self.embedder.embed_texts([query])[0]

            results = self.client.query_points(
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
            return retrieved

        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []