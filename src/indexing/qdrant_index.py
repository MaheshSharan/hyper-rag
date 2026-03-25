import logging
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config.settings import settings
from src.core.schemas import DocumentChunk

logger = logging.getLogger("hyperrag.qdrant")

class QdrantIndexer:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=60
        )
        self.collection_name = "hyper_rag_chunks"
        self.vector_size = 2048 

    def ensure_collection(self):
        """Create or recreate collection with correct dimension"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if exists:
                self.client.delete_collection(self.collection_name)
                logger.info(f"🗑️  Deleted old Qdrant collection")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created Qdrant collection '{self.collection_name}' with dimension {self.vector_size}")
        except Exception as e:
            logger.error(f"❌ Qdrant setup failed: {e}")

    def index_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return

        points = []
        for chunk in chunks:
            if chunk.embedding is None or len(chunk.embedding) != self.vector_size:
                continue

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:50000], 
                    "source": chunk.metadata.source,
                    "file_type": chunk.metadata.file_type,
                    "language": chunk.metadata.language,
                    "section_path": chunk.metadata.section_path,
                    "hierarchy_level": chunk.metadata.hierarchy_level,
                }
            ))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"   → Upserted {len(points)} vectors to Qdrant (dim={self.vector_size})")
