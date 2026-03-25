import logging
from typing import List
from opensearchpy import OpenSearch, RequestsHttpConnection
from src.config.settings import settings
from src.core.schemas import DocumentChunk

logger = logging.getLogger("hyperrag.opensearch")

class OpenSearchIndexer:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
            connection_class=RequestsHttpConnection,
            verify_certs=False,
            timeout=30
        )
        self.index_name = "hyper_rag_bm25"

    def ensure_index(self):
        try:
            if not self.client.indices.exists(index=self.index_name):
                mapping = {
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        }
                    },
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "text": {"type": "text", "analyzer": "standard"},
                            "source": {"type": "keyword"},
                            "file_type": {"type": "keyword"},
                            "language": {"type": "keyword"}
                        }
                    }
                }
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"✅ Created OpenSearch index: {self.index_name}")
            else:
                logger.info(f"✅ OpenSearch index ready: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ OpenSearch setup failed: {e}")

    def index_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        
        try:
            for chunk in chunks:
                doc = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.metadata.source,
                    "file_type": chunk.metadata.file_type,
                    "language": chunk.metadata.language or "",
                }
                self.client.index(index=self.index_name, body=doc, id=chunk.chunk_id, refresh=True)
            
            logger.info(f"   → Indexed {len(chunks)} documents to OpenSearch (BM25)")
        except Exception as e:
            logger.error(f"❌ OpenSearch indexing failed: {e}")
