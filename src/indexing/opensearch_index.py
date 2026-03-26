import logging
from typing import List
from src.core.connection_pool import connection_pool
from src.core.schemas import DocumentChunk

logger = logging.getLogger("hyperrag.opensearch")


class OpenSearchIndexer:
    def __init__(self):
        self.index_name = "hyper_rag_bm25"

    def ensure_index(self):
        try:
            if not connection_pool.opensearch.indices.exists(index=self.index_name):
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
                connection_pool.opensearch.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created OpenSearch index '{self.index_name}'")
            else:
                logger.info(f"OpenSearch index ready: {self.index_name}")
        except Exception as e:
            logger.error(f"OpenSearch setup failed: {e}")

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
                    "language": chunk.metadata.language or ""
                }
                connection_pool.opensearch.index(
                    index=self.index_name,
                    id=chunk.chunk_id,
                    body=doc,
                    refresh=True
                )
            
            logger.info(f"Indexed {len(chunks)} documents to OpenSearch (BM25)")
        except Exception as e:
            logger.error(f"OpenSearch indexing failed: {e}")
