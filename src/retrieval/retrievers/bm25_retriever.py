import logging
from src.core.connection_pool import connection_pool
from typing import List, Dict, Any

logger = logging.getLogger("hyperrag.bm25_retriever")


class BM25Retriever:
    def __init__(self):
        self.index_name = "hyper_rag_bm25"

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "source"],
                        "type": "best_fields"
                    }
                },
                "size": top_k
            }
            response = connection_pool.opensearch.search(index=self.index_name, body=body, request_timeout=10)
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "chunk_id": source["chunk_id"],
                    "text": source["text"],
                    "source": source["source"],
                    "file_type": source.get("file_type"),
                    "bm25_score": hit["_score"],
                    "metadata": {"retriever": "bm25"}
                })
            
            logger.debug(f"BM25 retriever: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 error: {e}")
            return []