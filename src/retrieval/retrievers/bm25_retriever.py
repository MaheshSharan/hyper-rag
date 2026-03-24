from opensearchpy import OpenSearch
from src.config.settings import settings
from typing import List, Dict, Any
import time

class BM25Retriever:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=15
        )
        self.index_name = "hyper_rag_bm25"

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """BM25 keyword search - production ready"""
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
            response = self.client.search(index=self.index_name, body=body, request_timeout=10)
            
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
            return results

        except Exception as e:
            print(f"❌ BM25 error: {e}")
            return []