from src.retrieval.retrievers.bm25_retriever import BM25Retriever
from src.retrieval.retrievers.vector_retriever import VectorRetriever
from src.retrieval.retrievers.graph_retriever import GraphRetriever
from src.retrieval.retrievers.pageindex_router import PageIndexRouter
from typing import List, Dict, Any
import asyncio

class QueryPlanner:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever()
        self.graph = GraphRetriever()
        self.pageindex = PageIndexRouter()

    async def retrieve_candidates(self, query: str, top_k_per_retriever: int = 50) -> List[Dict[str, Any]]:
        """Parallel multi-retrieval"""
        tasks = [
            asyncio.to_thread(self.bm25.retrieve, query, top_k_per_retriever),
            asyncio.to_thread(self.vector.retrieve, query, top_k_per_retriever),
            asyncio.to_thread(self.graph.retrieve, query, top_k_per_retriever),
            asyncio.to_thread(self.pageindex.retrieve, query, top_k_per_retriever)
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_candidates = []
        seen = set()

        for res in raw_results:
            if isinstance(res, Exception) or not isinstance(res, list):
                continue
            for item in res:
                cid = item.get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    all_candidates.append(item)

        print(f"🔍 Candidate pool built: {len(all_candidates)} unique chunks from all retrievers")
        return all_candidates