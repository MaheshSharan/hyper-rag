from src.retrieval.query_planner import QueryPlanner
from src.retrieval.reranker import NvidiaReranker
from src.retrieval.fusion import fuse_scores
from typing import List, Dict, Any

class RetrievalOrchestrator:
    def __init__(self):
        self.planner = QueryPlanner()
        self.reranker = NvidiaReranker()

    async def retrieve(self, query: str, final_top_k: int = 10) -> List[Dict[str, Any]]:
        """Full retrieval pipeline: Planner → Reranker → Fusion"""
        # Step 1: Get candidates from all retrievers
        candidates = await self.planner.retrieve_candidates(query, top_k_per_retriever=40)

        if not candidates:
            print("⚠️ No candidates found")
            return []

        # Step 2: Rerank with NVIDIA
        reranked = self.reranker.rerank(query, candidates)

        # Step 3: Fusion using your exact weights
        final_results = fuse_scores(reranked)

        print(f"✅ Retrieval completed. Returning top {final_top_k} results")
        return final_results[:final_top_k]