from src.retrieval.query_planner import QueryPlanner
from src.retrieval.reranker import NvidiaReranker
from src.retrieval.fusion import fuse_scores
from typing import List, Dict, Any

class RetrievalOrchestrator:
    def __init__(self):
        self.planner = QueryPlanner()
        self.reranker = NvidiaReranker()

    async def retrieve(self, query: str, final_top_k: int = 10, weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        # 1. Retrieve raw candidates
        candidates = await self.planner.retrieve_candidates(query, top_k_per_retriever=40)

        if not candidates:
            print("⚠️ No candidates found")
            return []

        # 2. Apply Nvidia Reranker
        reranked = self.reranker.rerank(query, candidates)

        # 3. Hybrid Fusion with weights
        final_results = fuse_scores(reranked, weights)

        print(f"✅ Retrieval completed. Returning top {final_top_k} results")
        return final_results[:final_top_k]