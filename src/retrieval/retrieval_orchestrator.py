import logging
import time
from src.retrieval.query_planner import QueryPlanner
from src.retrieval.reranker import NvidiaReranker
from src.retrieval.fusion import fuse_scores
from src.core.cache import query_cache
from src.core.metrics import metrics_collector, RetrievalMetrics
from typing import List, Dict, Any, Optional

logger = logging.getLogger("hyperrag.orchestrator")


class RetrievalOrchestrator:
    def __init__(self):
        self.planner = QueryPlanner()
        self.reranker = NvidiaReranker()

    async def retrieve(self, query: str, final_top_k: int = 10, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        start_time = time.time()
        metrics = RetrievalMetrics(query=query)
        
        # Check cache first
        cached = query_cache.get(query, final_top_k, weights)
        if cached:
            logger.info(f"Retrieved {len(cached)} results from cache")
            return cached
        
        # 1. Retrieve raw candidates
        candidates = await self.planner.retrieve_candidates(query, top_k_per_retriever=40)

        if not candidates:
            logger.warning("No candidates found")
            return []

        # 2. Apply Nvidia Reranker
        rerank_start = time.time()
        reranked = self.reranker.rerank(query, candidates)
        metrics.rerank_time = time.time() - rerank_start

        # 3. Hybrid Fusion with weights
        fusion_start = time.time()
        final_results = fuse_scores(reranked, weights)
        metrics.fusion_time = time.time() - fusion_start
        
        # Record metrics
        metrics.total_time = time.time() - start_time
        metrics.final_count = len(final_results[:final_top_k])
        metrics.top_score = final_results[0].get("final_score", 0) if final_results else 0
        
        # Score distribution
        if final_results:
            scores = [r.get("final_score", 0) for r in final_results[:final_top_k]]
            metrics.score_distribution = {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores)
            }
        
        metrics_collector.record_retrieval(metrics)
        
        # Cache results
        query_cache.set(query, final_top_k, final_results[:final_top_k], weights)

        logger.info(f"Retrieval completed. Returning top {final_top_k} results")
        return final_results[:final_top_k]