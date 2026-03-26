"""
Performance metrics and observability
Tracks retrieval timing, score distributions, and component performance
"""
import time
import logging
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("hyperrag.metrics")


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation"""
    query: str
    total_time: float = 0.0
    retriever_times: Dict[str, float] = field(default_factory=dict)
    candidate_counts: Dict[str, int] = field(default_factory=dict)
    rerank_time: float = 0.0
    fusion_time: float = 0.0
    final_count: int = 0
    top_score: float = 0.0
    score_distribution: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Singleton metrics collector"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.retrieval_history: List[RetrievalMetrics] = []
        self.retriever_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "total_results": 0})
        self._initialized = True
    
    def record_retrieval(self, metrics: RetrievalMetrics):
        """Record a completed retrieval operation"""
        self.retrieval_history.append(metrics)
        
        # Update per-retriever stats
        for retriever, duration in metrics.retriever_times.items():
            self.retriever_stats[retriever]["count"] += 1
            self.retriever_stats[retriever]["total_time"] += duration
            self.retriever_stats[retriever]["total_results"] += metrics.candidate_counts.get(retriever, 0)
        
        # Log summary
        logger.info(
            f"Query completed in {metrics.total_time:.3f}s | "
            f"Candidates: {sum(metrics.candidate_counts.values())} | "
            f"Final: {metrics.final_count} | "
            f"Top score: {metrics.top_score:.4f}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary"""
        if not self.retrieval_history:
            return {"message": "No retrievals recorded yet"}
        
        total_queries = len(self.retrieval_history)
        avg_time = sum(m.total_time for m in self.retrieval_history) / total_queries
        
        retriever_summary = {}
        for name, stats in self.retriever_stats.items():
            retriever_summary[name] = {
                "avg_time": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                "avg_results": stats["total_results"] / stats["count"] if stats["count"] > 0 else 0,
                "total_calls": stats["count"]
            }
        
        return {
            "total_queries": total_queries,
            "avg_query_time": round(avg_time, 3),
            "retrievers": retriever_summary
        }
    
    def clear(self):
        """Clear all metrics"""
        self.retrieval_history.clear()
        self.retriever_stats.clear()
        logger.info("Metrics cleared")


# Global metrics instance
metrics_collector = MetricsCollector()
