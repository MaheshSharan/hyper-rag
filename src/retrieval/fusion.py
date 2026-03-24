from typing import List, Dict, Any

def fuse_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Final fusion using your exact weights from the original diagram:
    final_score = 0.35 * reranker + 0.25 * bm25 + 0.25 * vector + 0.15 * graph
    """
    for item in candidates:
        final_score = 0.0
        
        final_score += item.get("rerank_score", 0.0) * 0.35
        final_score += item.get("bm25_score", 0.0) * 0.25
        final_score += item.get("vector_score", 0.0) * 0.25
        final_score += item.get("graph_score", 0.0) * 0.15
        
        item["final_score"] = round(final_score, 4)

    # Sort by final_score descending
    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    print(f"✅ Fusion completed - Top score: {candidates[0]['final_score'] if candidates else 0}")
    return candidates