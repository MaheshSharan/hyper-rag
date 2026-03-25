from typing import List, Dict, Any

def fuse_scores(candidates: List[Dict[str, Any]], weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Ranks candidates based on hybrid weights.
    Default: 0.35 Rerank + 0.25 BM25 + 0.25 Vector + 0.15 Graph
    """
    if weights is None:
        weights = {
            "rerank_score": 0.35,
            "bm25_score": 0.25,
            "vector_score": 0.25,
            "graph_score": 0.15
        }

    for item in candidates:
        final_score = 0.0
        final_score += item.get("rerank_score", 0.0) * weights.get("rerank_score", 0.35)
        final_score += item.get("bm25_score", 0.0) * weights.get("bm25_score", 0.25)
        final_score += item.get("vector_score", 0.0) * weights.get("vector_score", 0.25)
        final_score += item.get("graph_score", 0.0) * weights.get("graph_score", 0.15)
        
        item["final_score"] = round(final_score, 4)

    # Sort candidates by the fusion score
    candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    
    if candidates:
        print(f"✅ Fusion completed - Top score: {candidates[0].get('final_score', 0)}")
        
    return candidates