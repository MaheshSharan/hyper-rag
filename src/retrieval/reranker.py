import logging
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
from src.config.settings import settings
from typing import List, Dict, Any

logger = logging.getLogger("hyperrag.reranker")


class NvidiaReranker:
    def __init__(self):
        self.reranker = NVIDIARerank(
            model=settings.NVIDIA_RERANK_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            top_n=15,
            truncate="END"
        )

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Production-ready NVIDIA reranker with proper Document conversion"""
        if not candidates:
            return []

        # Filter out candidates with empty or whitespace-only text
        filtered_candidates = [c for c in candidates if c.get("text") and c["text"].strip()]
        if not filtered_candidates:
            logger.warning("All candidates have empty text. Skipping reranker.")
            return []

        # Convert dicts to LangChain Document objects
        docs = [Document(page_content=c["text"], metadata={"chunk_id": c["chunk_id"]}) for c in filtered_candidates]

        try:
            # Run reranker
            reranked_docs = self.reranker.compress_documents(query=query, documents=docs)

            # Map scores back to original candidates
            reranked = []
            for rd in reranked_docs:
                chunk_id = rd.metadata.get("chunk_id")
                original = next((c for c in candidates if c["chunk_id"] == chunk_id), None)
                if original:
                    enriched = original.copy()
                    enriched["rerank_score"] = float(rd.metadata.get("relevance_score", 0.0))
                    reranked.append(enriched)

            # Add any missing candidates with zero score
            processed_ids = {c["chunk_id"] for c in reranked}
            for c in candidates:
                if c["chunk_id"] not in processed_ids:
                    c_copy = c.copy()
                    c_copy["rerank_score"] = 0.0
                    reranked.append(c_copy)

            logger.info(f"NVIDIA Reranker applied successfully on {len(reranked)} candidates")
            return reranked

        except Exception as e:
            logger.warning(f"NVIDIA Reranker failed: {e}. Falling back to original order with dummy scores (0.5)")
            
            # Safe fallback
            reranked = []
            for c in candidates:
                c_copy = c.copy()
                c_copy["rerank_score"] = 0.5
                reranked.append(c_copy)
            
            return reranked