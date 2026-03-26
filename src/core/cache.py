"""
LRU Cache for query results and embeddings
Reduces redundant API calls and database queries
"""
import hashlib
import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger("hyperrag.cache")


class QueryCache:
    """LRU cache for query results"""
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, query: str, top_k: int, weights: Optional[Dict] = None) -> str:
        """Create cache key from query parameters"""
        key_data = {
            "query": query.lower().strip(),
            "top_k": top_k,
            "weights": weights or {}
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, top_k: int, weights: Optional[Dict] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached results"""
        key = self._make_key(query, top_k, weights)
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Cache HIT: {query[:50]}")
            return self._cache[key]
        logger.debug(f"Cache MISS: {query[:50]}")
        return None
    
    def set(self, query: str, top_k: int, results: List[Dict[str, Any]], weights: Optional[Dict] = None):
        """Cache results with LRU eviction"""
        key = self._make_key(query, top_k, weights)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = results
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached results"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Query cache cleared")


class EmbeddingCache:
    """LRU cache for text embeddings"""
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
    
    def _make_key(self, text: str) -> str:
        """Create cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._make_key(text)
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, text: str, embedding: List[float]):
        """Cache embedding with LRU eviction"""
        key = self._make_key(text)
        
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = embedding
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached embeddings"""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Embedding cache cleared")


# Global cache instances
query_cache = QueryCache(maxsize=128)
embedding_cache = EmbeddingCache(maxsize=512)
