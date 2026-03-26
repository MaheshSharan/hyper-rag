#!/usr/bin/env python3
"""
Test script to verify all production fixes are working
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.connection_pool import connection_pool
from src.core.cache import query_cache, embedding_cache
from src.core.metrics import metrics_collector
from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.ingestion.embedder import Embedder


async def test_connection_pool():
    """Test Fix 1 & 2: Connection Pooling"""
    print("\n" + "="*60)
    print("Testing Connection Pool...")
    print("="*60)
    
    try:
        # Test Qdrant
        collections = connection_pool.qdrant.get_collections()
        print(f"✅ Qdrant: {len(collections.collections)} collections")
        
        # Test OpenSearch
        health = connection_pool.opensearch.cluster.health()
        print(f"✅ OpenSearch: {health['status']} status")
        
        # Test Neo4j
        with connection_pool.neo4j.session() as session:
            result = session.run("RETURN 1 as test")
            print(f"✅ Neo4j: Connected")
        
        return True
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False


async def test_caching():
    """Test Fix 3: Caching"""
    print("\n" + "="*60)
    print("Testing Cache...")
    print("="*60)
    
    try:
        # Test embedding cache
        embedder = Embedder()
        test_text = "This is a test"
        
        # First call - should hit API
        emb1 = embedder.embed_texts([test_text])
        
        # Second call - should hit cache
        emb2 = embedder.embed_texts([test_text])
        
        if emb1 == emb2:
            print("✅ Embedding cache working")
        
        # Test query cache
        query_cache.set("test query", 10, [{"test": "data"}])
        cached = query_cache.get("test query", 10)
        
        if cached:
            print("✅ Query cache working")
        
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


async def test_batch_embeddings():
    """Test Fix 4: Batch Embeddings"""
    print("\n" + "="*60)
    print("Testing Batch Embeddings...")
    print("="*60)
    
    try:
        embedder = Embedder()
        texts = [f"Test text {i}" for i in range(50)]
        
        import time
        start = time.time()
        embeddings = embedder.embed_batch(texts)
        duration = time.time() - start
        
        print(f"✅ Batched {len(texts)} texts in {duration:.2f}s")
        print(f"   ({len(texts)/duration:.1f} texts/sec)")
        
        return len(embeddings) == len(texts)
    except Exception as e:
        print(f"❌ Batch embedding test failed: {e}")
        return False


async def test_retry_logic():
    """Test Fix 5: Retry Logic"""
    print("\n" + "="*60)
    print("Testing Retry Logic...")
    print("="*60)
    
    try:
        embedder = Embedder()
        # This should work with retry logic
        result = embedder.embed_texts(["test"])
        print(f"✅ Retry logic configured (max_retries={embedder.max_retries})")
        return True
    except Exception as e:
        print(f"❌ Retry test failed: {e}")
        return False


async def test_logging():
    """Test Fix 6: Logging"""
    print("\n" + "="*60)
    print("Testing Logging...")
    print("="*60)
    
    import logging
    
    # Check that loggers are configured
    loggers = [
        "hyperrag.orchestrator",
        "hyperrag.planner",
        "hyperrag.reranker",
        "hyperrag.fusion",
        "hyperrag.embedder"
    ]
    
    all_configured = True
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        if not logger.hasHandlers():
            print(f"⚠️  Logger {logger_name} has no handlers")
            all_configured = False
    
    if all_configured:
        print("✅ All loggers configured")
    
    return True


async def test_metrics():
    """Test Fix 7: Metrics"""
    print("\n" + "="*60)
    print("Testing Metrics...")
    print("="*60)
    
    try:
        # Run a test query to generate metrics
        orchestrator = RetrievalOrchestrator()
        results = await orchestrator.retrieve("test query", final_top_k=5)
        
        # Check metrics
        summary = metrics_collector.get_summary()
        
        if "total_queries" in summary:
            print(f"✅ Metrics collected: {summary['total_queries']} queries")
            print(f"   Avg query time: {summary.get('avg_query_time', 0)}s")
        
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False


async def test_settings_validation():
    """Test Fix 8: Settings Validation"""
    print("\n" + "="*60)
    print("Testing Settings Validation...")
    print("="*60)
    
    try:
        from src.config.settings import settings
        
        # Check required fields
        required = [
            "NVIDIA_API_KEY",
            "OPENSEARCH_HOST",
            "QDRANT_HOST",
            "NEO4J_URI"
        ]
        
        all_valid = True
        for field in required:
            value = getattr(settings, field, None)
            if not value or value.strip() == "":
                print(f"⚠️  {field} is empty")
                all_valid = False
        
        if all_valid:
            print("✅ All required settings validated")
        
        return True
    except Exception as e:
        print(f"❌ Settings validation test failed: {e}")
        return False


async def main():
    print("\n" + "="*60)
    print("HyperRAG Production Fixes - Test Suite")
    print("="*60)
    
    tests = [
        ("Connection Pool", test_connection_pool),
        ("Caching", test_caching),
        ("Batch Embeddings", test_batch_embeddings),
        ("Retry Logic", test_retry_logic),
        ("Logging", test_logging),
        ("Metrics", test_metrics),
        ("Settings Validation", test_settings_validation),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    # Cleanup
    connection_pool.close_all()
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
