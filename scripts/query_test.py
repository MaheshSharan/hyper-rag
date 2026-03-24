import asyncio
from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator

async def test():
    orchestrator = RetrievalOrchestrator()
    
    query = "What does the sample.py file contain?"   # Change this to test different queries
    
    print(f"\n🔎 Query: {query}")
    print("=" * 60)
    
    results = await orchestrator.retrieve(query, final_top_k=8)
    
    print(f"\n=== FINAL TOP {len(results)} RESULTS (after rerank + fusion) ===")
    for i, r in enumerate(results, 1):
        score_str = f"final={r.get('final_score', 0):.3f}"
        if "rerank_score" in r:
            score_str += f" | rerank={r['rerank_score']:.2f}"
        print(f"{i:2d}. [{score_str}] {r['source']} → {r['text'][:120].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    asyncio.run(test())