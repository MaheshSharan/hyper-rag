import os
import sys
import shutil
from pathlib import Path

# Ensure 'src' is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.indexing.qdrant_index import QdrantIndexer
from src.indexing.opensearch_index import OpenSearchIndexer
from src.indexing.neo4j_graph import Neo4jGraphBuilder

def wipe_all():
    print("\n" + "="*50)
    print("🧹 WIPING HYPER-RAG DATABASES & CACHE")
    print("="*50)
    
    # 1. Wipe Qdrant
    try:
        qdrant = QdrantIndexer()
        qdrant.client.delete_collection(qdrant.collection_name)
        print("✅ Qdrant: Collection deleted.")
    except Exception as e:
        print(f"⚠️  Qdrant skip: {e}")

    # 2. Wipe OpenSearch
    try:
        opensearch = OpenSearchIndexer()
        if opensearch.client.indices.exists(index=opensearch.index_name):
            opensearch.client.indices.delete(index=opensearch.index_name)
            print("✅ OpenSearch: Index deleted.")
    except Exception as e:
        print(f"⚠️  OpenSearch skip: {e}")

    # 3. Wipe Neo4j
    try:
        graph = Neo4jGraphBuilder()
        with graph.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Neo4j: Graph cleared.")
        graph.close()
    except Exception as e:
        print(f"⚠️  Neo4j skip: {e}")

    # 4. Wipe processed files (The new Project Folders)
    try:
        processed_dir = Path(project_root) / "data" / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache: data/processed/ cleared.")
        else:
            print("✅ Cache: data/processed/ is already empty.")
    except Exception as e:
        print(f"⚠️  Cache wipe skip: {e}")

    print("\n✨ SYSTEM CLEAN. All databases and caches are empty.")
    print("="*50 + "\n")

if __name__ == "__main__":
    wipe_all()
