import sys
import os

# Ensure 'src' is importable regardless of working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


import asyncio
import sys
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline
from src.indexing.qdrant_index import QdrantIndexer
from src.indexing.opensearch_index import OpenSearchIndexer
from src.indexing.neo4j_graph import Neo4jGraphBuilder

async def main(folder_path: str):
    pipeline = IngestionPipeline()
    qdrant = QdrantIndexer()
    opensearch = OpenSearchIndexer()
    graph_builder = Neo4jGraphBuilder()

    # Ensure all indexes exist
    print("🔧 Ensuring indexes exist...")
    qdrant.ensure_collection()
    opensearch.ensure_index()
    graph_builder.ensure_constraints()

    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder {folder_path} does not exist!")
        return

    print(f"🚀 Starting full ingestion + indexing + graph building from: {folder}")
    processed = 0

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".py", ".js", ".ts", ".md", ".txt"]:
            print(f"Processing: {file_path.name}")
            try:
                result = await pipeline.ingest_file(str(file_path))
                
                if result.get("chunks"):
                    qdrant.index_chunks(result["chunks"])
                    opensearch.index_chunks(result["chunks"])
                    graph_builder.build_graph_from_chunks(result["chunks"], file_path.name)
                
                processed += 1
            except Exception as e:
                print(f"  ❌ Error on {file_path.name}: {e}")

    graph_builder.close()
    print(f"\n✅ Full Pipeline Completed!")
    print(f"   Processed {processed} files")
    print(f"   → Qdrant Vector + OpenSearch BM25 + Neo4j Graph")
    print(f"   PageIndex trees saved in data/processed/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_folder.py <folder_path>")
        sys.exit(1)

    asyncio.run(main(sys.argv[1]))
