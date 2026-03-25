import sys
import os
import asyncio
import logging
from pathlib import Path
from tqdm import tqdm

# Ensure 'src' is importable regardless of working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("hyperrag.log", encoding='utf-8')]
)
logging.getLogger().handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]

from src.ingestion.pipeline import IngestionPipeline
from src.indexing.qdrant_index import QdrantIndexer
from src.indexing.opensearch_index import OpenSearchIndexer
from src.indexing.neo4j_graph import Neo4jGraphBuilder
from src.core.ignore_filter import IgnoreFilter

async def main(folder_path: str):
    pipeline = IngestionPipeline()
    qdrant = QdrantIndexer()
    opensearch = OpenSearchIndexer()
    graph_builder = Neo4jGraphBuilder()

    folder = Path(folder_path).resolve()
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist.")
        return

    project_name = folder.name
    print("\n" + "="*50)
    print(f"Starting HyperRAG Ingestion: [{project_name.upper()}]")
    print("="*50)

    # Load Universal Ignore Filter
    ignore_filter = IgnoreFilter(str(folder))

    print("Checking database connections...")
    try:
        qdrant.ensure_collection()
        opensearch.ensure_index()
        graph_builder.ensure_constraints()
    except Exception as e:
        print(f"Database warning: {e}")

    # Universal extensions support (python, js, c, go, rust, java, documentation, config)
    search_exts = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".pdf", ".json", 
        ".yaml", ".yml", ".c", ".cpp", ".cc", ".h", ".hpp", ".go", ".rust", 
        ".java", ".kt", ".rs", ".css", ".scss", ".html", ".sh", ".bash"
    }

    # Find files using the new universal ignore filter
    files = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in search_exts:
            if not ignore_filter.is_ignored(str(f)):
                files.append(f)
    
    total = len(files)
    print(f"Found {total} valid files. Using .hyperragignore filters if present.")
    print(f"Saving artifacts to: data/processed/{project_name}/")

    with tqdm(total=total, desc="Indexing", unit="file", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        success_count = 0
        error_count = 0
        total_chunks = 0
        
        for file_path in files:
            pbar.set_description(f"Processing: {file_path.name[:20]:20}")
            
            try:
                result = await pipeline.ingest_file(str(file_path), project_name=project_name)
                chunks = result.get("chunks") or []
                
                if chunks:
                    total_chunks += len(chunks)
                    qdrant.index_chunks(chunks)
                    opensearch.index_chunks(chunks)
                    graph_builder.build_graph_from_chunks(chunks, file_path.name)
                
                success_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=success_count, chunks=total_chunks)
                
            except Exception as e:
                error_count += 1
                pbar.write(f" Error in {file_path.name}: {e}")
                pbar.update(1)

    graph_builder.close()
    
    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print(f"Processed: {success_count} success | {error_count} failed")
    print(f"Total Chunks: {total_chunks}")
    print(f"Full logs saved to: hyperrag.log")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_folder.py <folder_path>")
        sys.exit(1)

    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        print("\nStopped.")
