"""
HyperRAG MCP Server
===================
Production-grade MCP server exposing HyperRAG capabilities to Claude Desktop.

Tools exposed:
  - query_codebase      : Full hybrid RAG query (BM25 + Vector + Graph + PageIndex)
  - ingest_folder       : Ingest a project folder into all three indexes
  - get_retrieval_chunks: Raw ranked chunks without LLM generation
  - check_health        : Verify all backend services are running
  - list_indexed_projects: List all projects currently indexed in HyperRAG
  - wipe_project_index  : Remove all data for a specific project from all indexes
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Any

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Allow the MCP server to import from the parent hyper-rag src/ tree
_HERE = Path(__file__).resolve().parent          # mcp_server/hyper_rag_mcp/
_PROJECT_ROOT = _HERE.parent.parent              # hyper-rag/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Change working directory to project root so relative paths (data/, hyperrag.log) resolve correctly
os.chdir(_PROJECT_ROOT)

# ── MCP SDK ───────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── HyperRAG internals ────────────────────────────────────────────────────────
from src.config.settings import settings
from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.retrieval.context_builder import ContextBuilder
from src.generation.generator import LLMGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.indexing.qdrant_index import QdrantIndexer
from src.indexing.opensearch_index import OpenSearchIndexer
from src.indexing.neo4j_graph import Neo4jGraphBuilder
from src.core.ignore_filter import IgnoreFilter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_PROJECT_ROOT / "hyperrag_mcp.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),   # MCP uses stdout for protocol; log to stderr
    ],
)
logger = logging.getLogger("hyperrag.mcp")

# ── Lazy-initialised singletons ───────────────────────────────────────────────
_orchestrator: RetrievalOrchestrator | None = None
_context_builder: ContextBuilder | None = None
_generator: LLMGenerator | None = None
_ingestion_pipeline: IngestionPipeline | None = None


def _get_orchestrator() -> RetrievalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrievalOrchestrator()
    return _orchestrator


def _get_context_builder() -> ContextBuilder:
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder(max_tokens=12000)
    return _context_builder


def _get_generator() -> LLMGenerator | None:
    global _generator
    if _generator is None:
        if settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY or settings.NVIDIA_API_KEY:
            _generator = LLMGenerator()
    return _generator


def _get_ingestion_pipeline() -> IngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


# ── Server ────────────────────────────────────────────────────────────────────
server = Server("hyper-rag-mcp")


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_codebase",
            description=(
                "Run a full hybrid RAG query against all indexed projects in HyperRAG. "
                "Searches simultaneously across BM25 (keyword), vector (semantic), graph (code relationships), "
                "and PageIndex (document hierarchy) retrievers. Applies NVIDIA cross-encoder reranking "
                "and weighted fusion. Optionally generates an LLM answer.\n\n"
                "Use this when you need to:\n"
                "  • Understand how a function/class works\n"
                "  • Find where something is defined or called\n"
                "  • Get context from documentation or code\n"
                "  • Answer questions about an indexed codebase"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or code search query."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of final chunks to return (default: 10, max: 40).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 40,
                    },
                    "include_llm_answer": {
                        "type": "boolean",
                        "description": "If true, generates a full LLM answer from the retrieved context. Slower but complete.",
                        "default": False,
                    },
                    "weights": {
                        "type": "object",
                        "description": (
                            "Optional custom retrieval weights. Keys: rerank_score, bm25_score, "
                            "vector_score, graph_score. Must sum to 1.0. "
                            "Default: {rerank_score:0.35, bm25_score:0.25, vector_score:0.25, graph_score:0.15}"
                        ),
                        "properties": {
                            "rerank_score": {"type": "number"},
                            "bm25_score":   {"type": "number"},
                            "vector_score": {"type": "number"},
                            "graph_score":  {"type": "number"},
                        },
                    },
                },
                "required": ["query"],
            },
        ),

        types.Tool(
            name="ingest_folder",
            description=(
                "Ingest an entire project folder into HyperRAG's triple index "
                "(Qdrant vector store, OpenSearch BM25, Neo4j knowledge graph). "
                "Supports Python, JavaScript/TypeScript, Markdown, PDF, YAML, JSON, Go, Rust, C/C++, "
                "Java, HTML, CSS, and shell scripts. Respects .hyperragignore files.\n\n"
                "Use this when you want to:\n"
                "  • Index a new codebase or project for querying\n"
                "  • Re-index an updated project\n"
                "  • Add documentation to the knowledge base"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Absolute path to the project folder to ingest. E.g. D:/Projects/my-app",
                    },
                    "project_name": {
                        "type": "string",
                        "description": (
                            "Optional project name for scoping. Defaults to the folder's name. "
                            "Used to namespace artifacts in data/processed/."
                        ),
                    },
                },
                "required": ["folder_path"],
            },
        ),

        types.Tool(
            name="get_retrieval_chunks",
            description=(
                "Retrieve raw ranked chunks from HyperRAG without LLM generation. "
                "Returns structured JSON with chunk text, source file, scores per retriever, "
                "and final fusion score. Useful for debugging retrieval quality or building "
                "custom pipelines on top of HyperRAG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (default: 10).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 40,
                    },
                },
                "required": ["query"],
            },
        ),

        types.Tool(
            name="check_health",
            description=(
                "Check the health of all HyperRAG backend services: "
                "Qdrant (vector DB), OpenSearch (BM25), Neo4j (graph DB), and LLM provider. "
                "Returns status and document/chunk counts for each service."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),

        types.Tool(
            name="list_indexed_projects",
            description=(
                "List all projects that have been ingested and indexed in HyperRAG. "
                "Returns project names, file counts, and chunk counts from the local artifact store."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),

        types.Tool(
            name="wipe_project_index",
            description=(
                "⚠️ DESTRUCTIVE: Wipe all indexed data from Qdrant, OpenSearch, and Neo4j, "
                "and delete local artifacts for a given project. "
                "Use only when you want a clean re-index."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to proceed. Safety gate.",
                    },
                },
                "required": ["confirm"],
            },
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  TOOL HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    logger.info(f"Tool called: {name} | args: {json.dumps({k: v for k, v in arguments.items() if k != 'weights'}, default=str)}")

    try:
        if name == "query_codebase":
            return await _handle_query(arguments)
        elif name == "ingest_folder":
            return await _handle_ingest(arguments)
        elif name == "get_retrieval_chunks":
            return await _handle_get_chunks(arguments)
        elif name == "check_health":
            return await _handle_health()
        elif name == "list_indexed_projects":
            return await _handle_list_projects()
        elif name == "wipe_project_index":
            return await _handle_wipe(arguments)
        else:
            return [types.TextContent(type="text", text=f"❌ Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Tool {name} raised an unhandled exception")
        return [types.TextContent(type="text", text=f"❌ Internal error in `{name}`: {e}")]


# ── Individual handlers ────────────────────────────────────────────────────────

async def _handle_query(args: dict) -> list[types.TextContent]:
    query       = args["query"]
    top_k       = min(int(args.get("top_k", 10)), 40)
    include_llm = bool(args.get("include_llm_answer", False))
    weights     = args.get("weights", None)

    orchestrator   = _get_orchestrator()
    context_builder = _get_context_builder()
    generator       = _get_generator()

    ranked = await orchestrator.retrieve(query, final_top_k=top_k, weights=weights)

    if not ranked:
        return [types.TextContent(type="text", text=(
            "⚠️ No results found. Make sure you have ingested a project first "
            "using the `ingest_folder` tool."
        ))]

    context = context_builder.build_context(ranked)

    # ── Build result text ──────────────────────────────────────────────────────
    lines = [
        f"# HyperRAG Query Results",
        f"**Query:** {query}",
        f"**Retrieved:** {len(ranked)} chunks\n",
        "## Retrieved Context\n",
        context,
    ]

    if include_llm and generator:
        lines.append("\n## LLM Answer\n")
        answer_parts = []
        async for token in generator.generate_stream(query, context):
            answer_parts.append(token)
        lines.append("".join(answer_parts))
    elif include_llm and not generator:
        lines.append("\n⚠️ LLM generation requested but no valid API key is configured.")

    lines.append("\n---")
    lines.append(f"*Powered by HyperRAG — BM25 + Vector + Graph + PageIndex fusion*")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_ingest(args: dict) -> list[types.TextContent]:
    from tqdm import tqdm
    
    folder_path  = args["folder_path"]
    project_name = args.get("project_name", "")

    folder = Path(folder_path).resolve()
    if not folder.exists():
        return [types.TextContent(type="text", text=f"❌ Folder not found: `{folder_path}`")]

    if not project_name:
        project_name = folder.name

    pipeline      = _get_ingestion_pipeline()
    qdrant        = QdrantIndexer()
    opensearch    = OpenSearchIndexer()
    graph_builder = Neo4jGraphBuilder()

    # ── Setup indexes ─────────────────────────────────────────────────────────
    setup_errors = []
    try:
        qdrant.ensure_collection()
    except Exception as e:
        setup_errors.append(f"Qdrant setup warning: {e}")
    try:
        opensearch.ensure_index()
    except Exception as e:
        setup_errors.append(f"OpenSearch setup warning: {e}")
    try:
        graph_builder.ensure_constraints()
    except Exception as e:
        setup_errors.append(f"Neo4j setup warning: {e}")

    # ── Discover files ────────────────────────────────────────────────────────
    SEARCH_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".pdf",
        ".json", ".yaml", ".yml", ".c", ".cpp", ".cc", ".h", ".hpp",
        ".go", ".rs", ".java", ".kt", ".css", ".scss", ".html",
        ".sh", ".bash",
    }
    ignore_filter = IgnoreFilter(str(folder))
    files = [
        f for f in folder.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SEARCH_EXTS
        and not ignore_filter.is_ignored(str(f))
    ]

    total         = len(files)
    success_count = 0
    error_count   = 0
    total_chunks  = 0
    errors        = []

    logger.info(f"Ingesting [{project_name}] — {total} files found at {folder}")
    print(f"\n{'='*60}")
    print(f"Starting HyperRAG Ingestion: [{project_name.upper()}]")
    print(f"{'='*60}")
    print(f"Found {total} valid files")
    print(f"Saving artifacts to: data/processed/{project_name}/\n")

    with tqdm(total=total, desc="Indexing", unit="file", ncols=80, 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
              file=sys.stderr) as pbar:
        
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
                errors.append(f"  • {file_path.name}: {e}")
                pbar.write(f" Error in {file_path.name}: {e}")
                pbar.update(1)

    graph_builder.close()
    
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"Processed: {success_count} success | {error_count} failed")
    print(f"Total Chunks: {total_chunks}")
    print(f"{'='*60}\n")

    lines = [
        f"# ✅ Ingestion Complete — `{project_name}`",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Folder | `{folder}` |",
        f"| Files found | {total} |",
        f"| Files indexed | {success_count} |",
        f"| Files failed | {error_count} |",
        f"| Total chunks created | {total_chunks} |",
        f"",
    ]

    if setup_errors:
        lines.append("**Setup warnings:**")
        lines.extend(f"  - {w}" for w in setup_errors)
        lines.append("")

    if errors:
        lines.append(f"**{error_count} file(s) failed:**")
        lines.extend(errors)
        lines.append("")

    lines.append(f"You can now query this project with `query_codebase`.")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_get_chunks(args: dict) -> list[types.TextContent]:
    query = args["query"]
    top_k = min(int(args.get("top_k", 10)), 40)

    orchestrator = _get_orchestrator()
    ranked = await orchestrator.retrieve(query, final_top_k=top_k)

    if not ranked:
        return [types.TextContent(type="text", text="⚠️ No chunks found. Ensure a project has been ingested.")]

    lines = [
        f"# Raw Retrieval Chunks",
        f"**Query:** {query}  ",
        f"**Chunks returned:** {len(ranked)}\n",
    ]

    for i, chunk in enumerate(ranked, 1):
        source        = chunk.get("source", "unknown")
        final_score   = chunk.get("final_score", 0)
        rerank_score  = chunk.get("rerank_score", 0)
        bm25_score    = chunk.get("bm25_score", 0)
        vector_score  = chunk.get("vector_score", 0)
        graph_score   = chunk.get("graph_score", 0)
        section       = " > ".join(chunk.get("section_path", [])) if chunk.get("section_path") else ""
        text_preview  = chunk.get("text", "")[:400].strip()

        lines.append(f"---\n### Chunk {i} — `{source}`")
        if section:
            lines.append(f"**Section:** {section}")
        lines.append(
            f"**Scores:** Final `{final_score:.4f}` | Rerank `{rerank_score:.4f}` "
            f"| BM25 `{bm25_score:.4f}` | Vector `{vector_score:.4f}` | Graph `{graph_score:.4f}`"
        )
        lines.append(f"\n```\n{text_preview}{'...' if len(chunk.get('text','')) > 400 else ''}\n```\n")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_health() -> list[types.TextContent]:
    results = {}

    # ── Qdrant ────────────────────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_HOST, timeout=5)
        col_info = client.get_collection("hyper_rag_chunks")
        # Use points_count instead of deprecated vectors_count
        results["qdrant"] = {
            "status": "✅ healthy",
            "points_count": col_info.points_count,
        }
    except Exception as e:
        results["qdrant"] = {"status": f"❌ unreachable — {e}"}

    # ── OpenSearch ────────────────────────────────────────────────────────────
    try:
        from opensearchpy import OpenSearch
        os_client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=5,
        )
        stats = os_client.indices.stats(index="hyper_rag_bm25")
        doc_count = stats["_all"]["primaries"]["docs"]["count"]
        results["opensearch"] = {"status": "✅ healthy", "document_count": doc_count}
    except Exception as e:
        results["opensearch"] = {"status": f"❌ unreachable — {e}"}

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        with driver.session() as session:
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        driver.close()
        results["neo4j"] = {
            "status": "✅ healthy",
            "chunk_nodes": chunk_count,
            "entity_nodes": entity_count,
        }
    except Exception as e:
        results["neo4j"] = {"status": f"❌ unreachable — {e}"}

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_configured = bool(
        settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY or settings.NVIDIA_API_KEY
    )
    results["llm"] = {
        "status": "✅ configured" if llm_configured else "⚠️ no API key",
        "provider": settings.LLM_PROVIDER,
        "model": settings.NVIDIA_LLM_MODEL if settings.LLM_PROVIDER == "nvidia" else "default",
    }

    lines = ["# HyperRAG Health Check\n"]
    for service, info in results.items():
        lines.append(f"## {service.upper()}")
        for k, v in info.items():
            lines.append(f"  - **{k}:** {v}")
        lines.append("")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_list_projects() -> list[types.TextContent]:
    processed_dir = _PROJECT_ROOT / "data" / "processed"

    if not processed_dir.exists():
        return [types.TextContent(type="text", text=(
            "No projects indexed yet. Use `ingest_folder` to index your first project."
        ))]

    projects = [d for d in processed_dir.iterdir() if d.is_dir()]

    if not projects:
        return [types.TextContent(type="text", text=(
            "No projects found in `data/processed/`. Use `ingest_folder` to get started."
        ))]

    lines = ["# Indexed Projects\n"]
    lines.append(f"| Project | Chunk files | PageIndex files | Est. chunks |")
    lines.append(f"|---------|-------------|-----------------|-------------|")

    for project_dir in sorted(projects):
        chunk_files    = list(project_dir.glob("*_chunks.json"))
        pageindex_files = list(project_dir.glob("*_pageindex.json"))

        # Estimate total chunks
        total_chunks = 0
        for cf in chunk_files:
            try:
                with open(cf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_chunks += len(data)
            except Exception:
                pass

        lines.append(
            f"| `{project_dir.name}` | {len(chunk_files)} | {len(pageindex_files)} | ~{total_chunks} |"
        )

    lines.append(f"\n*Artifacts stored in `{processed_dir}`*")
    return [types.TextContent(type="text", text="\n".join(lines))]


async def _handle_wipe(args: dict) -> list[types.TextContent]:
    if not args.get("confirm", False):
        return [types.TextContent(type="text", text=(
            "⚠️ Wipe aborted. Set `confirm: true` to proceed. "
            "This will delete ALL data from Qdrant, OpenSearch, and Neo4j."
        ))]

    results = []

    # Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.QDRANT_HOST, timeout=10)
        client.delete_collection("hyper_rag_chunks")
        results.append("✅ Qdrant: collection deleted")
    except Exception as e:
        results.append(f"⚠️ Qdrant: {e}")

    # OpenSearch
    try:
        from opensearchpy import OpenSearch
        os_client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
            verify_certs=False, timeout=10,
        )
        os_client.indices.delete(index="hyper_rag_bm25", ignore=[400, 404])
        results.append("✅ OpenSearch: index deleted")
    except Exception as e:
        results.append(f"⚠️ OpenSearch: {e}")

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        results.append("✅ Neo4j: all nodes and relationships deleted")
    except Exception as e:
        results.append(f"⚠️ Neo4j: {e}")

    # Local artifacts
    import shutil
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    try:
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            results.append("✅ Local artifacts: wiped")
    except Exception as e:
        results.append(f"⚠️ Local artifacts: {e}")

    lines = ["# 🗑️ Index Wipe Complete\n"] + [f"  {r}" for r in results]
    lines.append("\nAll indexes cleared. You can now re-ingest with `ingest_folder`.")
    return [types.TextContent(type="text", text="\n".join(lines))]


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    logger.info("🚀 HyperRAG MCP Server starting...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
