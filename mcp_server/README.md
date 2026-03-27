# HyperRAG MCP Server

Production-grade [Model Context Protocol](https://modelcontextprotocol.io) server that bridges **Claude Desktop** directly into your running HyperRAG backend.

---

## What This Does

Once installed, Claude Desktop can call eight tools against your HyperRAG instance:

| Tool | Description |
|---|---|
| `query_codebase` | Full hybrid RAG query — BM25 + Vector + Graph + PageIndex + NVIDIA reranking |
| `ingest_folder` | Index any project folder into all three stores (Qdrant, OpenSearch, Neo4j) |
| `summarize_project` | ⚡ Fast project analysis WITHOUT indexing — tech stack, structure, key files (~3s, optional LLM insights) |
| `get_retrieval_chunks` | Raw ranked chunks with per-retriever scores — great for debugging |
| `check_health` | Live health check of all services (Qdrant, OpenSearch, Neo4j, LLM) |
| `list_indexed_projects` | Show all ingested projects with chunk counts |
| `wipe_project_index` | ⚠️ Full wipe of all indexes (requires `confirm: true`) |
| `get_metrics` | Performance metrics and statistics about retrieval operations |

---

## Prerequisites

1. **HyperRAG running** — Docker services up: `docker-compose up -d` from the project root  
2. **Python 3.11+** — inside the project's `venv` (recommended)  
3. **Claude Desktop** installed  

---

## Installation

```bash
# From the hyper-rag project root
cd mcp_server

# Install MCP SDK into the project venv
..\venv\Scripts\pip install mcp anyio

# Run the one-shot Claude Desktop config patcher
..\venv\Scripts\python setup_claude_desktop.py
```

That's it. **Restart Claude Desktop** and you'll see `hyper-rag` in the 🔌 integrations panel.

---

## Manual Config (if you prefer)

Add this to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hyper-rag": {
      "command": "D:\\Projects\\hyper-rag\\venv\\Scripts\\python.exe",
      "args": ["D:\\Projects\\hyper-rag\\mcp_server\\run_server.py"],
      "env": {
        "PYTHONPATH": "D:\\Projects\\hyper-rag",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## Usage Examples

Once connected, ask Claude Desktop naturally:

> *"Summarize the project at D:/Projects/my-app"*  
> → Calls `summarize_project` (fast, no indexing needed)

> *"Ingest my project at D:/Projects/my-app"*  
> → Calls `ingest_folder`

> *"How does the authentication middleware work?"*  
> → Calls `query_codebase` against all indexed projects

> *"Check if HyperRAG is healthy"*  
> → Calls `check_health`

> *"Show me the raw retrieval scores for the query: database connection pooling"*  
> → Calls `get_retrieval_chunks`

> *"Show me performance metrics"*  
> → Calls `get_metrics`

---

## Architecture

```
Claude Desktop
     │  MCP (stdio)
     ▼
run_server.py
     │
hyper_rag_mcp/server.py
     │
     ├── RetrievalOrchestrator  (BM25 + Vector + Graph + PageIndex → Rerank → Fusion)
     ├── ContextBuilder         (token-aware context assembly)
     ├── LLMGenerator           (OpenAI / Anthropic / NVIDIA streaming)
     ├── IngestionPipeline      (AST parse → embed → triple-index)
     └── QdrantIndexer / OpenSearchIndexer / Neo4jGraphBuilder
```

---

## Logs

All MCP server activity is logged to `hyperrag_mcp.log` in the project root.

---

## Remote Hosting (Optional)

To expose the server over HTTP/SSE instead of stdio (for use with Claude.ai Integrations):

1. Wrap `server.py` with a FastAPI/SSE adapter using `mcp[server]` extras  
2. Deploy to any cloud (Fly.io, Railway, AWS EC2)  
3. Add the public URL in Claude.ai → Settings → Integrations  

A production HTTP wrapper (`server_http.py`) will be added in v1.1.
