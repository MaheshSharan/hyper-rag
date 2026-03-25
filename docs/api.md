# HyperRAG — API Documentation

This document describes the endpoints available in the HyperRAG backend.

## Base URL
`http://127.0.0.1:8000`

---

## Endpoints

### 1. Query Endpoint
`POST /query`

Executes a hybrid search across all indexes and optionally generates an LLM response.

**Request Body:**
```json
{
  "query": "What does the hello_world function do?",
  "final_top_k": 10,
  "include_llm": false,
  "stream": true
}
```

**Parameters:**
- `query` (string): The user's question or search term.
- `final_top_k` (int): Number of top results to return after fusion.
- `include_llm` (bool): If `false`, returns only chunks/context (perfect for IDE extensions). If `true`, requires an LLM API key.
- `stream` (bool): If `true`, returns a Server-Sent Events (SSE) stream.

**Use Cases:**
- **Retrieval-Only:** Set `include_llm: false` to get the raw context for your own local LLM.
- **Full RAG:** Set `include_llm: true` for a complete streaming answer.

---

### 2. Index Codebase (Live Progress)
`POST /ingest`

Scans a local directory and indexes all supported files into Qdrant, OpenSearch, and Neo4j.

**Request Body:**
```json
{
  "folder_path": "D:/Projects/my-codebase"
}
```

**Response:**
Returns a **Server-Sent Event (SSE)** stream with real-time progress:
- `status`: "started", "progress", "completed", or "error"
- `progress`: Integer (0–100)
- `file`: Path of the file currently being processed
- `chunks`: Number of chunks created for the file
- `pageindex_nodes`: Number of PageIndex tree nodes created

---

### 3. Health Check
`GET /health`

Returns the current health status of the service and confirms if LLM features are enabled.

**Response Example:**
```json
{
  "status": "healthy",
  "llm_enabled": true,
  "llm_provider": "openai"
}
```

---

## Integration Guide for VS Code Extensions

1. **Indexing:** Call `/ingest` when the user clicks an "Index Codebase" button. Listen to the SSE stream to update your progress bar.
2. **Context Retrieval:** Call `/query` with `include_llm: false` to get the ranked context and chunks.
3. **Local LLM:** Take the returned `context` and pass it to your extension's own LLM provider (like Copilot, Ollama, or a custom model).

### Swagger UI
Full interactive documentation and testing is available at:
`http://127.0.0.1:8000/docs`
