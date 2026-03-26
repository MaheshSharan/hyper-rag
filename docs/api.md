# HyperRAG — API Documentation

Professional-grade API for hybrid document search, code retrieval, and agentic context-gathering.

---

## Connection Info
- **Base URL:** `http://127.0.0.1:8000`
- **Interactive Documentation:** `http://127.0.0.1:8000/docs` (Swagger UI)

---

## Endpoints

### 1. Hybrid Search & Generation
`POST /query`

Executes a parallel, quad-retrieval search (Vector + BM25 + Graph + PageIndex) followed by NVIDIA Reranking and Weighted Fusion.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `string` | **Required** | The question or search term. |
| `final_top_k` | `int` | `10` | The number of results to return after fusion. |
| `include_llm` | `bool` | `False` | Whether to generate an answer (requires an LLM API key). |
| `stream` | `bool` | `True` | Whether to use Server-Sent Events (SSE). |
| `weights` | `dict` | `None` | (Optional) Custom fusion weights. |

#### **Example Request:**
```json
{
  "query": "What is the payment verification logic?",
  "final_top_k": 8,
  "include_llm": true,
  "stream": true,
  "weights": {
    "rerank_score": 0.40,
    "bm25_score": 0.20,
    "vector_score": 0.25,
    "graph_score": 0.15
  }
}
```

#### **Real-time Thinking Support:**
If using a thinking-capable LLM, the stream will include reasoning-tokens interleaved with content. These tokens are prefixed with a thinking tag (e.g., [THINKING] or emoji-like indicator) in the content delta as provided by the model provider.

---

### 2. Live Ingestion
`POST /ingest`

Scans a folder and builds the multi-index project scope. Files are automatically organized into project-scoped subfolders under data/processed/{folder_name}/.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `folder_path` | `string` | **Required** | The absolute path to the directory. |

#### **Progress SSE Stream:**
Returns a stream of progress status events:
```json
{
  "status": "progress",
  "progress": 45,
  "file": "verify-payment.ts",
  "chunks": 12,
  "pageindex_nodes": 3
}
```

---

### 3. Service Health Check
`GET /health`

Checks the status of the RAG engine and its connected LLM provider.

#### **Response:**
```json
{
  "status": "healthy",
  "llm_enabled": true,
  "llm_provider": "nvidia"
}
```

---

## Integration Patterns

### 1. Retrieval-only (Extension Mode)
If you are building an IDE extension and have your own local LLM (e.g., Ollama or Copilot-integrated), set `include_llm: false`. You will receive high-relevance chunks and a pre-built context string that you can inject directly into your local prompt.

### 2. Agentic Ingestion
Integrate /ingest into your project setup workflow. Monitor the progress field to update your UI's progress bar in real-time.

### 3. Manual Override Weights
For code-heavy searches, increase graph_score. For keyword-heavy legacy documentation, increase bm25_score. The system defaults are optimized for modern codebases (0.35 Rerank, 0.25 Vector, 0.25 BM25, 0.15 Graph).

---
© 2026 HyperRAG
