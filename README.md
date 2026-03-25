# HyperRAG — Advanced Hybrid RAG Backend

Production-grade hybrid RAG system with structured ingestion, multi-retrieval, NVIDIA reranking, graph truth layer, and intelligent fusion.

Designed as a **portable backend** for VS Code extensions, IDEs, and CLI tools. It provides a robust context-gathering engine that can be used independently of any specific LLM.

## Architecture

User Query / IDE
↓
[ /query ] → Query Planner → BM25 + Vector + Graph + PageIndex
↓
NVIDIA Reranker → Fusion Layer (Weighted Fusion)
↓
Context Builder → [Optional] LLM Generator

```mermaid
flowchart TD
	A["Raw Data Sources"] --> B["SemTools Parser"]

	B --> C1["AST Parser Tree-sitter"]
	B --> C2["Text Cleaner normalize"]
	B --> C3["PageIndex Builder hierarchical"]

	C1 --> D1["Structured code"]
	C2 --> D2["Clean chunks"]
	C3 --> D3["Tree index JSON"]

	D1 --> E1["BM25 OpenSearch"]
	D1 --> E2["Embedding NVIDIA"]
	D1 --> E3["Graph dependencies"]

	D2 --> E1
	D2 --> E2
	D2 --> E3

	D3 --> E1
	D3 --> E2
	D3 --> E3

	E1 --> F1["Search ready"]
	E2 --> F2["Vector DB Qdrant"]
	E3 --> F3["Graph DB Neo4j"]
```

## Features
- **Hybrid Retrieval:** BM25 + Vector + Graph + Selective PageIndex.
- **Dynamic Truth Layer:** Neo4j for relationship-based knowledge.
- **Live Indexing:** Real-time progress (SSE) for easy integration into IDE progress bars.
- **Retrieval-Only Mode:** Use the system as a pure context-builder (no LLM key required).
- **Quality Gate:** NVIDIA reranking ensures higher accuracy before context construction.

## Documentation
For detailed endpoint specifications, request/response examples, and integration patterns, see the [API Documentation](docs/api.md).

## Quick Start

```bash
# 1. Start databases
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your keys in .env (if using LLM)

# 4. Start API server
python src/main.py --api
```

**Interactive API Docs (Swagger):** `http://127.0.0.1:8000/docs`
