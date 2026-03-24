# HyperRAG — Advanced Hybrid RAG System

Production-grade Retrieval-Augmented Generation with:
- Structured ingestion (AST + PageIndex + Graph)
- Hybrid retrieval (BM25 + Vector + Graph)
- NVIDIA reranking + intelligent fusion
- Selective PageIndex reasoning
- Graph as truth layer

## Architecture

**Pipeline Overview:**
- Raw data (code, docs, PDFs, logs)
	→ SemTools parser (clean + structured)
		→ AST parser, Text cleaner, PageIndex builder
			→ BM25 (OpenSearch), Vector DB (Qdrant), Graph DB (Neo4j)
				→ Hybrid retrieval, NVIDIA reranking, Fusion

(Arch Diag)

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

## Quick Start

```bash
cp .env.example .env
# Fill your NVIDIA API key, OpenSearch/Qdrant/Neo4j credentials

pip install -r requirements.txt

# Ingest sample data
python scripts/ingest_folder.py data/raw/

# Run a test query
python scripts/query_cli.py "Your question here"
```

## Modules

src/ingestion/ — SemTools + Tree-sitter + PageIndex  
src/retrieval/ — Multi-retriever + NVIDIA reranker + Fusion  
etc.