import asyncio
import logging
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Optional

from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.retrieval.context_builder import ContextBuilder
from src.generation.generator import LLMGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.config.settings import settings

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("hyperrag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("hyperrag")

app = FastAPI(title="HyperRAG API", version="1.0", description="Advanced Hybrid RAG Backend")

# ====================== MODELS ======================
class QueryRequest(BaseModel):
    query: str
    final_top_k: int = 10
    include_llm: bool = False          # Default: retrieval-only (no LLM key needed)
    stream: bool = True

class IngestRequest(BaseModel):
    folder_path: str

# ====================== SERVICES ======================
orchestrator = RetrievalOrchestrator()
context_builder = ContextBuilder(max_tokens=12000)
generator = LLMGenerator() if settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY else None
ingestion_pipeline = IngestionPipeline()

logger.info(f"🚀 HyperRAG started | LLM enabled: {generator is not None} | Provider: {settings.LLM_PROVIDER}")

# ====================== STREAMING HELPERS ======================
async def stream_query_response(request: QueryRequest) -> AsyncGenerator[str, None]:
    try:
        yield 'data: {"choices":[{"delta":{"role":"assistant","content":"[START]"},"index":0}]}\n\n'

        ranked_results = await orchestrator.retrieve(request.query, final_top_k=request.final_top_k)
        context = context_builder.build_context(ranked_results)

        if request.include_llm and generator:
            async for token in generator.generate_stream(request.query, context):
                yield f'data: {json.dumps({"choices": [{"delta": {"content": token}, "index": 0}]})}\n\n'
                await asyncio.sleep(0.01)
        else:
            # Retrieval-only mode
            summary = f"[RETRIEVAL_ONLY] Found {len(ranked_results)} relevant chunks. Context ready."
            yield f'data: {json.dumps({"choices": [{"delta": {"content": summary}, "index": 0}]})}\n\n'

        yield 'data: [DONE]\n\n'

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        yield f'data: {json.dumps({"error": str(e)})}\n\n'

async def stream_ingest_response(folder_path: str) -> AsyncGenerator[str, None]:
    try:
        folder = Path(folder_path)
        if not folder.exists():
            raise HTTPException(400, f"Folder not found: {folder_path}")

        # Improved exclude list - production grade
        exclude_dirs = {".git", "node_modules", "__pycache__", ".next", "venv", ".venv", 
                       "dist", "build", ".pytest_cache", "coverage", "logs"}
        exclude_files = {".min.js", ".bundle.js", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

        files = [
            f for f in folder.rglob("*") 
            if f.is_file() 
            and f.suffix.lower() in [".py", ".js", ".ts", ".md", ".txt", ".pdf", ".json", ".yaml", ".yml"]
            and not any(part in exclude_dirs for part in f.parts)
            and not any(f.name.endswith(ext) for ext in exclude_files)
            and not f.name.startswith(".")
        ]

        total = len(files)
        processed = 0

        yield 'data: {"status":"started","progress":0,"message":"Starting ingestion..."}\n\n'

        for file_path in files:
            processed += 1
            progress = int((processed / total) * 100) if total > 0 else 100

            logger.info(f"[{progress}%] Processing {file_path.name}")

            result = await ingestion_pipeline.ingest_file(str(file_path))

            data = {
                "status": "progress",
                "progress": progress,
                "file": str(file_path.relative_to(folder)),
                "chunks": len(result.get("chunks") or []),
                "pageindex_nodes": len(result.get("pageindex_nodes") or [])
            }
            yield f'data: {json.dumps(data)}\n\n'

        yield 'data: {"status":"completed","progress":100,"message":"Ingestion + indexing completed successfully"}\n\n'
        logger.info(f"✅ Ingestion completed for {folder_path}")

    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        yield f'data: {json.dumps({"status":"error","error":str(e)})}\n\n'

# ====================== ENDPOINTS ======================
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Universal endpoint - works with or without LLM"""
    if request.stream:
        return StreamingResponse(stream_query_response(request), media_type="text/event-stream")
    else:
        ranked = await orchestrator.retrieve(request.query, request.final_top_k)
        ctx = context_builder.build_context(ranked)
        answer = None
        if request.include_llm and generator:
            answer = await generator.generate(request.query, ctx)
        return {"ranked_results": ranked, "context": ctx, "answer": answer}

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """Live indexing with detailed progress (perfect for VS Code extension)"""
    return StreamingResponse(stream_ingest_response(request.folder_path), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_enabled": generator is not None,
        "llm_provider": settings.LLM_PROVIDER if generator else None
    }

# ====================== CLI ======================
if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        logger.info("🚀 Starting HyperRAG API server → http://127.0.0.1:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Use --api flag to start server or call endpoints directly.")
