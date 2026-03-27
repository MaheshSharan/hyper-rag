import asyncio
import logging
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, Dict, Any
from colorama import Fore, Style, init

# Initialize colorama for Windows/Unix
init(autoreset=True)

# Ensure 'src' is importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retrieval.retrieval_orchestrator import RetrievalOrchestrator
from src.retrieval.context_builder import ContextBuilder
from src.generation.generator import LLMGenerator
from src.ingestion.pipeline import IngestionPipeline
from src.analysis.project_summarizer import ProjectSummarizer
from src.config.settings import settings

# ====================== PRO LOGGING ======================
class ColoredFormatter(logging.Formatter):
    """Custom formatter for a more readable, color-coded terminal experience"""
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.WHITE + "%(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "⚠️ %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "❌ %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + "🚨 %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, "%(message)s")
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# File handler for persistency
file_handler = logging.FileHandler("hyperrag.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
root_logger.addHandler(file_handler)

# Stream handler for the "Pro" terminal look
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColoredFormatter())
root_logger.addHandler(stream_handler)

logger = logging.getLogger("hyperrag")

# Prevent Duplicate Loggers from FastAPI/Uvicorn if we want control
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="HyperRAG API", version="1.0", description="Advanced Hybrid RAG Backend")

# ====================== MODELS ======================
class QueryRequest(BaseModel):
    query: str
    final_top_k: int = 10
    include_llm: bool = False
    stream: bool = True
    weights: Optional[Dict[str, float]] = None

class IngestRequest(BaseModel):
    folder_path: str

class SummarizeRequest(BaseModel):
    folder_path: str
    max_files: int = 30
    include_llm_analysis: bool = False  # Default FALSE like other tools

# ====================== SERVICES ======================
orchestrator = RetrievalOrchestrator()
context_builder = ContextBuilder(max_tokens=12000)
generator = LLMGenerator() if (settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY or settings.NVIDIA_API_KEY) else None
ingestion_pipeline = IngestionPipeline()

# ====================== STREAMING HELPERS ======================
async def stream_query_response(request: QueryRequest) -> AsyncGenerator[str, None]:
    try:
        yield 'data: {"choices":[{"delta":{"role":"assistant","content":"[START]"},"index":0}]}\n\n'

        logger.info(Fore.CYAN + f"🔎 Querying: {Style.BRIGHT}{request.query}")
        
        ranked_results = await orchestrator.retrieve(
            request.query, 
            final_top_k=request.final_top_k,
            weights=request.weights
        )
        context = context_builder.build_context(ranked_results)

        if request.include_llm and generator:
            logger.info(Fore.MAGENTA + f"Generating Response via {settings.LLM_PROVIDER.upper()}...")
            async for token in generator.generate_stream(request.query, context):
                yield f'data: {json.dumps({"choices": [{"delta": {"content": token}, "index": 0}]})}\n\n'
                await asyncio.sleep(0.01)
        else:
            summary = f"[RETRIEVAL_ONLY] Found {len(ranked_results)} relevant chunks."
            yield f'data: {json.dumps({"choices": [{"delta": {"content": summary}, "index": 0}]})}\n\n'

        logger.info(Fore.GREEN + "Response Streamed Successfully.\n")
        yield 'data: [DONE]\n\n'

    except Exception as e:
        logger.error(f"Query Process Failed: {e}")
        yield f'data: {json.dumps({"error": str(e)})}\n\n'

async def stream_ingest_response(folder_path: str) -> AsyncGenerator[str, None]:
    try:
        folder = Path(folder_path)
        if not folder.exists():
            raise HTTPException(400, f"Folder not found: {folder_path}")

        exclude_dirs = {".git", "node_modules", "__pycache__", ".next", "venv", ".venv", 
                       "dist", "build", ".pytest_cache", "coverage", "logs", "out", 
                       ".turbo", ".output", ".idea", ".vscode"}
        exclude_extensions = {".min.js", ".bundle.js", ".chunk.js"}
        exclude_filenames = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "tsconfig.tsbuildinfo"}

        files = [
            f for f in folder.rglob("*") 
            if f.is_file() 
            and f.suffix.lower() in [".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".pdf", ".json", ".yaml", ".yml"]
            and not any(part in exclude_dirs for part in f.parts)
            and not any(f.name.endswith(ext) for ext in exclude_extensions)
            and not any(f.name == name for name in exclude_filenames)
            and not f.name.startswith(".")
            and not any(k in f.name.lower() for k in ["turbopack", "webpack", "minified"])
        ]

        total = len(files)
        processed = 0

        yield 'data: {"status":"started","progress":0,"message":"Starting ingestion..."}\n\n'

        for file_path in files:
            processed += 1
            progress = int((processed / total) * 100) if total > 0 else 100
            
            # Use logger for ingestion too
            logger.info(Fore.BLUE + f"[{progress}%] Indexing: {file_path.name}")

            result = await ingestion_pipeline.ingest_file(str(file_path))

            data = {
                "status": "progress", "progress": progress, "file": str(file_path.relative_to(folder)),
                "chunks": len(result.get("chunks") or []),
                "pageindex_nodes": len(result.get("pageindex_nodes") or [])
            }
            yield f'data: {json.dumps(data)}\n\n'

        yield 'data: {"status":"completed","progress":100,"message":"Ingestion completed successfully"}\n\n'
        logger.info(Fore.GREEN + f"Ingestion complete for {folder_path}")

    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        yield f'data: {json.dumps({"status":"error","error":str(e)})}\n\n'

# ====================== ENDPOINTS ======================
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if request.stream:
        return StreamingResponse(stream_query_response(request), media_type="text/event-stream")
    else:
        logger.info(Fore.CYAN + f"🔎 Sync Query: {request.query}")
        ranked = await orchestrator.retrieve(request.query, request.final_top_k, request.weights)
        ctx = context_builder.build_context(ranked)
        answer = None
        if request.include_llm and generator:
            answer = await generator.generate(request.query, ctx)
        return {"ranked_results": ranked, "context": ctx, "answer": answer}

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    return StreamingResponse(stream_ingest_response(request.folder_path), media_type="text/event-stream")

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    """Fast project summarization with optional LLM intelligence"""
    try:
        folder = Path(request.folder_path)
        if not folder.exists():
            raise HTTPException(400, f"Folder not found: {request.folder_path}")
        
        logger.info(Fore.CYAN + f"📊 Summarizing project: {folder.name}")
        
        summarizer = ProjectSummarizer(str(folder))
        summary = summarizer.summarize(
            max_files_to_analyze=request.max_files,
            include_llm_analysis=request.include_llm_analysis
        )
        
        logger.info(Fore.GREEN + f"✅ Summary complete: {summary['total_files']} files analyzed")
        
        return summary
    
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(500, f"Summarization error: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_enabled": generator is not None,
        "llm_provider": settings.LLM_PROVIDER if generator else None
    }

def print_banner():
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}    __  __                          ____  ___   ______
{Fore.CYAN}{Style.BRIGHT}   / / / /_  ______  ___  _________ / __ \/   | / ____/
{Fore.CYAN}{Style.BRIGHT}  / /_/ / / / / __ \/ _ \/ ___/ __  / /_/ / /| |/ / __  
{Fore.CYAN}{Style.BRIGHT} / __  / /_/ / /_/ /  __/ /  / /_/ / _, _/ ___ / /_/ /  
{Fore.CYAN}{Style.BRIGHT}/_/ /_/\__, / .___/\___/_/   \__,_/_/ |_/_/  |_\____/   
{Fore.CYAN}{Style.BRIGHT}      /____/_/                                          
    {Fore.WHITE}-------------------------------------------------- {Fore.CYAN}v1.0
    {Fore.WHITE}Headless Hybrid RAG Backend for IDE-Intelligence
    {Fore.WHITE}--------------------------------------------------
    {Fore.GREEN}INFO {Fore.WHITE}LLM Provider:   {Fore.YELLOW}{settings.LLM_PROVIDER.upper()}
    {Fore.GREEN}INFO {Fore.WHITE}Embeddings:     {Fore.YELLOW}{settings.NVIDIA_EMBEDDING_MODEL.split('/')[-1]}
    {Fore.GREEN}INFO {Fore.WHITE}Reranker:       {Fore.YELLOW}{settings.NVIDIA_RERANK_MODEL.split('/')[-1]}
    {Fore.GREEN}INFO {Fore.WHITE}Graph DB:       {Fore.YELLOW}Neo4j (Active)
    {Fore.GREEN}INFO {Fore.WHITE}Vector DB:      {Fore.YELLOW}Qdrant (Active)
    {Fore.WHITE}--------------------------------------------------
    """
    print(banner)

if __name__ == "__main__":
    import uvicorn
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        print_banner()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    else:
        print("Use --api flag to start server.")
