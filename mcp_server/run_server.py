"""
Entry-point for HyperRAG MCP Server.
Explicitly sets working directory to the project root before anything imports.
"""
import os
import sys
from pathlib import Path

# ── Force working directory to project root ───────────────────────────────────
# This file lives at: hyper-rag/mcp_server/run_server.py
# Project root is:    hyper-rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ── Ensure src is importable ──────────────────────────────────────────────────
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Now safe to import and run ────────────────────────────────────────────────
from hyper_rag_mcp.server import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
