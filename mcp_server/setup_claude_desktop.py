"""
setup_claude_desktop.py
=======================
One-shot script that patches claude_desktop_config.json to register
the HyperRAG MCP server with Claude Desktop.

Usage:
    python setup_claude_desktop.py

Run once after installing the MCP server. Claude Desktop must be
restarted after running this script.
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ── Detect venv Python ─────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
VENV_PYTHON    = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
SYSTEM_PYTHON  = shutil.which("python") or sys.executable
PYTHON_EXE     = str(VENV_PYTHON) if VENV_PYTHON.exists() else SYSTEM_PYTHON

SERVER_SCRIPT  = str(PROJECT_ROOT / "mcp_server" / "run_server.py")

# ── Claude Desktop config path ────────────────────────────────────────────────
CONFIG_PATH = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"

MCP_ENTRY = {
    "command": PYTHON_EXE,
    "args": [SERVER_SCRIPT],
    "env": {
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
    },
}


def main():
    print(f"\n{'='*55}")
    print("  HyperRAG MCP — Claude Desktop Setup")
    print(f"{'='*55}")
    print(f"  Config  : {CONFIG_PATH}")
    print(f"  Python  : {PYTHON_EXE}")
    print(f"  Script  : {SERVER_SCRIPT}")
    print(f"{'='*55}\n")

    if not CONFIG_PATH.parent.exists():
        print(f"❌ Claude Desktop config directory not found: {CONFIG_PATH.parent}")
        print("   Make sure Claude Desktop is installed.")
        sys.exit(1)

    # Load existing config or create fresh
    if CONFIG_PATH.exists():
        # Backup first
        backup = CONFIG_PATH.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(CONFIG_PATH, backup)
        print(f"📋 Backed up existing config → {backup.name}")

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Patch in HyperRAG entry
    config["mcpServers"]["hyper-rag"] = MCP_ENTRY

    # Write back
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ claude_desktop_config.json updated successfully!")
    print(f"\n   MCP server key : 'hyper-rag'")
    print(f"   Total MCP servers registered: {len(config['mcpServers'])}")
    print(f"\n⚡ Restart Claude Desktop to activate the HyperRAG MCP server.")
    print(f"   You'll see 'hyper-rag' appear in the 🔌 integrations panel.\n")


if __name__ == "__main__":
    main()
