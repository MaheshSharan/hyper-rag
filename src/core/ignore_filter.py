import os
from pathlib import Path
import fnmatch

class IgnoreFilter:
    """Universal ignore filter for different project types (Python, JS, Rust, Go, etc.)"""
    
    DEFAULT_IGNORES = {
        # General / VCS
        ".git", ".svn", ".hg", ".idea", ".vscode", ".DS_Store", "node_modules",
        
        # Python
        "__pycache__", "venv", ".venv", ".pytest_cache", ".tox", ".mypy_cache", "*.pyc",
        
        # JavaScript / Web
        ".next", ".turbo", ".output", "dist", "build", "coverage", "out", "target", "vendor",
        
        # Rust / C++
        "target", "*.o", "*.obj", "*.dll", "*.so", "*.exe",
        
        # Logs / Data
        "logs", "*.log", ".hyperrag", "data/processed", ".env"
    }

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.ignore_patterns = set(self.DEFAULT_IGNORES)
        self._load_local_ignore_file()

    def _load_local_ignore_file(self):
        """Loads .hyperragignore if it exists in the project root"""
        ignore_file = self.root_dir / ".hyperragignore"
        if ignore_file.exists():
            try:
                with open(ignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            self.ignore_patterns.add(line)
            except Exception:
                pass

    def is_ignored(self, path: str) -> bool:
        """Checks if a file or directory should be ignored"""
        try:
            p = Path(path).resolve()
            rel_path = str(p.relative_to(self.root_dir))
            parts = rel_path.split(os.sep)
            
            # Check if any directory part is in the ignore list
            for part in parts:
                if part in self.ignore_patterns:
                    return True
                
                # Check for glob patterns (e.g., *.pyc)
                for pattern in self.ignore_patterns:
                    if fnmatch.fnmatch(part, pattern):
                        return True
            
            return False
        except Exception:
            return True # If path is weird, ignore it for safety
