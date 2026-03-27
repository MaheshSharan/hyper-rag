"""
File importance scoring system
Ranks files by their architectural significance without hardcoding project types
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger("hyperrag.file_scorer")


class FileScorer:
    """Scores files by importance using universal heuristics"""
    
    # Universal important filenames (entry points, configs, docs)
    PRIORITY_NAMES = {
        "readme", "main", "index", "app", "server", "api", "routes",
        "package.json", "pyproject.toml", "cargo.toml", "go.mod",
        "setup.py", "tsconfig.json", "webpack.config", "vite.config",
        "docker-compose", "dockerfile", "makefile"
    }
    
    # Architectural folder names
    CORE_FOLDERS = {
        "core", "lib", "utils", "services", "api", "routes", 
        "controllers", "models", "handlers", "middleware"
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def score_files(self, files: List[Path]) -> List[Tuple[Path, float]]:
        """Score all files and return sorted by importance"""
        scored = []
        
        for file_path in files:
            score = self._calculate_score(file_path)
            scored.append((file_path, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Scored {len(files)} files. Top score: {scored[0][1] if scored else 0}")
        return scored
    
    def _calculate_score(self, file_path: Path) -> float:
        """Calculate importance score for a single file"""
        score = 0.0
        
        try:
            relative = file_path.relative_to(self.project_root)
        except ValueError:
            return 0.0
        
        # 1. Root level files are important
        depth = len(relative.parts) - 1
        if depth == 0:
            score += 30.0
        elif depth == 1:
            score += 15.0
        elif depth == 2:
            score += 5.0
        else:
            score -= (depth - 2) * 2.0  # Penalty for deep nesting
        
        # 2. Priority filenames
        name_lower = file_path.stem.lower()
        for priority in self.PRIORITY_NAMES:
            if priority in name_lower or priority in file_path.name.lower():
                score += 25.0
                break
        
        # 3. Core architectural folders
        parts_lower = [p.lower() for p in relative.parts]
        for core_folder in self.CORE_FOLDERS:
            if core_folder in parts_lower:
                score += 20.0
                break
        
        # 4. File size (smaller entry points are better)
        try:
            size = file_path.stat().st_size
            if size < 5000:  # < 5KB
                score += 10.0
            elif size < 20000:  # < 20KB
                score += 5.0
            elif size > 100000:  # > 100KB (likely generated or data)
                score -= 10.0
        except Exception:
            pass
        
        # 5. Extension bonuses
        ext = file_path.suffix.lower()
        if ext in [".md", ".txt"]:
            # Documentation
            if "readme" in name_lower:
                score += 15.0
            else:
                score += 5.0
        elif ext in [".json", ".toml", ".yaml", ".yml"]:
            # Config files
            score += 10.0
        elif ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"]:
            # Source code
            score += 8.0
        
        return max(score, 0.0)
    
    def get_top_files(self, files: List[Path], top_n: int = 20) -> List[Path]:
        """Get the N most important files"""
        scored = self.score_files(files)
        return [f for f, _ in scored[:top_n]]
