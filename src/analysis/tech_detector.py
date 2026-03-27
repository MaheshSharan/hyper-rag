"""
Technology stack detection from project files
Identifies languages, frameworks, and dependencies
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger("hyperrag.tech_detector")


class TechDetector:
    """Detects tech stack from file extensions and package manifests"""
    
    # Language detection by extension
    LANGUAGE_MAP = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "JavaScript (React)",
        ".tsx": "TypeScript (React)",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".kt": "Kotlin",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".dart": "Dart",
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def detect(self, files: List[Path]) -> Dict:
        """Detect full tech stack"""
        result = {
            "languages": self._detect_languages(files),
            "frameworks": [],
            "dependencies": [],
            "package_managers": [],
            "build_tools": []
        }
        
        # Check for package manifests
        package_json = self.project_root / "package.json"
        if package_json.exists():
            npm_info = self._parse_package_json(package_json)
            result["frameworks"].extend(npm_info["frameworks"])
            result["dependencies"].extend(npm_info["dependencies"])
            result["package_managers"].append("npm/yarn")
        
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            py_info = self._parse_pyproject(pyproject)
            result["dependencies"].extend(py_info["dependencies"])
            result["package_managers"].append("pip/poetry")
        
        requirements = self.project_root / "requirements.txt"
        if requirements.exists():
            req_info = self._parse_requirements(requirements)
            result["dependencies"].extend(req_info)
            if "pip/poetry" not in result["package_managers"]:
                result["package_managers"].append("pip")
        
        cargo = self.project_root / "Cargo.toml"
        if cargo.exists():
            result["package_managers"].append("cargo")
        
        go_mod = self.project_root / "go.mod"
        if go_mod.exists():
            result["package_managers"].append("go modules")
        
        # Detect build tools
        if (self.project_root / "webpack.config.js").exists():
            result["build_tools"].append("webpack")
        if (self.project_root / "vite.config.js").exists() or (self.project_root / "vite.config.ts").exists():
            result["build_tools"].append("vite")
        if (self.project_root / "Makefile").exists():
            result["build_tools"].append("make")
        if (self.project_root / "docker-compose.yml").exists():
            result["build_tools"].append("docker-compose")
        
        logger.info(f"Detected stack: {len(result['languages'])} languages, {len(result['frameworks'])} frameworks")
        return result
    
    def _detect_languages(self, files: List[Path]) -> List[str]:
        """Detect languages from file extensions"""
        languages: Set[str] = set()
        
        for file_path in files:
            ext = file_path.suffix.lower()
            if ext in self.LANGUAGE_MAP:
                languages.add(self.LANGUAGE_MAP[ext])
        
        return sorted(languages)
    
    def _parse_package_json(self, path: Path) -> Dict:
        """Parse package.json for frameworks and dependencies"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            frameworks = []
            dependencies = []
            
            all_deps = {}
            all_deps.update(data.get("dependencies", {}))
            all_deps.update(data.get("devDependencies", {}))
            
            # Detect frameworks
            if "react" in all_deps:
                frameworks.append("React")
            if "vue" in all_deps:
                frameworks.append("Vue")
            if "@angular/core" in all_deps:
                frameworks.append("Angular")
            if "next" in all_deps:
                frameworks.append("Next.js")
            if "express" in all_deps:
                frameworks.append("Express")
            if "fastify" in all_deps:
                frameworks.append("Fastify")
            
            # Top dependencies (limit to 10)
            dependencies = list(all_deps.keys())[:10]
            
            return {"frameworks": frameworks, "dependencies": dependencies}
        
        except Exception as e:
            logger.warning(f"Failed to parse package.json: {e}")
            return {"frameworks": [], "dependencies": []}
    
    def _parse_pyproject(self, path: Path) -> Dict:
        """Parse pyproject.toml for dependencies"""
        try:
            import tomli
            with open(path, 'rb') as f:
                data = tomli.load(f)
            
            dependencies = []
            
            # Poetry format
            if "tool" in data and "poetry" in data["tool"]:
                deps = data["tool"]["poetry"].get("dependencies", {})
                dependencies = [k for k in deps.keys() if k != "python"][:10]
            
            # PEP 621 format
            elif "project" in data:
                deps = data["project"].get("dependencies", [])
                dependencies = [d.split("[")[0].split(">")[0].split("=")[0].strip() for d in deps][:10]
            
            return {"dependencies": dependencies}
        
        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml: {e}")
            return {"dependencies": []}
    
    def _parse_requirements(self, path: Path) -> List[str]:
        """Parse requirements.txt"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            dependencies = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (before ==, >=, etc.)
                    pkg = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                    dependencies.append(pkg)
            
            return dependencies[:10]
        
        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")
            return []
