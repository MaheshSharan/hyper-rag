"""
Fast project summarization without indexing
Analyzes raw files to provide instant codebase understanding
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.ignore_filter import IgnoreFilter
from src.analysis.file_scorer import FileScorer
from src.analysis.tech_detector import TechDetector
from src.ingestion.parsers.ast_parser import ASTParser
from src.generation.generator import LLMGenerator
from src.config.settings import settings

logger = logging.getLogger("hyperrag.project_summarizer")


class ProjectSummarizer:
    """Generates comprehensive project summaries from raw files"""
    
    SEARCH_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt",
        ".json", ".yaml", ".yml", ".c", ".cpp", ".cc", ".h", ".hpp",
        ".go", ".rs", ".java", ".kt", ".css", ".scss", ".html",
        ".sh", ".bash", ".toml"
    }
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.ignore_filter = IgnoreFilter(str(self.project_root))
        self.file_scorer = FileScorer(self.project_root)
        self.tech_detector = TechDetector(self.project_root)
        self.ast_parser = ASTParser()
        
        # Initialize LLM generator if available (but don't use by default)
        self.generator = None
        if settings.OPENAI_API_KEY or settings.ANTHROPIC_API_KEY or settings.NVIDIA_API_KEY:
            self.generator = LLMGenerator()
    
    def summarize(self, max_files_to_analyze: int = 30, use_cache: bool = True, include_llm_analysis: bool = False) -> Dict:
        """
        Generate fast project summary with optional LLM intelligence
        
        Args:
            max_files_to_analyze: Number of key files to analyze
            use_cache: If True, return cached summary if available
            include_llm_analysis: If True, generate LLM-powered insights (slower but smarter)
        
        Returns:
            {
                "project_name": str,
                "total_files": int,
                "tech_stack": {...},
                "structure": {...},
                "key_files": [...],
                "readme_content": str | None,
                "llm_analysis": str | None  # Only if include_llm_analysis=True
            }
        """
        logger.info(f"Starting project summarization: {self.project_root}")
        
        # Check for cached summary
        cache_path = self._get_cache_path()
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                logger.info(f"Using cached summary from {cache_path}")
                return cached
            except Exception as e:
                logger.warning(f"Failed to load cached summary: {e}")
        
        # 1. Discover all files
        all_files = self._discover_files()
        logger.info(f"Discovered {len(all_files)} files")
        
        # 2. Detect tech stack
        tech_stack = self.tech_detector.detect(all_files)
        
        # 3. Analyze project structure
        structure = self._analyze_structure(all_files)
        
        # 4. Score and select key files
        top_files = self.file_scorer.get_top_files(all_files, top_n=max_files_to_analyze)
        logger.info(f"Selected {len(top_files)} key files for analysis")
        
        # 5. Extract key file summaries
        key_files = self._analyze_key_files(top_files)
        
        # 6. Extract README content
        readme_content = self._extract_readme()
        
        # 7. LLM-powered intelligent analysis (OPTIONAL)
        llm_analysis = None
        if include_llm_analysis and self.generator:
            logger.info("Generating LLM-powered project analysis...")
            llm_analysis = self._generate_llm_analysis(
                tech_stack, structure, key_files, readme_content
            )
        elif include_llm_analysis and not self.generator:
            logger.warning("LLM analysis requested but no API key configured")
        
        summary = {
            "project_name": self.project_root.name,
            "project_path": str(self.project_root),
            "total_files": len(all_files),
            "tech_stack": tech_stack,
            "structure": structure,
            "key_files": key_files,
            "readme_content": readme_content,
            "llm_analysis": llm_analysis
        }
        
        # Save to cache
        self._save_cache(summary, cache_path)
        
        logger.info(f"Summary complete: {len(key_files)} key files analyzed")
        return summary
    
    def _get_cache_path(self) -> Path:
        """Get path to cached summary file"""
        from pathlib import Path
        project_name = self.project_root.name
        cache_dir = Path(__file__).parent.parent.parent / "data" / "processed" / project_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "_project_summary.json"
    
    def _save_cache(self, summary: Dict, cache_path: Path):
        """Save summary to cache"""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache summary: {e}")
    
    def _discover_files(self) -> List[Path]:
        """Discover all relevant files in the project"""
        files = []
        
        for file_path in self.project_root.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix.lower() not in self.SEARCH_EXTS:
                continue
            
            # Check ignore filter
            if self.ignore_filter.is_ignored(str(file_path)):
                continue
            
            files.append(file_path)
        
        return files
    
    def _analyze_structure(self, files: List[Path]) -> Dict:
        """Analyze project folder structure"""
        folders = set()
        file_types = {}
        
        for file_path in files:
            try:
                relative = file_path.relative_to(self.project_root)
                
                # Track folders
                if len(relative.parts) > 1:
                    folders.add(relative.parts[0])
                
                # Track file types
                ext = file_path.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            except ValueError:
                continue
        
        return {
            "top_level_folders": sorted(folders),
            "file_types": dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:15])
        }
    
    def _analyze_key_files(self, files: List[Path]) -> List[Dict]:
        """Extract summaries from key files"""
        key_files = []
        
        for file_path in files:
            try:
                relative_path = str(file_path.relative_to(self.project_root))
            except ValueError:
                relative_path = file_path.name
            
            file_info = {
                "path": relative_path,
                "type": self._classify_file(file_path),
                "size": file_path.stat().st_size,
            }
            
            # Extract content based on file type
            ext = file_path.suffix.lower()
            
            if ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"]:
                # Parse code structure
                structure = self._parse_code_structure(file_path)
                if structure:
                    file_info["structure"] = structure
            
            elif ext in [".md", ".txt"]:
                # Extract text preview
                preview = self._extract_text_preview(file_path, max_lines=20)
                if preview:
                    file_info["preview"] = preview
            
            elif ext in [".json", ".yaml", ".yml", ".toml"]:
                # Config file - just note it
                file_info["description"] = "Configuration file"
            
            key_files.append(file_info)
        
        return key_files
    
    def _classify_file(self, file_path: Path) -> str:
        """Classify file by its role"""
        name_lower = file_path.name.lower()
        
        if "readme" in name_lower:
            return "documentation"
        elif "test" in name_lower or "spec" in name_lower:
            return "test"
        elif file_path.suffix.lower() in [".json", ".yaml", ".yml", ".toml"]:
            return "config"
        elif file_path.suffix.lower() in [".md", ".txt"]:
            return "documentation"
        elif "main" in name_lower or "index" in name_lower or "app" in name_lower:
            return "entry_point"
        else:
            return "source"
    
    def _parse_code_structure(self, file_path: Path) -> Optional[Dict]:
        """Parse code file structure using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use AST parser
            parsed = self.ast_parser.parse(content, str(file_path))
            
            if not parsed:
                return None
            
            # Extract key elements
            functions = [f["name"] for f in parsed.get("functions", [])][:10]
            classes = [c["name"] for c in parsed.get("classes", [])][:10]
            imports = parsed.get("imports", [])[:15]
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "total_lines": content.count('\n') + 1
            }
        
        except Exception as e:
            logger.debug(f"Failed to parse {file_path.name}: {e}")
            return None
    
    def _extract_text_preview(self, file_path: Path, max_lines: int = 20) -> Optional[str]:
        """Extract preview from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
                
                return '\n'.join(lines)
        
        except Exception as e:
            logger.debug(f"Failed to read {file_path.name}: {e}")
            return None
    
    def _extract_readme(self) -> Optional[str]:
        """Find and extract README content"""
        readme_names = ["README.md", "readme.md", "README.txt", "README"]
        
        for name in readme_names:
            readme_path = self.project_root / name
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Limit to first 2000 chars
                        return content[:2000] if len(content) > 2000 else content
                except Exception as e:
                    logger.warning(f"Failed to read README: {e}")
        
        return None
    
    def _generate_llm_analysis(
        self, 
        tech_stack: Dict, 
        structure: Dict, 
        key_files: List[Dict],
        readme_content: Optional[str]
    ) -> Optional[str]:
        """Generate intelligent project analysis using LLM"""
        try:
            # Build context for LLM
            context_parts = []
            
            # Add README if available
            if readme_content:
                context_parts.append(f"=== README ===\n{readme_content}\n")
            
            # Add tech stack
            context_parts.append(f"=== TECH STACK ===")
            context_parts.append(f"Languages: {', '.join(tech_stack.get('languages', []))}")
            if tech_stack.get('frameworks'):
                context_parts.append(f"Frameworks: {', '.join(tech_stack['frameworks'])}")
            if tech_stack.get('dependencies'):
                context_parts.append(f"Key Dependencies: {', '.join(tech_stack['dependencies'][:10])}")
            context_parts.append("")
            
            # Add structure
            context_parts.append(f"=== PROJECT STRUCTURE ===")
            context_parts.append(f"Top Folders: {', '.join(structure['top_level_folders'][:10])}")
            context_parts.append(f"Total Files: {sum(structure['file_types'].values())}")
            context_parts.append("")
            
            # Add key files with their structure
            context_parts.append(f"=== KEY FILES (Top 15) ===")
            for i, file_info in enumerate(key_files[:15], 1):
                context_parts.append(f"\n{i}. {file_info['path']} ({file_info['type']})")
                
                if 'structure' in file_info:
                    struct = file_info['structure']
                    if struct.get('classes'):
                        context_parts.append(f"   Classes: {', '.join(struct['classes'][:5])}")
                    if struct.get('functions'):
                        context_parts.append(f"   Functions: {', '.join(struct['functions'][:5])}")
                    if struct.get('imports'):
                        context_parts.append(f"   Imports: {', '.join(struct['imports'][:5])}")
                
                elif 'preview' in file_info:
                    preview = file_info['preview'][:300]
                    context_parts.append(f"   Preview: {preview}...")
            
            context = "\n".join(context_parts)
            
            # Create analysis prompt
            prompt = f"""You are analyzing a codebase to provide intelligent insights for developers.

Based on the following project information, provide a comprehensive analysis:

{context}

Please provide:
1. **Project Purpose**: What this project does (2-3 sentences)
2. **Architecture Overview**: How the code is organized and key architectural patterns
3. **Entry Points**: Main files where execution starts
4. **Key Components**: Important modules/classes and their roles
5. **Development Workflow**: How to build, test, and run this project
6. **Notable Patterns**: Any interesting design patterns or best practices used

Be specific, technical, and actionable. Focus on helping developers understand the codebase quickly."""

            # Generate analysis
            logger.info("Calling LLM for intelligent analysis...")
            analysis = self.generator.generate_sync(prompt, "")
            
            return analysis
        
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
