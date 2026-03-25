import hashlib
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.schemas import DocumentChunk, ChunkMetadata
from src.ingestion.parsers.semtools import SemToolsParser
from src.ingestion.parsers.ast_parser import ASTParser
from src.ingestion.parsers.pageindex_builder import PageIndexBuilder
from src.ingestion.embedder import Embedder

logger = logging.getLogger("hyperrag.ingestion")

class IngestionPipeline:
    def __init__(self):
        self.semtools = SemToolsParser()
        self.ast_parser = ASTParser()
        self.pageindex_builder = PageIndexBuilder()
        self.embedder = Embedder()

    def get_output_dir(self, project_name: str = "default"):
        """Creates a project-scoped directory for processed files"""
        out = Path("data/processed") / project_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    async def ingest_file(self, file_path: str, project_name: str = "default") -> Dict[str, Any]:
        """Main entry point for ingesting one file into a project scope"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
        except Exception as e:
            logger.error(f"  ❌ Could not read file {file_path}: {e}")
            return {"file": os.path.basename(file_path), "error": str(e)}

        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')

        # SemTools preprocessing
        parsed = self.semtools.parse(raw_text, file_ext)

        chunks: List[DocumentChunk] = []
        pageindex_nodes = None

        # Language mapping for Tree-Sitter
        code_exts = ["py", "js", "ts", "jsx", "tsx", "java", "go", "cpp", "c"]
        
        if file_ext in code_exts:
            code_chunks = self.ast_parser.parse_code(raw_text, file_ext)
            for c in code_chunks:
                chunk = DocumentChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    metadata=ChunkMetadata(
                        chunk_id=c["chunk_id"],
                        source=file_name,
                        file_type="code",
                        language=c.get("language"),
                        metadata={k: v for k, v in c.items() if k != "text"}
                    )
                )
                chunks.append(chunk)

        else:  # Documents / Markdown / Text
            # Build PageIndex tree
            pageindex_nodes = self.pageindex_builder.build_tree(parsed["cleaned_text"], file_name)

            # Create chunks from PageIndex nodes
            for node in pageindex_nodes:
                if node.content and node.content.strip():
                    chunk_id = f"doc_{hashlib.md5(node.content.encode()).hexdigest()[:12]}"
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        text=node.content,
                        metadata=ChunkMetadata(
                            chunk_id=chunk_id,
                            source=file_name,
                            file_type="document",
                            section_path=[node.title],
                            hierarchy_level=node.level
                        )
                    )
                    chunks.append(chunk)

            # Fallback if no headings found
            if not chunks and parsed["cleaned_text"].strip():
                chunk_id = f"doc_{hashlib.md5(parsed['cleaned_text'].encode()).hexdigest()[:12]}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=parsed["cleaned_text"],
                    metadata=ChunkMetadata(
                        chunk_id=chunk_id,
                        source=file_name,
                        file_type=file_ext or "text"
                    )
                )
                chunks.append(chunk)

        # Generate embeddings
        if chunks:
            texts = [c.text for c in chunks]
            try:
                embeddings = self.embedder.embed_texts(texts)
                for chunk, emb in zip(chunks, embeddings):
                    chunk.embedding = emb
            except Exception as e:
                logger.warning(f"  ⚠️ Embedding failed for {file_name}: {e}")

        # === SAVE RESULTS TO PROJECT FOLDER ===
        output_dir = self.get_output_dir(project_name)
        self._save_chunks(file_name, chunks, output_dir)
        if pageindex_nodes:
            self._save_pageindex(file_name, pageindex_nodes, output_dir)

        logger.info(f"  → {len(chunks)} chunks created for project [{project_name}]")

        return {
            "file": file_name,
            "chunks": chunks,
            "pageindex_nodes": pageindex_nodes
        }

    def _save_chunks(self, file_name: str, chunks: List[DocumentChunk], output_dir: Path):
        """Save chunks as JSON for later indexing"""
        output_path = output_dir / f"{file_name}_chunks.json"
        data = [chunk.dict() for chunk in chunks]
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _save_pageindex(self, file_name: str, nodes: List, output_dir: Path):
        """Save PageIndex tree"""
        output_path = output_dir / f"{file_name}_pageindex.json"
        data = [node.dict() for node in nodes]
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
