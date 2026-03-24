import hashlib
import os
import json
from typing import List, Dict, Any
from pathlib import Path
from src.core.schemas import DocumentChunk, ChunkMetadata
from src.ingestion.parsers.semtools import SemToolsParser
from src.ingestion.parsers.ast_parser import ASTParser
from src.ingestion.parsers.pageindex_builder import PageIndexBuilder
from src.ingestion.embedder import Embedder

class IngestionPipeline:
	def __init__(self):
		self.semtools = SemToolsParser()
		self.ast_parser = ASTParser()
		self.pageindex_builder = PageIndexBuilder()
		self.embedder = Embedder()

	@property
	def output_dir(self):
		out = Path("data/processed")
		out.mkdir(exist_ok=True)
		return out
	async def ingest_file(self, file_path: str) -> Dict[str, Any]:
		"""Main entry point for ingesting one file + saving results"""
		import json
		from pathlib import Path
		try:
			with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
				raw_text = f.read()
		except Exception as e:
			print(f"  ❌ Could not read file {file_path}: {e}")
			return {"file": os.path.basename(file_path), "error": str(e)}

		file_name = os.path.basename(file_path)
		file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')

		# SemTools preprocessing
		parsed = self.semtools.parse(raw_text, file_ext)

		chunks: List[DocumentChunk] = []
		pageindex_nodes = None

		if file_ext in ["py", "js", "ts", "java", "go", "cpp", "c"]:  # Code files
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
				print(f"  ⚠️  Embedding failed for {file_name}: {e}")
				# Continue without embeddings for now

		# === SAVE RESULTS ===
		self._save_chunks(file_name, chunks)
		if pageindex_nodes:
			self._save_pageindex(file_name, pageindex_nodes)

		print(f"  → {len(chunks)} chunks created | PageIndex nodes: {len(pageindex_nodes) if pageindex_nodes else 0}")

		return {
			"file": file_name,
			"chunks": chunks,
			"pageindex_nodes": pageindex_nodes
		}

	def _save_chunks(self, file_name: str, chunks: List[DocumentChunk]):
		"""Save chunks as JSON for later indexing"""
		output_path = self.output_dir / f"{file_name}_chunks.json"
		data = [chunk.dict() for chunk in chunks]
		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(data, f, indent=2, ensure_ascii=False, default=str)

	def _save_pageindex(self, file_name: str, nodes: List):
		"""Save PageIndex tree"""
		output_path = self.output_dir / f"{file_name}_pageindex.json"
		data = [node.dict() for node in nodes]
		with open(output_path, 'w', encoding='utf-8') as f:
			json.dump(data, f, indent=2, ensure_ascii=False, default=str)
