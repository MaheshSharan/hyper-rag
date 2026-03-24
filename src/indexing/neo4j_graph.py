# Neo4j graph writer

from neo4j import GraphDatabase
from src.config.settings import settings
from src.core.schemas import DocumentChunk
import re
from typing import List, Dict

class Neo4jGraphBuilder:
	def __init__(self):
		self.driver = GraphDatabase.driver(
			settings.NEO4J_URI,
			auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
		)
		self.database = "neo4j"

	def close(self):
		self.driver.close()

	def ensure_constraints(self):
		with self.driver.session(database=self.database) as session:
			# Unique constraints
			session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
			session.run("CREATE CONSTRAINT file_name IF NOT EXISTS FOR (f:File) REQUIRE f.name IS UNIQUE")
			print("✅ Neo4j constraints created")

	def build_graph_from_chunks(self, chunks: List[DocumentChunk], file_name: str):
		with self.driver.session(database=self.database) as session:
			# Create File node
			session.run("""
				MERGE (f:File {name: $file_name})
				SET f.file_type = $file_type
			""", file_name=file_name, file_type=chunks[0].metadata.file_type if chunks else "unknown")

			for chunk in chunks:
				if chunk.metadata.file_type != "code":
					continue  # For now we only build graph from code files

				# Create Chunk node
				session.run("""
					MERGE (c:Chunk {chunk_id: $chunk_id})
					SET c.text = $text,
						c.source = $source,
						c.language = $language
				""", 
					chunk_id=chunk.chunk_id,
					text=chunk.text[:10000],  # limit size
					source=chunk.metadata.source,
					language=chunk.metadata.language or ""
				)

				# Link Chunk to File
				session.run("""
					MATCH (f:File {name: $file_name})
					MATCH (c:Chunk {chunk_id: $chunk_id})
					MERGE (f)-[:CONTAINS]->(c)
				""", file_name=file_name, chunk_id=chunk.chunk_id)

				# Extract simple dependencies (imports, function calls)
				self._extract_dependencies(session, chunk)

		print(f"   → Built graph for {file_name} in Neo4j")

	def _extract_dependencies(self, session, chunk: DocumentChunk):
		text = chunk.text
		language = chunk.metadata.language or ""

		if language == "python":
			# Extract imports
			imports = re.findall(r'^\s*(?:from|import)\s+([^\s]+)', text, re.MULTILINE)
			for imp in imports:
				session.run("""
					MATCH (c:Chunk {chunk_id: $chunk_id})
					MERGE (dep:Entity {name: $dep_name, type: 'module'})
					MERGE (c)-[:IMPORTS]->(dep)
				""", chunk_id=chunk.chunk_id, dep_name=imp.strip())

			# Extract function calls (very basic)
			calls = re.findall(r'(\w+)\s*\(', text)
			for call in calls:
				if call not in ['def', 'if', 'for', 'while', 'print', 'len']:
					session.run("""
						MATCH (c:Chunk {chunk_id: $chunk_id})
						MERGE (func:Entity {name: $func_name, type: 'function'})
						MERGE (c)-[:CALLS]->(func)
					""", chunk_id=chunk.chunk_id, func_name=call)

		# We can add JS, Java etc. later
# Neo4j graph writer
