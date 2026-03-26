import logging
from src.core.connection_pool import connection_pool
from src.core.schemas import DocumentChunk
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from typing import List

logger = logging.getLogger("hyperrag.neo4j")


class Neo4jGraphBuilder:
    def __init__(self):
        self.database = "neo4j"
        
        # Load languages for Tree-Sitter
        try:
            self.PY_LANGUAGE = Language(tspython.language())
            self.JS_LANGUAGE = Language(tsjavascript.language())
            
            self.python_parser = Parser(self.PY_LANGUAGE)
            self.js_parser = Parser(self.JS_LANGUAGE)
            logger.info("Tree-Sitter parsers initialized for Neo4j Graph Builder")
        except Exception as e:
            logger.error(f"Failed to initialize Tree-Sitter: {e}")
            self.python_parser = None
            self.js_parser = None

    def close(self):
        """No-op: connection pool handles cleanup"""
        pass

    def ensure_constraints(self):
        with connection_pool.neo4j.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
            session.run("CREATE CONSTRAINT file_name IF NOT EXISTS FOR (f:File) REQUIRE f.name IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        logger.info("Neo4j constraints created")

    def build_graph_from_chunks(self, chunks: List[DocumentChunk], file_name: str):
        with connection_pool.neo4j.session(database=self.database) as session:
            session.run("""
                MERGE (f:File {name: $file_name})
                SET f.file_type = $file_type, f.last_indexed = timestamp()
            """, file_name=file_name, file_type=chunks[0].metadata.file_type if chunks else "unknown")

            for chunk in chunks:
                if chunk.metadata.file_type != "code":
                    continue

                session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.source = $source,
                        c.language = $language,
                        c.file = $file_name
                """, 
                    chunk_id=chunk.chunk_id,
                    text=chunk.text[:10000],
                    source=chunk.metadata.source,
                    language=chunk.metadata.language or "",
                    file_name=file_name
                )

                session.run("""
                    MATCH (f:File {name: $file_name})
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (f)-[:CONTAINS]->(c)
                """, file_name=file_name, chunk_id=chunk.chunk_id)

                self._extract_code_relationships_ast(session, chunk)

        logger.info(f"   → Built world-class graph for {file_name} in Neo4j")

    def _extract_code_relationships_ast(self, session, chunk: DocumentChunk):
        """Uses Tree-Sitter to extract relationships instead of Regex"""
        text = chunk.text
        lang = (chunk.metadata.language or "").lower()
        
        if not self.python_parser or not self.js_parser:
            return  # Fallback failed
            
        code_bytes = bytes(text, "utf8")
        
        if lang in ["python", "py"]:
            tree = self.python_parser.parse(code_bytes)
            self._traverse_python_tree(session, chunk.chunk_id, tree.root_node, code_bytes)
        elif lang in ["javascript", "js", "ts", "jsx", "tsx"]:
            tree = self.js_parser.parse(code_bytes)
            self._traverse_js_tree(session, chunk.chunk_id, tree.root_node, code_bytes)

    def _traverse_python_tree(self, session, chunk_id, root, code_bytes):
        """Python relationship traverser"""
        for node in root.children:
            # Imports
            if node.type in ["import_statement", "import_from_statement"]:
                # Simple extraction for now - find module names
                for child in node.children:
                    if child.type == "dotted_name":
                        name = code_bytes[child.start_byte:child.end_byte].decode("utf8")
                        self._safe_entity_merge(session, chunk_id, name, 'module', 'IMPORTS')
                    elif child.type == "aliased_import":
                        # Handle 'import x as y'
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                name = code_bytes[sub.start_byte:sub.end_byte].decode("utf8")
                                self._safe_entity_merge(session, chunk_id, name, 'module', 'IMPORTS')

            # Definitions
            elif node.type in ["function_definition", "class_definition"]:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
                    self._safe_entity_merge(session, chunk_id, name, 'definition', 'DEFINES')

            # Recurse for nested calls (like inside functions)
            self._find_calls(session, chunk_id, node, code_bytes, "python")

    def _traverse_js_tree(self, session, chunk_id, root, code_bytes):
        """JS/TS relationship traverser"""
        for node in root.children:
            # Imports
            if node.type == "import_statement":
                source = node.child_by_field_name("source")
                if source:
                    name = code_bytes[source.start_byte:source.end_byte].decode("utf8").strip("'\"")
                    self._safe_entity_merge(session, chunk_id, name, 'module', 'IMPORTS')

            # Definitions
            elif node.type in ["function_declaration", "class_declaration"]:
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
                    self._safe_entity_merge(session, chunk_id, name, 'definition', 'DEFINES')
            
            elif node.type == "variable_declaration":
                # Handle const x = () => ...
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_node = child.child_by_field_name("name")
                        value_node = child.child_by_field_name("value")
                        if name_node and value_node and value_node.type in ["arrow_function", "function_expression"]:
                            name = code_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
                            self._safe_entity_merge(session, chunk_id, name, 'definition', 'DEFINES')

            self._find_calls(session, chunk_id, node, code_bytes, "javascript")

    def _find_calls(self, session, chunk_id, node, code_bytes, lang_family):
        """Recursive call finder (ignores control flow)"""
        if node.type == "call" if lang_family == "python" else node.type == "call_expression":
            # For Python: call -> function
            # For JS: call_expression -> function
            func_node = node.child_by_field_name("function")
            if func_node:
                name = code_bytes[func_node.start_byte:func_node.end_byte].decode("utf8")
                # Filter out obvious keywords
                if name not in {'if', 'for', 'while', 'print', 'console.log', 'super'}:
                    self._safe_entity_merge(session, chunk_id, name, 'function', 'CALLS')

        for child in node.children:
            self._find_calls(session, chunk_id, child, code_bytes, lang_family)

    def _safe_entity_merge(self, session, chunk_id, name, entity_type, relation):
        """Safe merge that respects unique constraint on name while allowing type updates"""
        if not name or len(name) > 255: return
        session.run(f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            MERGE (e:Entity {{name: $name}})
            ON CREATE SET e.type = $type
            MERGE (c)-[:{relation}]->(e)
        """, chunk_id=chunk_id, name=name, type=entity_type)
