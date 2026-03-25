from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from typing import List, Dict
import hashlib
import logging

logger = logging.getLogger("hyperrag.ast_parser")

class ASTParser:
    """Tree-sitter based parser for code files (updated for tree-sitter >= 0.22)"""

    def __init__(self):
        try:
            # Load languages
            self.PY_LANGUAGE = Language(tspython.language())
            self.JS_LANGUAGE = Language(tsjavascript.language())

            # Create parsers
            self.python_parser = Parser(self.PY_LANGUAGE)
            self.js_parser = Parser(self.JS_LANGUAGE)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Tree-Sitter grammars: {e}")
            self.python_parser = None
            self.js_parser = None

    def parse_code(self, code: str, language: str) -> List[Dict]:
        lang = language.lower()
        
        if not self.python_parser or not self.js_parser:
             return self._fallback_full_file(code, lang)

        if lang in ["python", "py"]:
            parser = self.python_parser
            lang_name = "python"
        elif lang in ["javascript", "js", "ts", "jsx", "tsx"]:
            parser = self.js_parser
            lang_name = "javascript" # using JS grammar for TS/JSX fallback
        else:
            return self._fallback_full_file(code, lang)

        # Parse the code
        tree = parser.parse(bytes(code, "utf8"))

        chunks = []
        # Extract top-level functions and classes
        for node in tree.root_node.children:
            if node.type in ["function_definition", "class_definition", "method_definition", 
                           "function_declaration", "class_declaration", "lexical_declaration"]:
                chunk_text = code[node.start_byte : node.end_byte]
                if not chunk_text.strip():
                    continue
                    
                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]

                chunks.append({
                    "chunk_id": f"code_{chunk_id}",
                    "text": chunk_text,
                    "type": node.type,
                    "language": lang_name,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                })

        # If no meaningful chunks found, fall back to full file
        if not chunks:
            return self._fallback_full_file(code, lang_name)

        return chunks

    def _fallback_full_file(self, code: str, language: str) -> List[Dict]:
        chunk_id = hashlib.md5(code.encode()).hexdigest()[:12]
        return [{
            "chunk_id": f"code_{chunk_id}",
            "text": code,
            "type": "full_file",
            "language": language,
        }]