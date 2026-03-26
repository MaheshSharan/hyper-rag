import logging
from typing import List, Dict, Any
import tiktoken

logger = logging.getLogger("hyperrag.context")


class ContextBuilder:
    def __init__(self, max_tokens: int = 12000):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Works for both OpenAI and Claude

    def build_context(self, ranked_results: List[Dict[str, Any]]) -> str:
        """Build structured context from final ranked chunks"""
        context_parts = []
        current_tokens = 0

        for i, item in enumerate(ranked_results, 1):
            source = item.get("source", "unknown")
            section = " > ".join(item.get("section_path", [])) if item.get("section_path") else ""
            text = item["text"].strip()

            header = f"--- Source: {source}"
            if section:
                header += f" | Section: {section}"
            header += f" | Relevance: {item.get('final_score', 0):.3f} ---\n"

            chunk_text = header + text + "\n\n"

            # Token check
            chunk_tokens = len(self.encoder.encode(chunk_text))
            if current_tokens + chunk_tokens > self.max_tokens:
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        context = "".join(context_parts)

        logger.info(f"Context built: {len(context_parts)} chunks | ~{current_tokens} tokens")
        return context
