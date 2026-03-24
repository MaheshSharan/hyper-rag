import json
from pathlib import Path
from typing import List, Dict, Any

class PageIndexRouter:
    def __init__(self):
        self.processed_dir = Path("data/processed")

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Selective PageIndex router - only on docs + tree traversal"""
        try:
            candidates = []
            # Load all PageIndex files
            for json_file in self.processed_dir.glob("*_pageindex.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    nodes = json.load(f)

                # Simple relevance: check if query keywords appear in titles or content
                for node in nodes:
                    node_text = (node.get("title", "") + " " + node.get("content", "")).lower()
                    query_lower = query.lower()
                    score = sum(1 for word in query_lower.split() if word in node_text)

                    if score > 0:
                        candidates.append({
                            "chunk_id": node.get("node_id", "unknown"),
                            "text": node.get("content", ""),
                            "source": json_file.stem.replace("_pageindex", ""),
                            "pageindex_score": float(score),
                            "section_path": [node.get("title", "")],
                            "metadata": {"retriever": "pageindex", "hierarchy_level": node.get("level", 0)}
                        })

            # Sort + limit
            candidates.sort(key=lambda x: x.get("pageindex_score", 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            print(f"❌ PageIndex router error: {e}")
            return []