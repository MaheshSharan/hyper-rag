import json
from pathlib import Path
from typing import List, Dict

class PageIndexStore:
    def __init__(self):
        self.base_dir = Path("data/processed")

    def save_pageindex(self, file_name: str, nodes: List[Dict]):
        """Already saved in pipeline, but this can be used for loading later"""
        pass  # For now we reuse the JSON files saved by pipeline

    def load_pageindex_for_file(self, file_name: str) -> List[Dict]:
        path = self.base_dir / f"{file_name}_pageindex.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
