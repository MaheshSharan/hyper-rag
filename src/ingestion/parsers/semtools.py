from src.ingestion.parsers.text_cleaner import TextCleaner
from typing import Dict, Any

class SemToolsParser:
	"""Your custom preprocessing layer - acts as the entry point"""

	def __init__(self):
		self.cleaner = TextCleaner()

	def parse(self, raw_text: str, file_type: str) -> Dict[str, Any]:
		cleaned = self.cleaner.clean(raw_text)
		metadata = self.cleaner.extract_metadata(cleaned)

		return {
			"cleaned_text": cleaned,
			"file_type": file_type,
			"metadata": metadata,
			"structured": self._basic_structure(cleaned, file_type)
		}

	def _basic_structure(self, text: str, file_type: str) -> Dict:
		"""Simple structure detection - will be enhanced with PageIndex later"""
		if file_type in ["md", "markdown"]:
			return {"type": "markdown"}
		elif file_type == "pdf":
			return {"type": "document"}
		return {"type": "plain_text"}
