import re
from typing import Dict

class TextCleaner:
	"""Simple but effective text cleaner - part of SemTools preprocessing"""

	def clean(self, text: str, remove_code_blocks: bool = False) -> str:
		if not text:
			return ""

		# Normalize whitespace
		text = re.sub(r'\s+', ' ', text)
		text = text.strip()

		# Remove excessive newlines
		text = re.sub(r'\n\s*\n', '\n\n', text)

		if remove_code_blocks:
			# Basic code block removal (can be improved later)
			text = re.sub(r'```[\s\S]*?```', '', text)

		return text

	def extract_metadata(self, text: str) -> Dict:
		"""Basic metadata extraction"""
		return {
			"approx_token_count": len(text.split()),
			"has_code": "```" in text or "def " in text or "class " in text,
		}
