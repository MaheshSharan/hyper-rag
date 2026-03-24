
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.config.settings import settings
from typing import List

class Embedder:
	def __init__(self):
		self.embedder = NVIDIAEmbeddings(
			model=settings.NVIDIA_EMBEDDING_MODEL,
			api_key=settings.NVIDIA_API_KEY,
			truncate="END",
			# We can add dimensions=2048 here if the model supports it in future
		)
		self.dimension = 2048  # Matches current model

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		if not texts:
			return []
		try:
			embeddings = self.embedder.embed_documents(texts)
			# Quick validation
			if embeddings and len(embeddings[0]) != self.dimension:
				print(f"⚠️  Unexpected embedding dimension: {len(embeddings[0])} (expected {self.dimension})")
			return embeddings
		except Exception as e:
			print(f"❌ Embedding error: {e}")
			raise
