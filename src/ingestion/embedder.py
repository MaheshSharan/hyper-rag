import logging
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from src.config.settings import settings
from src.core.cache import embedding_cache
from typing import List

logger = logging.getLogger("hyperrag.embedder")


class Embedder:
	def __init__(self):
		self.embedder = NVIDIAEmbeddings(
			model=settings.NVIDIA_EMBEDDING_MODEL,
			api_key=settings.NVIDIA_API_KEY,
			truncate="END",
		)
		self.dimension = 2048
		self.batch_size = 32  # Optimal batch size for NVIDIA API
		self.max_retries = 3
		self.retry_delay = 2  # seconds

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		"""Embed texts with caching and retry logic"""
		if not texts:
			return []
		
		# Check cache first
		embeddings = []
		uncached_texts = []
		uncached_indices = []
		
		for i, text in enumerate(texts):
			cached = embedding_cache.get(text)
			if cached:
				embeddings.append(cached)
			else:
				embeddings.append(None)
				uncached_texts.append(text)
				uncached_indices.append(i)
		
		# If all cached, return immediately
		if not uncached_texts:
			logger.debug(f"All {len(texts)} embeddings from cache")
			return embeddings
		
		# Embed uncached texts with retry logic
		logger.debug(f"Embedding {len(uncached_texts)} texts ({len(texts) - len(uncached_texts)} from cache)")
		new_embeddings = self._embed_with_retry(uncached_texts)
		
		# Cache new embeddings and insert into result
		for i, embedding in zip(uncached_indices, new_embeddings):
			embeddings[i] = embedding
			embedding_cache.set(texts[i], embedding)
		
		return embeddings

	def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
		"""Embed with exponential backoff retry"""
		for attempt in range(self.max_retries):
			try:
				embeddings = self.embedder.embed_documents(texts)
				
				# Validate dimensions
				if embeddings and len(embeddings[0]) != self.dimension:
					logger.warning(f"Unexpected embedding dimension: {len(embeddings[0])} (expected {self.dimension})")
				
				return embeddings
				
			except Exception as e:
				if attempt < self.max_retries - 1:
					wait_time = self.retry_delay * (2 ** attempt)
					logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
					time.sleep(wait_time)
				else:
					logger.error(f"Embedding failed after {self.max_retries} attempts: {e}")
					raise

	def embed_batch(self, texts: List[str]) -> List[List[float]]:
		"""Embed large batches efficiently by splitting into chunks"""
		if len(texts) <= self.batch_size:
			return self.embed_texts(texts)
		
		all_embeddings = []
		for i in range(0, len(texts), self.batch_size):
			batch = texts[i:i + self.batch_size]
			batch_embeddings = self.embed_texts(batch)
			all_embeddings.extend(batch_embeddings)
		
		logger.info(f"Batch embedded {len(texts)} texts in {(len(texts) + self.batch_size - 1) // self.batch_size} batches")
		return all_embeddings
