import logging
from pydantic_settings import BaseSettings
from pydantic import field_validator, ValidationError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("hyperrag.settings")


class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    NVIDIA_EMBEDDING_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    NVIDIA_RERANK_MODEL: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

    # Database connections
    OPENSEARCH_HOST: str
    OPENSEARCH_USER: str
    OPENSEARCH_PASSWORD: str
    QDRANT_HOST: str
    QDRANT_API_KEY: str = ""
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str

    # LLM
    LLM_PROVIDER: str = "openai"  # openai, anthropic, nvidia
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    # NVIDIA LLM Specifics
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_LLM_MODEL: str = "moonshotai/kimi-k2-thinking"

    class Config:
        env_file = ".env"
    
    @field_validator("NVIDIA_API_KEY")
    @classmethod
    def validate_nvidia_key(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("NVIDIA_API_KEY is required for embeddings and reranking")
        return v
    
    @field_validator("OPENSEARCH_HOST", "QDRANT_HOST", "NEO4J_URI")
    @classmethod
    def validate_hosts(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Database connection strings cannot be empty")
        return v
    
    def validate_llm_config(self) -> bool:
        """Validate LLM configuration based on provider"""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            logger.warning("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set")
            return False
        elif self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            logger.warning("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set")
            return False
        elif self.LLM_PROVIDER == "nvidia" and not self.NVIDIA_API_KEY:
            logger.warning("LLM_PROVIDER is 'nvidia' but NVIDIA_API_KEY is not set")
            return False
        return True


try:
    settings = Settings()
    settings.validate_llm_config()
    logger.info("Settings validated successfully")
except ValidationError as e:
    logger.error(f"Settings validation failed: {e}")
    raise
