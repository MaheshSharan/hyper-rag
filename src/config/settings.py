from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    NVIDIA_EMBEDDING_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    NVIDIA_RERANK_MODEL: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

    # Database connections...
    OPENSEARCH_HOST: str
    OPENSEARCH_USER: str
    OPENSEARCH_PASSWORD: str
    QDRANT_HOST: str
    QDRANT_API_KEY: str
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

settings = Settings()
