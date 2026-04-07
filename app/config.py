from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Vector Database
    qdrant_url: str = "http://localhost:6333"

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"

    # RAG Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
