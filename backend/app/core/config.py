"""
Application configuration using pydantic-settings.
All config is loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""

    # AI / LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Redis (optional — caching layer for embeddings and classifications)
    redis_url: str = ""

    # App Config
    confidence_threshold: float = 0.7
    rag_top_k: int = 5
    max_question_length: int = 500
    rate_limit_per_minute: int = 30

    # Environment
    env: str = "development"
    debug: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once, reused everywhere."""
    return Settings()
