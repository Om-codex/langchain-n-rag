import os

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------
# Project Configuration
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application configuration.

    Values come from environment variables or the project-root ``.env`` file.
    Field names are lower-case; pydantic-settings matches them case-insensitively
    (for example ``LOG_LEVEL`` -> ``log_level``).
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------
    # LLM Configuration
    # -------------------------------------

    groq_api_key: SecretStr | None = None

    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 1024

    # -------------------------------------
    # Embedding Configuration
    # -------------------------------------

    # Google Gemini Embedding API configuration.
    # Replaces local sentence-transformers inference for low-memory deployment.
    gemini_api_key: SecretStr | None = None
    gemini_embedding_model: str = "gemini-embedding-2"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 100
    embedding_max_retries: int = 6
    embedding_initial_backoff_seconds: float = 3.0

    # Normalize embeddings (L2) after retrieval from the API.
    normalize_embeddings: bool = True

    # -------------------------------------
    # ChromaDB Configuration (kept for rollback)
    # -------------------------------------

    chroma_collection_name: str = "hybrid_memory"
    chroma_persist_directory: Path = BASE_DIR / "data" / "chroma_db"

    # -------------------------------------
    # Qdrant Cloud Configuration
    # -------------------------------------

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "hybrid_memory_gemini"
    qdrant_distance: str = "Cosine"
    qdrant_upsert_batch_size: int = 100
    qdrant_timeout_seconds: float = 30.0


    # -------------------------------------
    # Database Configuration
    # -------------------------------------

    # Use the environment variable if it exists, otherwise fall back to local SQLite for testing
    database_url: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{(BASE_DIR / 'data' / 'users.db').as_posix()}"
    )

# -----------------------------------------
# Global Settings Instance
# -----------------------------------------

settings = Settings()
