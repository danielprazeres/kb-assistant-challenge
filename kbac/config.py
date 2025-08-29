"""Configuration module for KB Assistant Challenge."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field(
        default="text-embedding-3-large", env="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=256, env="EMBEDDING_DIMENSIONS")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(
        default="matrix_script", env="QDRANT_COLLECTION_NAME"
    )

    # Application Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    script_path: str = Field(
        default="resources/movie-scripts/the-matrix-1999.pdf", env="SCRIPT_PATH"
    )
    data_dir: str = Field(default="data", env="DATA_DIR")

    # Retrieval Configuration
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    @validator("openai_api_key")
    def validate_openai_api_key(cls, v):
        """Validate OpenAI API key is not empty."""
        if not v or not v.strip():
            raise ValueError("OpenAI API key cannot be empty")
        return v

    @validator("embedding_dimensions")
    def validate_embedding_dimensions(cls, v):
        """Validate embedding dimensions is positive."""
        if v <= 0:
            raise ValueError("Embedding dimensions must be positive")
        return v

    @validator("qdrant_port")
    def validate_qdrant_port(cls, v):
        """Validate Qdrant port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("Qdrant port must be between 1 and 65535")
        return v

    @validator("similarity_threshold")
    def validate_similarity_threshold(cls, v):
        """Validate similarity threshold is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v

    @validator("top_k_results")
    def validate_top_k_results(cls, v):
        """Validate top_k_results is positive."""
        if v <= 0:
            raise ValueError("Top K results must be positive")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance - lazy loading
_settings_instance = None


def get_settings() -> Settings:
    """Get the global settings instance with lazy loading."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# For backward compatibility
settings = get_settings()
