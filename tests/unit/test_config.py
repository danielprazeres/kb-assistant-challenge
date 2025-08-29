"""Unit tests for configuration module."""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError
from kbac.config import Settings


class TestSettings:
    """Test cases for Settings class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()

            assert settings.openai_api_key == "test-key"
            assert settings.embedding_model == "text-embedding-3-large"
            assert settings.embedding_dimensions == 256
            assert settings.llm_model == "gpt-4o-mini"
            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333
            assert settings.qdrant_collection_name == "matrix_script"
            assert settings.log_level == "INFO"
            assert settings.top_k_results == 5
            assert settings.similarity_threshold == 0.7

    def test_environment_override(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "EMBEDDING_MODEL": "text-embedding-ada-002",
            "EMBEDDING_DIMENSIONS": "512",
            "LLM_MODEL": "gpt-4",
            "QDRANT_HOST": "custom-host",
            "QDRANT_PORT": "6334",
            "LOG_LEVEL": "DEBUG",
            "TOP_K_RESULTS": "10",
            "SIMILARITY_THRESHOLD": "0.8",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()

            assert settings.embedding_model == "text-embedding-ada-002"
            assert settings.embedding_dimensions == 512
            assert settings.llm_model == "gpt-4"
            assert settings.qdrant_host == "custom-host"
            assert settings.qdrant_port == 6334
            assert settings.log_level == "DEBUG"
            assert settings.top_k_results == 10
            assert settings.similarity_threshold == 0.8

    def test_missing_required_key(self):
        """Test that missing OpenAI API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "openai_api_key" in str(exc_info.value)

    def test_invalid_port_number(self):
        """Test that invalid port number raises validation error."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_PORT": "invalid"}
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "qdrant_port" in str(exc_info.value)

    def test_invalid_embedding_dimensions(self):
        """Test that invalid embedding dimensions raises validation error."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key", "EMBEDDING_DIMENSIONS": "-1"}
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "embedding_dimensions" in str(exc_info.value)

    def test_invalid_similarity_threshold(self):
        """Test that invalid similarity threshold raises validation error."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "SIMILARITY_THRESHOLD": "1.5",  # Should be <= 1.0
            },
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "similarity_threshold" in str(exc_info.value)

    def test_invalid_top_k_results(self):
        """Test that invalid top_k_results raises validation error."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "TOP_K_RESULTS": "0"},  # Should be > 0
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "top_k_results" in str(exc_info.value)

    def test_empty_api_key(self):
        """Test that empty API key raises validation error."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "openai_api_key" in str(exc_info.value)

    def test_case_insensitive_environment_vars(self):
        """Test that environment variables are case insensitive."""
        env_vars = {
            "openai_api_key": "test-key",  # lowercase
            "LLM_MODEL": "gpt-4o-mini",
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.openai_api_key == "test-key"
            assert settings.llm_model == "gpt-4o-mini"

    def test_script_path_default(self):
        """Test that script path has correct default value."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.script_path == "resources/movie-scripts/the-matrix-1999.pdf"

    def test_data_dir_default(self):
        """Test that data directory has correct default value."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            settings = Settings()
            assert settings.data_dir == "data"
