"""Integration tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import asyncio

from kbac.api import app, assistant


class TestFastAPI:
    """Test cases for FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch("kbac.api.MatrixAssistant")
    def test_root_endpoint(self, mock_assistant_class, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "🎬 Matrix Script Assistant API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data

    @patch("kbac.api.MatrixAssistant")
    def test_health_check_unhealthy(self, mock_assistant_class, client):
        """Test health check when assistant is not initialized."""
        # Mock assistant to be None
        with patch("kbac.api.assistant", None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["assistant_ready"] is False

    @patch("kbac.api.MatrixAssistant")
    def test_health_check_healthy(self, mock_assistant_class, client):
        """Test health check when assistant is initialized."""
        # Mock assistant to be available
        mock_assistant = Mock()
        with patch("kbac.api.assistant", mock_assistant):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["assistant_ready"] is True

    @patch("kbac.api.MatrixAssistant")
    def test_ask_question_success(self, mock_assistant_class, client):
        """Test successful question asking."""
        # Mock assistant
        mock_assistant = Mock()
        mock_assistant.answer_question = AsyncMock(
            return_value="The Matrix is a computer-generated dream world."
        )

        with patch("kbac.api.assistant", mock_assistant):
            response = client.post(
                "/ask", json={"question": "What is the Matrix?", "model": "gpt-4o-mini"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "The Matrix is a computer-generated dream world."
            assert data["question"] == "What is the Matrix?"
            assert "model_used" in data
            assert "retrieval_stats" in data

    @patch("kbac.api.MatrixAssistant")
    def test_ask_question_no_assistant(self, mock_assistant_class, client):
        """Test asking question when assistant is not initialized."""
        with patch("kbac.api.assistant", None):
            response = client.post("/ask", json={"question": "What is the Matrix?"})

            assert response.status_code == 503
            data = response.json()
            assert "Assistant not initialized" in data["detail"]

    @patch("kbac.api.MatrixAssistant")
    def test_ask_question_error(self, mock_assistant_class, client):
        """Test error handling in question asking."""
        # Mock assistant to raise exception
        mock_assistant = Mock()
        mock_assistant.answer_question = AsyncMock(side_effect=Exception("Test error"))

        with patch("kbac.api.assistant", mock_assistant):
            response = client.post("/ask", json={"question": "What is the Matrix?"})

            assert response.status_code == 500
            data = response.json()
            assert "Failed to answer question" in data["detail"]

    @patch("kbac.api.MatrixAssistant")
    def test_chat_message_success(self, mock_assistant_class, client):
        """Test successful chat message."""
        # Mock assistant
        mock_assistant = Mock()
        mock_assistant.answer_question = AsyncMock(
            return_value="Hello! How can I help you?"
        )

        with patch("kbac.api.assistant", mock_assistant):
            response = client.post(
                "/chat", json={"message": "Hello", "conversation_id": "test_conv"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Hello! How can I help you?"
            assert data["conversation_id"] == "test_conv"
            assert len(data["message_history"]) == 2
            assert data["message_history"][0]["role"] == "user"
            assert data["message_history"][1]["role"] == "assistant"

    @patch("kbac.api.MatrixAssistant")
    def test_get_models(self, mock_assistant_class, client):
        """Test getting available models."""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models

    @patch("kbac.api.MatrixAssistant")
    def test_get_config(self, mock_assistant_class, client):
        """Test getting configuration."""
        response = client.get("/config")
        assert response.status_code == 200
        config = response.json()
        assert "llm_model" in config
        assert "embedding_model" in config
        assert "top_k_results" in config
        assert "similarity_threshold" in config

    @patch("kbac.api.MatrixAssistant")
    def test_reinitialize(self, mock_assistant_class, client):
        """Test reinitialization endpoint."""
        response = client.post("/reinitialize")
        assert response.status_code == 200
        data = response.json()
        assert "Reinitialization started" in data["message"]

    @patch("kbac.api.MatrixAssistant")
    def test_validation_errors(self, mock_assistant_class, client):
        """Test input validation."""
        # Test empty question
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

        # Test question too long
        long_question = "a" * 1001
        response = client.post("/ask", json={"question": long_question})
        assert response.status_code == 422

        # Test invalid top_k
        response = client.post(
            "/ask", json={"question": "What is the Matrix?", "top_k": 0}
        )
        assert response.status_code == 422

        # Test invalid similarity threshold
        response = client.post(
            "/ask",
            json={"question": "What is the Matrix?", "similarity_threshold": 1.5},
        )
        assert response.status_code == 422
