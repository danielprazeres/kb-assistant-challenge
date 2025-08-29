"""Unit tests for API client module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import httpx
from kbac.api_client import (
    MatrixAPIClient,
    QuestionRequest,
    QuestionResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    health_check,
    ask_question,
    chat_message,
)


class TestAPIModels:
    """Test cases for API models."""

    def test_question_request_creation(self):
        """Test QuestionRequest model creation."""
        request = QuestionRequest(
            question="What is the Matrix?",
            model="gpt-4o-mini",
            top_k=5,
            similarity_threshold=0.7,
        )

        assert request.question == "What is the Matrix?"
        assert request.model == "gpt-4o-mini"
        assert request.top_k == 5
        assert request.similarity_threshold == 0.7

    def test_question_request_optional_fields(self):
        """Test QuestionRequest with optional fields."""
        request = QuestionRequest(question="What is the Matrix?")

        assert request.question == "What is the Matrix?"
        assert request.model is None
        assert request.top_k is None
        assert request.similarity_threshold is None

    def test_question_response_creation(self):
        """Test QuestionResponse model creation."""
        response = QuestionResponse(
            answer="The Matrix is a computer-generated dream world.",
            question="What is the Matrix?",
            model_used="gpt-4o-mini",
            retrieval_stats={"top_k": 5, "similarity_threshold": 0.7},
        )

        assert response.answer == "The Matrix is a computer-generated dream world."
        assert response.question == "What is the Matrix?"
        assert response.model_used == "gpt-4o-mini"
        assert response.retrieval_stats["top_k"] == 5

    def test_chat_request_creation(self):
        """Test ChatRequest model creation."""
        request = ChatRequest(message="Hello", conversation_id="conv_123")

        assert request.message == "Hello"
        assert request.conversation_id == "conv_123"

    def test_chat_response_creation(self):
        """Test ChatResponse model creation."""
        response = ChatResponse(
            response="Hello! How can I help you?",
            conversation_id="conv_123",
            message_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
            ],
        )

        assert response.response == "Hello! How can I help you?"
        assert response.conversation_id == "conv_123"
        assert len(response.message_history) == 2

    def test_health_response_creation(self):
        """Test HealthResponse model creation."""
        response = HealthResponse(
            status="healthy", version="1.0.0", assistant_ready=True
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.assistant_ready is True


class TestMatrixAPIClient:
    """Test cases for MatrixAPIClient class."""

    @pytest.fixture
    def client(self):
        """Create API client instance."""
        return MatrixAPIClient("http://test-api.com")

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = Mock()
        response.json.return_value = {"status": "success"}
        response.raise_for_status.return_value = None
        return response

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.base_url == "http://test-api.com"
        assert client.client is None  # Client is initialized lazily

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager."""
        async with client as c:
            assert c == client
            assert c.client is not None  # Client should be initialized in context
        # Client should be closed after context exit
        assert client.client.is_closed

    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_response):
        """Test successful health check."""
        # Mock the client creation
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response

            mock_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
                "assistant_ready": True,
            }

            result = await client.health_check()

            assert isinstance(result, HealthResponse)
            assert result.status == "healthy"
            assert result.version == "1.0.0"
            assert result.assistant_ready is True

    @pytest.mark.asyncio
    async def test_health_check_http_error(self, client):
        """Test health check with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.get.side_effect = httpx.HTTPStatusError(
                "404", request=Mock(), response=Mock()
            )

            with pytest.raises(httpx.HTTPStatusError):
                await client.health_check()

    @pytest.mark.asyncio
    async def test_ask_question_success(self, client, mock_response):
        """Test successful question asking."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            mock_response.json.return_value = {
                "answer": "The Matrix is a computer-generated dream world.",
                "question": "What is the Matrix?",
                "model_used": "gpt-4o-mini",
                "retrieval_stats": {"top_k": 5},
            }

            result = await client.ask_question(
                question="What is the Matrix?",
                model="gpt-4o-mini",
                top_k=5,
                similarity_threshold=0.7,
            )

            assert isinstance(result, QuestionResponse)
            assert result.answer == "The Matrix is a computer-generated dream world."
            assert result.question == "What is the Matrix?"
            assert result.model_used == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_ask_question_503_error(self, client):
        """Test question asking with 503 error."""
        error_response = Mock()
        error_response.status_code = 503
        error_response.text = "Service unavailable"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.post.side_effect = httpx.HTTPStatusError(
                "503", request=Mock(), response=error_response
            )

            with pytest.raises(Exception) as exc_info:
                await client.ask_question("What is the Matrix?")
            assert "Assistant not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ask_question_500_error(self, client):
        """Test question asking with 500 error."""
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal server error"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.post.side_effect = httpx.HTTPStatusError(
                "500", request=Mock(), response=error_response
            )

            with pytest.raises(Exception) as exc_info:
                await client.ask_question("What is the Matrix?")
            assert "Server error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_chat_message_success(self, client, mock_response):
        """Test successful chat message."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            mock_response.json.return_value = {
                "response": "Hello! How can I help you?",
                "conversation_id": "conv_123",
                "message_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hello! How can I help you?"},
                ],
            }

            result = await client.chat_message(
                message="Hello", conversation_id="conv_123"
            )

            assert isinstance(result, ChatResponse)
            assert result.response == "Hello! How can I help you?"
            assert result.conversation_id == "conv_123"
            assert len(result.message_history) == 2

    @pytest.mark.asyncio
    async def test_get_models_success(self, client, mock_response):
        """Test successful models retrieval."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response

            mock_response.json.return_value = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

            result = await client.get_models()

            assert result == ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    @pytest.mark.asyncio
    async def test_get_config_success(self, client, mock_response):
        """Test successful config retrieval."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.get.return_value = mock_response

            mock_response.json.return_value = {
                "embedding_model": "text-embedding-3-large",
                "llm_model": "gpt-4o-mini",
            }

            result = await client.get_config()

            assert result["embedding_model"] == "text-embedding-3-large"
            assert result["llm_model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_reinitialize_success(self, client, mock_response):
        """Test successful reinitialization."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = AsyncMock()
            mock_client_class.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            mock_response.json.return_value = {"status": "reinitialized"}

            result = await client.reinitialize()

            assert result["status"] == "reinitialized"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch("kbac.api_client.MatrixAPIClient")
    def test_health_check_sync(self, mock_client_class):
        """Test synchronous health check."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.health_check = AsyncMock(
            return_value=HealthResponse(
                status="healthy", version="1.0.0", assistant_ready=True
            )
        )

        result = health_check()

        assert isinstance(result, HealthResponse)
        assert result.status == "healthy"
        mock_client.health_check.assert_called_once()

    @patch("kbac.api_client.MatrixAPIClient")
    def test_ask_question_sync(self, mock_client_class):
        """Test synchronous question asking."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.ask_question = AsyncMock(
            return_value=QuestionResponse(
                answer="Test answer",
                question="Test question",
                model_used="gpt-4o-mini",
                retrieval_stats={},
            )
        )

        result = ask_question("Test question")

        assert isinstance(result, QuestionResponse)
        assert result.answer == "Test answer"
        mock_client.ask_question.assert_called_once()

    @patch("kbac.api_client.MatrixAPIClient")
    def test_chat_message_sync(self, mock_client_class):
        """Test synchronous chat message."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat_message = AsyncMock(
            return_value=ChatResponse(
                response="Test response", conversation_id="conv_123", message_history=[]
            )
        )

        result = chat_message("Test message")

        assert isinstance(result, ChatResponse)
        assert result.response == "Test response"
        mock_client.chat_message.assert_called_once()
