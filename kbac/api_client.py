"""API client for communicating with the Matrix Script Assistant API."""

import httpx
import asyncio
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str
    model: Optional[str] = None
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response model for question answers."""

    answer: str
    question: str
    model_used: str
    retrieval_stats: Dict[str, Any]


class ChatRequest(BaseModel):
    """Request model for chat conversations."""

    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat conversations."""

    response: str
    conversation_id: str
    message_history: List[Dict[str, str]]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    assistant_ready: bool


class MatrixAPIClient:
    """Client for communicating with the Matrix Script Assistant API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url.rstrip("/")
        self.client = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()

    async def health_check(self) -> HealthResponse:
        """Check API health."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return HealthResponse(**response.json())
        except httpx.HTTPStatusError as e:
            logger.error(f"Health check failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Health check error: {e}")
            raise

    async def ask_question(
        self,
        question: str,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> QuestionResponse:
        """Ask a question via the API."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            request_data = QuestionRequest(
                question=question,
                model=model,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            response = await self.client.post(
                f"{self.base_url}/ask", json=request_data.model_dump(exclude_none=True)
            )
            response.raise_for_status()

            return QuestionResponse(**response.json())

        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed: {e}")
            if e.response.status_code == 503:
                raise Exception("Assistant not initialized. Please try again later.")
            elif e.response.status_code == 500:
                raise Exception("Server error. Please try again.")
            else:
                raise Exception(f"API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Question request error: {e}")
            raise

    async def chat_message(
        self, message: str, conversation_id: Optional[str] = None
    ) -> ChatResponse:
        """Send a chat message via the API."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            request_data = ChatRequest(message=message, conversation_id=conversation_id)

            response = await self.client.post(
                f"{self.base_url}/chat", json=request_data.model_dump(exclude_none=True)
            )
            response.raise_for_status()

            return ChatResponse(**response.json())

        except httpx.HTTPStatusError as e:
            logger.error(f"Chat request failed: {e}")
            if e.response.status_code == 503:
                raise Exception("Assistant not initialized. Please try again later.")
            elif e.response.status_code == 500:
                raise Exception("Server error. Please try again.")
            else:
                raise Exception(f"API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Chat request error: {e}")
            raise

    async def get_models(self) -> List[str]:
        """Get available models."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self.client.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get models error: {e}")
            raise

    async def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self.client.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get config error: {e}")
            raise

    async def reinitialize(self) -> Dict[str, str]:
        """Reinitialize the assistant."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self.client.post(f"{self.base_url}/reinitialize")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Reinitialize error: {e}")
            raise


# Convenience functions for synchronous usage
def health_check(base_url: str = "http://localhost:8000") -> HealthResponse:
    """Synchronous health check."""

    async def _health():
        client = MatrixAPIClient(base_url)
        return await client.health_check()

    return asyncio.run(_health())


def ask_question(
    question: str,
    model: Optional[str] = None,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    base_url: str = "http://localhost:8000",
) -> QuestionResponse:
    """Synchronous question asking."""

    async def _ask():
        client = MatrixAPIClient(base_url)
        return await client.ask_question(question, model, top_k, similarity_threshold)

    return asyncio.run(_ask())


def chat_message(
    message: str,
    conversation_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
) -> ChatResponse:
    """Synchronous chat message."""

    async def _chat():
        client = MatrixAPIClient(base_url)
        return await client.chat_message(message, conversation_id)

    return asyncio.run(_chat())
