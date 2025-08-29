"""FastAPI application for Matrix Script Assistant API."""

import asyncio
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from .main import MatrixAssistant
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str = Field(
        ...,
        description="The question about The Matrix script",
        min_length=1,
        max_length=1000,
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)",
    )
    top_k: Optional[int] = Field(
        default=None, description="Number of top results to retrieve", ge=1, le=20
    )
    similarity_threshold: Optional[float] = Field(
        default=None, description="Similarity threshold for retrieval", ge=0.0, le=1.0
    )


class QuestionResponse(BaseModel):
    """Response model for question answers."""

    answer: str = Field(..., description="The AI-generated answer")
    question: str = Field(..., description="The original question")
    model_used: str = Field(..., description="The LLM model used")
    retrieval_stats: dict = Field(..., description="Retrieval statistics")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    assistant_ready: bool = Field(
        ..., description="Whether the assistant is initialized"
    )


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request model for chat conversations."""

    message: str = Field(..., description="User message", min_length=1, max_length=1000)
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )


class ChatResponse(BaseModel):
    """Response model for chat conversations."""

    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    message_history: List[ChatMessage] = Field(..., description="Message history")


# Global assistant instance
assistant: Optional[MatrixAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global assistant

    # Startup
    logger.info("Starting Matrix Script Assistant API...")
    try:
        assistant = MatrixAssistant()
        assistant.initialize()
        logger.info("Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        assistant = None

    yield

    # Shutdown
    logger.info("Shutting down Matrix Script Assistant API...")


# Create FastAPI app
app = FastAPI(
    title="Matrix Script Assistant API",
    description="""
    🎬 **Matrix Script Assistant API**
    
    A powerful API for querying The Matrix movie script using AI-powered retrieval and generation.
    
    ## Features
    
    - **RAG Pipeline**: Retrieval-augmented generation for accurate answers
    - **Vector Search**: Fast semantic search using Qdrant
    - **Multiple Models**: Support for GPT-4o-mini, GPT-4o, and GPT-3.5-turbo
    - **Conversation Support**: Maintain chat context across messages
    - **Configurable Retrieval**: Adjustable similarity thresholds and result counts
    
    ## Quick Start
    
    1. **Health Check**: `GET /health` - Check if the service is ready
    2. **Ask a Question**: `POST /ask` - Get a single answer
    3. **Start a Chat**: `POST /chat` - Begin a conversation
    
    ## Example Questions
    
    - "What is the Matrix?"
    - "How did Neo and Trinity first meet?"
    - "Describe the Nebuchadnezzar"
    - "Why do the Agents want to capture Morpheus?"
    """,
    version="1.0.0",
    contact={
        "name": "Matrix Script Assistant",
        "url": "https://github.com/your-repo/kb-assistant-challenge",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information.

    Returns basic information about the Matrix Script Assistant API.
    """
    return {
        "message": "🎬 Matrix Script Assistant API",
        "version": "1.0.0",
        "description": "AI-powered assistant for The Matrix movie script",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Check if the service is running and the assistant is properly initialized.
    """
    return HealthResponse(
        status="healthy" if assistant is not None else "unhealthy",
        version="1.0.0",
        assistant_ready=assistant is not None,
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a single question about The Matrix script.

    This endpoint provides a direct question-answer interface without maintaining conversation context.
    Perfect for one-off questions or integration with other systems.

    - **question**: Your question about The Matrix script
    - **model**: Optional LLM model override
    - **top_k**: Optional number of retrieval results
    - **similarity_threshold**: Optional similarity threshold
    """
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")

    try:
        # Update settings if provided
        original_model = settings.llm_model
        original_top_k = settings.top_k_results
        original_threshold = settings.similarity_threshold

        if request.model:
            settings.llm_model = request.model
        if request.top_k:
            settings.top_k_results = request.top_k
        if request.similarity_threshold:
            settings.similarity_threshold = request.similarity_threshold

        # Get answer
        answer = await assistant.answer_question(request.question)

        # Restore original settings
        settings.llm_model = original_model
        settings.top_k_results = original_top_k
        settings.similarity_threshold = original_threshold

        # Get retrieval stats (this would need to be implemented in the retriever)
        retrieval_stats = {
            "model_used": settings.llm_model,
            "top_k": settings.top_k_results,
            "similarity_threshold": settings.similarity_threshold,
        }

        return QuestionResponse(
            answer=answer,
            question=request.question,
            model_used=settings.llm_model,
            retrieval_stats=retrieval_stats,
        )

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to answer question: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    Send a chat message and get a response.

    This endpoint maintains conversation context across multiple messages.
    Provide a conversation_id to continue an existing conversation, or omit it to start a new one.

    - **message**: Your message
    - **conversation_id**: Optional conversation ID for context
    """
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant not initialized")

    try:
        # For now, we'll implement a simple version
        # In a full implementation, you'd store conversation history in a database
        conversation_id = (
            request.conversation_id or f"conv_{hash(request.message) % 10000}"
        )

        # Get response
        response = await assistant.answer_question(request.message)

        # Create message history (simplified)
        message_history = [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=response),
        ]

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            message_history=message_history,
        )

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/models", response_model=List[str])
async def get_available_models():
    """
    Get list of available LLM models.

    Returns the list of supported language models that can be used for question answering.
    """
    return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


@app.get("/config", response_model=dict)
async def get_config():
    """
    Get current configuration.

    Returns the current configuration settings for the assistant.
    """
    return {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "top_k_results": settings.top_k_results,
        "similarity_threshold": settings.similarity_threshold,
        "qdrant_host": settings.qdrant_host,
        "qdrant_port": settings.qdrant_port,
    }


@app.post("/reinitialize")
async def reinitialize_assistant(background_tasks: BackgroundTasks):
    """
    Reinitialize the assistant.

    This endpoint allows you to reinitialize the assistant, which can be useful
    if there were connection issues or configuration changes.
    """
    global assistant

    def _reinitialize():
        global assistant
        try:
            assistant = MatrixAssistant()
            assistant.initialize()
            logger.info("Assistant reinitialized successfully")
        except Exception as e:
            logger.error(f"Failed to reinitialize assistant: {e}")
            assistant = None

    background_tasks.add_task(_reinitialize)

    return {"message": "Reinitialization started in background"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
