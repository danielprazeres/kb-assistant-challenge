"""Streamlit web application for Matrix Script Assistant."""

import streamlit as st
import asyncio
import logging
from pathlib import Path

from kbac.api_client import MatrixAPIClient, health_check
from kbac.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_api_health():
    """Check if the API is healthy and ready."""
    try:
        health = health_check()
        return health.assistant_ready, health.status
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False, "unhealthy"


def initialize_api_client():
    """Initialize the API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = MatrixAPIClient()


async def get_models():
    """Get available models from API."""
    try:
        async with MatrixAPIClient() as client:
            return await client.get_models()
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]  # Fallback


async def get_config():
    """Get current configuration from API."""
    try:
        async with MatrixAPIClient() as client:
            return await client.get_config()
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return {}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Matrix Script Assistant",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #00ff00;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #cccccc;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .api-status.healthy {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
    }
    .api-status.unhealthy {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff0000;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        '<h1 class="main-header">🎬 Matrix Script Assistant</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Ask questions about The Matrix movie script and get AI-powered answers</p>',
        unsafe_allow_html=True,
    )

    # Check API health
    api_ready, api_status = check_api_health()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Status
        st.subheader("🔌 API Status")
        status_class = "healthy" if api_ready else "unhealthy"
        status_icon = "✅" if api_ready else "❌"
        st.markdown(
            f"""
        <div class="api-status {status_class}">
            {status_icon} API: {api_status.upper()}
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not api_ready:
            st.error("API is not ready. Please ensure the FastAPI server is running.")
            st.info("Run: `make api-run` or `docker-compose up kbac-api`")
            return

        # Use default model (hidden from user)
        model = "gpt-4o-mini"  # Default model

        # Use default values for retrieval settings (hidden from user)
        top_k = 5  # Default value
        threshold = 0.7  # Default value

        # Configuration info
        st.subheader("📊 Configuration")
        try:
            config = asyncio.run(get_config())
            if config:
                st.json(config)
        except Exception:
            st.info("Configuration not available")

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            if "messages" in st.session_state:
                st.session_state.messages = []
            st.rerun()

        # Reinitialize button
        if st.button("🔄 Reinitialize Assistant"):
            try:

                async def reinit():
                    async with MatrixAPIClient() as client:
                        await client.reinitialize()

                asyncio.run(reinit())
                st.success("Assistant reinitialized!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reinitialize: {e}")

    # Initialize API client
    initialize_api_client()

    # Main chat interface
    if api_ready:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask about The Matrix script..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from API
                        async def get_response():
                            async with MatrixAPIClient() as client:
                                response = await client.ask_question(
                                    question=prompt,
                                    model=model,
                                    top_k=top_k,
                                    similarity_threshold=threshold,
                                )
                                return response.answer

                        response = asyncio.run(get_response())

                        # Display response
                        response_placeholder = st.empty()
                        response_placeholder.markdown(response)

                        # Add assistant response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                        logger.error(f"Error generating response: {e}")

    else:
        # Show API connection error
        st.error("Cannot connect to the Matrix Script Assistant API.")

        st.markdown(
            """
        ### To fix this issue:
        
        1. **Start the API server:**
           ```bash
           make api-run
           ```
        
        2. **Or use Docker:**
           ```bash
           docker-compose up kbac-api
           ```
        
        3. **Check the API is running:**
           - Visit: http://localhost:8000/health
           - Or: http://localhost:8000/docs
        """
        )

        with st.expander("Debug Information"):
            st.write("API Configuration:")
            st.json(
                {
                    "api_url": "http://localhost:8000",
                    "health_endpoint": "http://localhost:8000/health",
                    "docs_endpoint": "http://localhost:8000/docs",
                }
            )


if __name__ == "__main__":
    main()
