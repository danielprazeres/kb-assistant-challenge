"""Integration tests for Streamlit web application."""

import pytest
import asyncio
from unittest.mock import patch, Mock
import streamlit as st

from kbac.web_app import initialize_assistant, main


class TestWebApp:
    """Test cases for web application."""

    @patch("kbac.web_app.MatrixAssistant")
    def test_initialize_assistant_success(self, mock_assistant_class):
        """Test successful assistant initialization."""
        # Mock the assistant
        mock_assistant = Mock()
        mock_assistant_class.return_value = mock_assistant

        # Mock session state
        if hasattr(st, "session_state"):
            st.session_state = {}

        initialize_assistant()

        assert st.session_state.get("assistant") == mock_assistant
        assert st.session_state.get("initialized") is True
        mock_assistant.initialize.assert_called_once()

    @patch("kbac.web_app.MatrixAssistant")
    def test_initialize_assistant_failure(self, mock_assistant_class):
        """Test assistant initialization failure."""
        # Mock the assistant to raise an exception
        mock_assistant_class.side_effect = Exception("Initialization failed")

        # Mock session state
        if hasattr(st, "session_state"):
            st.session_state = {}

        initialize_assistant()

        assert st.session_state.get("initialized") is False
        assert "assistant" not in st.session_state

    @patch("kbac.web_app.initialize_assistant")
    @patch("kbac.web_app.st")
    def test_main_function_calls(self, mock_st, mock_init):
        """Test that main function calls necessary Streamlit functions."""
        # Mock session state
        mock_st.session_state = {}

        # Mock chat input to return None (no input)
        mock_st.chat_input.return_value = None

        main()

        # Verify initialization was called
        mock_init.assert_called_once()

        # Verify page config was set
        mock_st.set_page_config.assert_called_once()

        # Verify chat input was called
        mock_st.chat_input.assert_called_once()

    @patch("kbac.web_app.initialize_assistant")
    @patch("kbac.web_app.asyncio.run")
    @patch("kbac.web_app.st")
    def test_main_with_user_input(self, mock_st, mock_asyncio_run, mock_init):
        """Test main function with user input."""
        # Mock session state
        mock_st.session_state = {"initialized": True, "messages": []}

        # Mock chat input to return a question
        mock_st.chat_input.return_value = "What is the Matrix?"

        # Mock assistant
        mock_assistant = Mock()
        mock_st.session_state["assistant"] = mock_assistant

        # Mock async response
        mock_asyncio_run.return_value = (
            "The Matrix is a computer-generated dream world."
        )

        main()

        # Verify async call was made
        mock_asyncio_run.assert_called_once()

        # Verify messages were added to session state
        assert len(mock_st.session_state["messages"]) == 2  # user + assistant
        assert mock_st.session_state["messages"][0]["role"] == "user"
        assert mock_st.session_state["messages"][0]["content"] == "What is the Matrix?"
        assert mock_st.session_state["messages"][1]["role"] == "assistant"
        assert (
            mock_st.session_state["messages"][1]["content"]
            == "The Matrix is a computer-generated dream world."
        )
