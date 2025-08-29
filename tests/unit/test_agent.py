"""Unit tests for agent module."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.documents import Document
from kbac.agent import MatrixAgent


class TestMatrixAgent:
    """Test cases for MatrixAgent class."""

    @patch("kbac.agent.Agent")
    def test_initialization(self, mock_agent_class):
        """Test agent initialization."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        assert agent.agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("kbac.agent.Agent")
    def test_get_system_prompt(self, mock_agent_class):
        """Test system prompt generation."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()
        prompt = agent._get_system_prompt()

        assert "Matrix movie script" in prompt
        assert "ONLY answer based on the provided context" in prompt
        assert "Do NOT use any external knowledge" in prompt
        assert "Be accurate and specific" in prompt
        assert "Cite specific parts" in prompt

    @patch("kbac.agent.Agent")
    def test_format_context(self, mock_agent_class):
        """Test context formatting."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        # Test with documents
        documents = [
            (
                Document(page_content="Test content 1", metadata={"character": "Neo"}),
                0.9,
            ),
            (
                Document(
                    page_content="Test content 2", metadata={"location": "Matrix"}
                ),
                0.8,
            ),
        ]

        formatted = agent._format_context(documents)

        assert "Context 1" in formatted
        assert "Context 2" in formatted
        assert "Test content 1" in formatted
        assert "Test content 2" in formatted
        assert "relevance: 0.900" in formatted
        assert "relevance: 0.800" in formatted
        assert "character: Neo" in formatted
        assert "location: Matrix" in formatted

    @patch("kbac.agent.Agent")
    def test_format_context_empty(self, mock_agent_class):
        """Test context formatting with empty documents."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        formatted = agent._format_context([])
        assert formatted == "No relevant context found."

    @patch("kbac.agent.Agent")
    def test_format_context_single_document(self, mock_agent_class):
        """Test context formatting with single document."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        documents = [
            (
                Document(
                    page_content="Single content", metadata={"character": "Morpheus"}
                ),
                0.95,
            )
        ]

        formatted = agent._format_context(documents)

        assert "Context 1" in formatted
        assert "Single content" in formatted
        assert "relevance: 0.950" in formatted
        assert "character: Morpheus" in formatted
        assert "Context 2" not in formatted

    @patch("kbac.agent.Agent")
    def test_format_context_complex_metadata(self, mock_agent_class):
        """Test context formatting with complex metadata."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        documents = [
            (
                Document(
                    page_content="Complex content",
                    metadata={
                        "character": "Neo",
                        "location": "Matrix",
                        "page_number": 42,
                        "text_type": "dialog",
                    },
                ),
                0.85,
            )
        ]

        formatted = agent._format_context(documents)

        assert "character: Neo" in formatted
        assert "location: Matrix" in formatted
        assert "page_number: 42" in formatted
        assert "text_type: dialog" in formatted

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_question_with_context(self, mock_agent_class):
        """Test answering question with context."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock async run method
        mock_agent.run = AsyncMock(return_value="Test answer")

        agent = MatrixAgent()

        documents = [(Document(page_content="Test content", metadata={}), 0.9)]

        answer = await agent.answer_question("Test question", documents)

        assert answer == "Test answer"
        mock_agent.run.assert_called_once()

        # Verify the call arguments
        call_args = mock_agent.run.call_args[0][0]
        assert "Test question" in call_args
        assert "Test content" in call_args
        assert "Context:" in call_args

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_question_no_context(self, mock_agent_class):
        """Test answering question without context."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock async run method
        mock_agent.run = AsyncMock(return_value="No context answer")

        agent = MatrixAgent()

        # When no context is provided, the agent should return a specific message
        answer = await agent.answer_question("Test question", [])

        # The agent should return the default message for no context
        assert (
            answer
            == "I don't have enough information from the script to answer this question."
        )
        # The agent should not be called when there's no context
        mock_agent.run.assert_not_called()

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_question_error(self, mock_agent_class):
        """Test error handling in question answering."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock error
        mock_agent.run = AsyncMock(side_effect=Exception("Test error"))

        agent = MatrixAgent()

        documents = [(Document(page_content="Test content", metadata={}), 0.9)]

        answer = await agent.answer_question("Test question", documents)

        assert "encountered an error" in answer
        assert "Test error" in answer

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_question_network_error(self, mock_agent_class):
        """Test error handling with network error."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock network error
        mock_agent.run = AsyncMock(side_effect=ConnectionError("Network timeout"))

        agent = MatrixAgent()

        documents = [(Document(page_content="Test content", metadata={}), 0.9)]

        answer = await agent.answer_question("Test question", documents)

        assert "encountered an error" in answer
        assert "Network timeout" in answer

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_with_retriever(self, mock_agent_class):
        """Test answering question with retriever."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock async run method
        mock_agent.run = AsyncMock(return_value="Retriever answer")

        agent = MatrixAgent()

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.search.return_value = [
            (Document(page_content="Retrieved content", metadata={}), 0.9)
        ]

        answer = await agent.answer_with_retriever("Test question", mock_retriever)

        assert answer == "Retriever answer"
        mock_retriever.search.assert_called_once_with("Test question")
        mock_agent.run.assert_called_once()

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_with_retriever_error(self, mock_agent_class):
        """Test answering question with retriever error."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock retriever error
        mock_retriever = Mock()
        mock_retriever.search.side_effect = Exception("Retriever error")

        agent = MatrixAgent()

        answer = await agent.answer_with_retriever("Test question", mock_retriever)

        assert "encountered an error" in answer
        assert "Retriever error" in answer

    @patch("kbac.agent.Agent")
    @pytest.mark.asyncio
    async def test_answer_with_retriever_no_results(self, mock_agent_class):
        """Test answering question with no retriever results."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        # Mock retriever with no results
        mock_retriever = Mock()
        mock_retriever.search.return_value = []

        agent = MatrixAgent()

        answer = await agent.answer_with_retriever("Test question", mock_retriever)

        assert (
            answer
            == "I don't have enough information from the script to answer this question."
        )
        mock_retriever.search.assert_called_once_with("Test question")
        mock_agent.run.assert_not_called()

    @patch("kbac.agent.Agent")
    def test_prompt_structure(self, mock_agent_class):
        """Test that the generated prompt has correct structure."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        agent = MatrixAgent()

        documents = [
            (Document(page_content="Test content", metadata={"character": "Neo"}), 0.9)
        ]

        # Get the prompt that would be sent to the agent
        formatted_context = agent._format_context(documents)
        system_prompt = agent._get_system_prompt()

        # Test prompt structure
        assert "Context 1" in formatted_context
        assert "Test content" in formatted_context
        assert "relevance: 0.900" in formatted_context
        assert "character: Neo" in formatted_context

        # Test system prompt content
        assert "Matrix movie script" in system_prompt
        assert "ONLY answer based on the provided context" in system_prompt
        assert "Do NOT use any external knowledge" in system_prompt
