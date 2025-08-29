"""Unit tests for retriever module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from kbac.retriever import MatrixRetriever
from kbac.loaders.matrix_script_loader import Document as MatrixDocument


class TestMatrixRetriever:
    """Test cases for MatrixRetriever class."""

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_initialization(self, mock_embeddings, mock_client, mock_vector_store):
        """Test retriever initialization."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock collection setup
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        assert retriever.client == mock_client_instance
        assert retriever.embeddings == mock_embeddings_instance
        assert retriever.vector_store == mock_vector_store_instance

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_initialization_collection_exists(
        self, mock_embeddings, mock_client, mock_vector_store
    ):
        """Test retriever initialization when collection already exists."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock collection already exists - mock the collections object structure
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "matrix_script"
        mock_collections.collections = [mock_collection]
        mock_client_instance.get_collections.return_value = mock_collections

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Should not create collection since it exists
        mock_client_instance.create_collection.assert_not_called()
        assert retriever.vector_store == mock_vector_store_instance

    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_initialization_error(self, mock_embeddings, mock_client):
        """Test retriever initialization with error."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock error during collection setup
        mock_client_instance.get_collections.side_effect = Exception("Connection error")

        with pytest.raises(Exception) as exc_info:
            MatrixRetriever()
        assert "Connection error" in str(exc_info.value)

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_add_documents(self, mock_embeddings, mock_client, mock_vector_store):
        """Test adding documents to vector store."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Create test documents
        test_docs = [
            MatrixDocument(
                text="Test dialog 1",
                metadata={"character": "Neo", "location": "Matrix"},
            ),
            MatrixDocument(
                text="Test dialog 2",
                metadata={"character": "Morpheus", "location": "Nebuchadnezzar"},
            ),
        ]

        # Mock vector store add_documents
        mock_vector_store_instance.add_documents.return_value = [
            Document(
                page_content="Test dialog 1",
                metadata={"character": "Neo", "location": "Matrix"},
            ),
            Document(
                page_content="Test dialog 2",
                metadata={"character": "Morpheus", "location": "Nebuchadnezzar"},
            ),
        ]

        added_docs = retriever.add_documents(test_docs)

        # Verify documents were added
        mock_vector_store_instance.add_documents.assert_called_once()
        assert len(added_docs) == 2
        assert isinstance(added_docs[0], Document)
        assert added_docs[0].page_content == "Test dialog 1"
        assert added_docs[0].metadata["character"] == "Neo"

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_add_documents_empty_list(
        self, mock_embeddings, mock_client, mock_vector_store
    ):
        """Test adding empty document list."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Should not call add_documents with empty list
        retriever.add_documents([])
        mock_vector_store_instance.add_documents.assert_not_called()

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_add_documents_error(self, mock_embeddings, mock_client, mock_vector_store):
        """Test adding documents with error."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Create test documents
        test_docs = [
            MatrixDocument(
                text="Test dialog 1",
                metadata={"character": "Neo", "location": "Matrix"},
            )
        ]

        # Mock error during add_documents
        mock_vector_store_instance.add_documents.side_effect = Exception(
            "Storage error"
        )

        with pytest.raises(Exception) as exc_info:
            retriever.add_documents(test_docs)
        assert "Storage error" in str(exc_info.value)

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_search(self, mock_embeddings, mock_client, mock_vector_store):
        """Test document search functionality."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock search results
        mock_results = [
            (Document(page_content="Result 1", metadata={}), 0.9),
            (Document(page_content="Result 2", metadata={}), 0.8),
        ]

        mock_vector_store_instance.similarity_search_with_score.return_value = (
            mock_results
        )

        results = retriever.search("test query")

        # Verify search was called
        mock_vector_store_instance.similarity_search_with_score.assert_called_once_with(
            "test query", k=5
        )
        assert len(results) == 2
        assert results[0][0].page_content == "Result 1"
        assert results[0][1] == 0.9
        assert results[1][0].page_content == "Result 2"
        assert results[1][1] == 0.8

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_search_custom_k(self, mock_embeddings, mock_client, mock_vector_store):
        """Test document search with custom k value."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock search results
        mock_results = [
            (Document(page_content="Result 1", metadata={}), 0.9),
            (Document(page_content="Result 2", metadata={}), 0.8),
        ]

        mock_vector_store_instance.similarity_search_with_score.return_value = (
            mock_results
        )

        results = retriever.search("test query", k=10)

        # Verify custom k was used
        mock_vector_store_instance.similarity_search_with_score.assert_called_once_with(
            "test query", k=10
        )
        assert len(results) == 2

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_search_no_results(self, mock_embeddings, mock_client, mock_vector_store):
        """Test document search with no results."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock empty search results
        mock_vector_store_instance.similarity_search_with_score.return_value = []

        results = retriever.search("test query")

        assert len(results) == 0

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_search_error(self, mock_embeddings, mock_client, mock_vector_store):
        """Test document search with error."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock error during search
        mock_vector_store_instance.similarity_search_with_score.side_effect = Exception(
            "Search error"
        )

        with pytest.raises(Exception) as exc_info:
            retriever.search("test query")
        assert "Search error" in str(exc_info.value)

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_clear_collection(self, mock_embeddings, mock_client, mock_vector_store):
        """Test clearing the collection."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock collection deletion
        mock_client_instance.delete_collection.return_value = None

        retriever.clear_collection()

        # Verify collection was deleted and recreated
        mock_client_instance.delete_collection.assert_called_once()

    @patch("kbac.retriever.QdrantVectorStore")
    @patch("kbac.retriever.QdrantClient")
    @patch("kbac.retriever.OpenAIEmbeddings")
    def test_clear_collection_error(
        self, mock_embeddings, mock_client, mock_vector_store
    ):
        """Test clearing the collection with error."""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        mock_client_instance.get_collections.return_value = Mock(
            collections=[Mock(name="other_collection")]
        )

        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        retriever = MatrixRetriever()

        # Mock error during deletion
        mock_client_instance.delete_collection.side_effect = Exception("Delete error")

        with pytest.raises(Exception) as exc_info:
            retriever.clear_collection()
        assert "Delete error" in str(exc_info.value)
