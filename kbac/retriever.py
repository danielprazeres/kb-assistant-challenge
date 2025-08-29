"""Qdrant-based retriever for Matrix script documents."""

import logging
from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .config import settings
from .loaders.matrix_script_loader import Document as MatrixDocument

logger = logging.getLogger(__name__)


class MatrixRetriever:
    """Retriever for Matrix script documents using Qdrant vector database."""

    def __init__(self):
        """Initialize the retriever with Qdrant client and embeddings."""
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            openai_api_key=settings.openai_api_key,
        )

        self.vector_store = None
        self._setup_collection()

    def _setup_collection(self):
        """Set up Qdrant collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if settings.qdrant_collection_name not in collection_names:
                logger.info(f"Creating collection: {settings.qdrant_collection_name}")
                self.client.create_collection(
                    collection_name=settings.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=settings.qdrant_collection_name,
                embedding=self.embeddings,
            )

        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise

    def add_documents(self, documents: List[MatrixDocument]) -> List[Document]:
        """Add Matrix script documents to the vector store."""
        if not documents:
            logger.warning("No documents to add")
            return []

        # Convert Matrix documents to LangChain documents
        langchain_docs = []
        for doc in documents:
            langchain_doc = Document(page_content=doc.text, metadata=doc.metadata)
            langchain_docs.append(langchain_doc)

        try:
            self.vector_store.add_documents(langchain_docs)
            logger.info(f"Added {len(documents)} documents to vector store")
            return langchain_docs
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Search for relevant documents based on query."""
        if k is None:
            k = settings.top_k_results

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Filter by similarity threshold
            filtered_results = [
                (doc, score)
                for doc, score in results
                if score >= settings.similarity_threshold
            ]

            logger.info(
                f"Found {len(filtered_results)} relevant documents for query: {query}"
            )
            return filtered_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(settings.qdrant_collection_name)
            self._setup_collection()
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
    def is_collection_empty(self) -> bool:
        """Check if the collection is empty (no documents indexed)."""
        try:
            # Get collection info to check point count
            info = self.client.get_collection(settings.qdrant_collection_name)
            return info.points_count == 0
        except Exception as e:
            logger.error(f"Failed to check collection status: {e}")
            # If we can't check, assume empty to trigger re-indexing
            return True
