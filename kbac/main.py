"""Main application module for KB Assistant Challenge."""

import asyncio
import logging
import os
from pathlib import Path

from .config import settings
from .loaders.matrix_script_loader import MatrixScriptLoader
from .retriever import MatrixRetriever
from .agent import MatrixAgent

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MatrixAssistant:
    """Main assistant class that orchestrates the RAG pipeline."""

    def __init__(self):
        """Initialize the Matrix assistant with all components."""
        self.loader = MatrixScriptLoader(source_path=settings.script_path)
        self.retriever = MatrixRetriever()
        self.agent = MatrixAgent()
        self.documents = None

    def load_script(self) -> None:
        """Load and process the Matrix script."""
        try:
            logger.info("Loading Matrix script...")
            self.documents = self.loader.load()
            logger.info(f"Loaded {len(self.documents)} documents from script")
        except Exception as e:
            logger.error(f"Failed to load script: {e}")
            raise

    def index_documents(self) -> None:
        """Index documents in the vector database."""
        if not self.documents:
            logger.error("No documents loaded. Call load_script() first.")
            return

        # Check if collection is already populated
        try:
            if not self.retriever.is_collection_empty():
                logger.info("Documents already indexed, skipping indexing")
                return
        except Exception as e:
            logger.warning(f"Could not check collection status: {e}. Proceeding with indexing.")

        try:
            logger.info("Indexing documents in vector database...")
            self.retriever.add_documents(self.documents)
            logger.info("Documents indexed successfully")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    async def answer_question(self, question: str) -> str:
        """Answer a question about The Matrix script."""
        # Check if retriever is ready (either documents loaded or collection has data)
        try:
            if not self.documents and self.retriever.is_collection_empty():
                logger.error("No documents available. System not initialized.")
                return "System not initialized. Please load the script first."
        except Exception as e:
            logger.warning(f"Could not check system status: {e}")

        try:
            logger.info(f"Processing question: {question}")
            answer = await self.agent.answer_with_retriever(question, self.retriever)
            return answer
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def initialize(self) -> None:
        """Initialize the assistant by loading and indexing the script."""
        # Check if we need to load and index documents
        try:
            if not self.retriever.is_collection_empty():
                logger.info("Collection already has documents, skipping initialization")
                return
        except Exception as e:
            logger.warning(f"Could not check collection status: {e}. Proceeding with full initialization.")
        
        # Only load and index if collection is empty
        self.load_script()
        self.index_documents()


async def main():
    """Main entry point for the application."""
    try:
        # Initialize assistant
        assistant = MatrixAssistant()
        assistant.initialize()

        # Interactive question answering
        print("\n=== Matrix Script Assistant ===")
        print("Ask questions about The Matrix script. Type 'quit' to exit.\n")

        while True:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            print("Thinking...")
            answer = await assistant.answer_question(question)
            print(f"\nAnswer: {answer}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
