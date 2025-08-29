"""Command-line interface for Matrix Script Assistant."""

import asyncio
import argparse
import logging
from pathlib import Path

from .main import MatrixAssistant

logger = logging.getLogger(__name__)


async def interactive_mode():
    """Run the assistant in interactive mode."""
    assistant = MatrixAssistant()
    assistant.initialize()

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


async def single_question_mode(question: str):
    """Answer a single question and exit."""
    assistant = MatrixAssistant()
    assistant.initialize()

    print(f"Question: {question}")
    print("Thinking...")
    answer = await assistant.answer_question(question)
    print(f"\nAnswer: {answer}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Matrix Script Assistant")
    parser.add_argument(
        "--question", "-q", type=str, help="Ask a single question and exit"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        if args.question:
            asyncio.run(single_question_mode(args.question))
        else:
            asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        raise


if __name__ == "__main__":
    main()
