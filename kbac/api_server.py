#!/usr/bin/env python3
"""FastAPI server script for Matrix Script Assistant API."""

import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kbac.api import app
from kbac.config import settings


def main():
    """Run the FastAPI server."""
    try:
        # Check if .env file exists
        if not os.path.exists(".env"):
            print(
                "⚠️  Warning: .env file not found. Please create one from env.example"
            )
            print("   cp env.example .env")
            print("   Then add your OpenAI API key to the .env file")
            print()

        print("🚀 Starting Matrix Script Assistant API...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Interactive API docs: http://localhost:8000/redoc")
        print("💚 Health check: http://localhost:8000/health")
        print("⏹️  Press Ctrl+C to stop")
        print()

        # Run the server
        uvicorn.run(
            "kbac.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info",
        )

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
