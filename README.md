# KB Assistant Challenge

## Objective

Your task is to build a system that enables users to query contextual information from the script of the movie The Matrix. The system should retrieve relevant excerpts from the script and use them to help an AI agent generate accurate, context-aware responses.

You may use a Retrieval-Augmented Generation (RAG) approach, or any alternative design that effectively combines retrieval and generation to produce grounded answers. The focus is on building a solution that demonstrates strong retrieval capabilities and uses that context effectively in AI-driven responses.

## Features

- **RAG Pipeline**: Complete retrieval-augmented generation system
- **Qdrant Vector Database**: Fast and efficient vector search
- **OpenAI Integration**: State-of-the-art language models
- **Web Interface**: Beautiful Streamlit chat interface
- **REST API**: FastAPI backend with automatic Swagger documentation
- **CLI Interface**: Command-line tool for scripting
- **Docker Support**: Containerized deployment
- **Unit Tests**: Comprehensive test coverage
- **Environment Configuration**: Flexible configuration management

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd kb-assistant-challenge

# Copy environment file
cp env.example .env

# Edit .env with your OpenAI API key
# OPENAI_API_KEY=your-openai-api-key-here
```

### 2. Run with Docker (Recommended)

```bash
# Build and run all services
make docker-build
make docker-run

# Access the web interface
open http://localhost:8501

# Access the API documentation
open http://localhost:8000/docs

# Stop services
make docker-stop
```

### 3. Run Locally

```bash
# Install dependencies
make install

# Run web interface
make streamlit-run

# Run API server
make api-run

# Or run CLI
python -m kbac.cli

# Or run interactive mode
python -m kbac.main
```

## Challenge Goals

This challenge is divided into two parts:

-   Part 1 (Mandatory): Completing this part is required for your submission to be considered complete.
-   Part 2 (Optional but Recommended): This part is not required but will demonstrate deeper reasoning, richer retrieval, and more advanced capabilities.

### Part 1 - Core Functionality (Mandatory)

Your system must be able to answer basic factual queries based on the provided script of **The Matrix**.

Example queries:

-   Under what circumstances does Neo see a white rabbit?
-   How did Trinity and Neo first meet?
-   Why is there no sunlight in the future?
-   Who needs solar power to survive?
-   Why do the Agents want to capture Morpheus?
-   Describe the Nebuchadnezzar.
-   What is Nebuchadnezzar's crew made up of?

### Part 2 - Advanced Capabilities (Optional)

This part evaluates your system's ability to handle complex, composed, and reasoning-based queries.

Example queries:

-   How many times does Morpheus mention that Neo is the One?
-   Why are humans similar to a virus? And who says that?
-   Describe Cypher's personality.
-   What does Cypher offer to the Agents, and in exchange for what?
-   What is the purpose of the human fields, and who created them?

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   REST API      │    │   CLI Interface │
│   (Streamlit)   │    │   (FastAPI)     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Matrix Agent   │
                    │                 │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Retriever     │
                    │   (Qdrant)      │
                    └─────────────────┘
```

## Project Structure

```
kb-assistant-challenge/
├── kbac/                    # Main application package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── main.py             # Main application logic
│   ├── cli.py              # Command-line interface
│   ├── web_app.py          # Streamlit web interface
│   ├── api.py              # FastAPI REST API
│   ├── api_server.py       # FastAPI server script
│   ├── agent.py            # AI agent implementation
│   ├── retriever.py        # Qdrant vector retriever
│   └── loaders/            # Document loaders
│       ├── __init__.py
│       └── matrix_script_loader.py
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── resources/             # Data files
│   └── movie-scripts/
│       └── the-matrix-1999.pdf
├── notebooks/             # Jupyter notebooks
├── .streamlit/           # Streamlit configuration
├── docker-compose.yml    # Docker services
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
├── requirements-dev.txt # Development dependencies
├── Makefile            # Build and run commands
└── README.md           # This file
```

## Configuration

The application uses environment variables for configuration. Copy `env.example` to `.env` and customize:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-large
LLM_MODEL=gpt-4o-mini

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=matrix_script

# Application Settings
LOG_LEVEL=INFO
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Clean up
make clean
```

### Docker Development

```bash
# Build containers
make docker-build

# Run services
make docker-run

# View logs
docker-compose logs -f

# Stop services
make docker-stop
```

## API Usage

### REST API

The FastAPI backend provides a complete REST API with automatic Swagger documentation.

#### **API Endpoints**

- **`GET /`** - API information
- **`GET /health`** - Health check
- **`POST /ask`** - Ask a single question
- **`POST /chat`** - Start or continue a conversation
- **`GET /models`** - List available models
- **`GET /config`** - Get current configuration
- **`POST /reinitialize`** - Reinitialize the assistant

#### **Interactive Documentation**

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### **Example API Usage**

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Matrix?", "model": "gpt-4o-mini"}'

# Start a chat
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "conversation_id": "my_chat"}'
```

### Web Interface

Access the Streamlit web interface at `http://localhost:8501` for an interactive chat experience.

### CLI Usage

```bash
# Interactive mode
python -m kbac.cli

# Single question mode
python -m kbac.cli --question "What is the Matrix?"

# Verbose mode
python -m kbac.cli --verbose
```

### Programmatic Usage

```python
from kbac.main import MatrixAssistant
import asyncio

# Initialize assistant
assistant = MatrixAssistant()
assistant.initialize()

# Ask questions
async def ask_question():
    answer = await assistant.answer_question("What is the Matrix?")
    print(answer)

asyncio.run(ask_question())
```

## Recommendations (Optional)

To help you get started, here are some suggestions and prebuilt components available in this project. These are not required but may help you complete the challenge more efficiently and effectively:

-   **Script Loader**

    A custom loader is provided to parse and load The Matrix script. You can use this loader as-is or modify it as needed.

    See: [notebooks/01-loaders/01-matrix-script-loader.ipynb](notebooks/01-loaders/01-matrix-script-loader.ipynb)

-   **Retriever**

    We recommend using a **Qdrant-based** retriever.

    See: [notebooks/02-retriever/01-qdrant-retriever.ipynb](notebooks/02-retriever/01-qdrant-retriever.ipynb)

-   **LLM Agent**

    For implementing the AI agent, we recommend using **Pydantic-AI**.

    See: [notebooks/03-llm-agents/01-llm-agents.ipynb](notebooks/03-llm-agents/01-llm-agents.ipynb)

-   **Advanced Capability - MCP Server**

    To handle advanced reasoning and agent orchestration, especially for the requirements in **Part 2** of this challenge, we recommend using an **MCP server**. The environment already includes **Pydantic-AI**, which has built-in support for the MCP protocol.

    See: [https://ai.pydantic.dev/mcp/](https://ai.pydantic.dev/mcp/)

#### System Evaluation

As part of building a robust system, you should carefully consider:

-   **How will you evaluate the system?**

    What metrics or criteria will you use to assess the quality and accuracy of the responses?

-   **How will you ensure the agent does not hallucinate or rely on prior knowledge of the movie?**

    Your system should be designed to **only answer based on the retrieved context**, not the agent's pretrained knowledge.

## Environment Setup

_This setup is highly recommended but not obligatory. Work on the challenge using the environment of your preference._

### Prerequisites

-   Install Make:

    ```bash
    sudo apt install make
    ```

-   Install Docker following the official [Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).

### Dev Container (Recommended)

To ensure a consistent development environment, this project uses a preconfigured Dev Container.

-   Open this repository in VS Code:
    ```bash
    code .
    ```
-   After installing the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, press F1 to open the Command Palette, type _Dev Containers_, and select: **Reopen in Container**

### Jupyter

Jupyter is preconfigured inside the Dev Container.
You can explore examples in the [notebooks/](notebooks/) directory.
When opening a notebook, select the appropriate kernel in the top-right corner: **Python Environments -> Python 3.12 (Global Env)**

### Custom Python Library

A local Python package named **kbac** (short for KB Assistant Challenge) is included in the environment. It contains utility functions to help you work with the project. Example usage can be found in: [notebooks/01-loaders/01-matrix-script-loader.ipynb](notebooks/01-loaders/01-matrix-script-loader.ipynb). After you add to or modify this library, it is not necessary to rebuild the container. However, if you are using it in a Jupyter notebook, you should restart that notebook.

### Python Dependencies

You can install additional Python libraries by adding them to the **requirements.txt**. You should rebuild the container afterward (F1 + Rebuild Container).

### Environment Variables

You can define environment variables (such as `OPENAI_API_KEY`) in a `.env` file placed at the root of the project. These variables will be automatically loaded into the environment inside the Dev Container.

**Example `.env` file:**

```env
OPENAI_API_KEY=your-key-here
MY_CUSTOM_VAR=some-value
```
