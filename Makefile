.PHONY: help install test test-unit test-integration lint format clean docker-build docker-run docker-stop streamlit-run api-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code with black"
	@echo "  clean          - Clean up generated files"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run application with Docker Compose"
	@echo "  docker-stop    - Stop Docker containers"
	@echo "  streamlit-run  - Run Streamlit web app locally"
	@echo "  api-run        - Run FastAPI server locally"

# Development commands
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 kbac/ tests/
	mypy kbac/

format:
	black kbac/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Streamlit commands
streamlit-run:
	streamlit run kbac/web_app.py

# FastAPI commands
api-run:
	python kbac/api_server.py

# Dev container (existing)
devcontainer-build:
	[ -e .env ] || touch .env
	docker compose -f .devcontainer/docker-compose.yml build kbac-devcontainer
