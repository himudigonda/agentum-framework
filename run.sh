#!/bin/bash

# Agentum Framework Run Script
# Usage: ./run.sh <command> [args]

set -e

case "$1" in
    "local")
        echo "🚀 Running agentum framework locally..."
        uv run python examples/01_hello_agentum.py
        ;;
    "test")
        echo "🧪 Running tests..."
        uv run pytest
        ;;
    "install")
        echo "📦 Installing dependencies..."
        uv sync --dev
        ;;
    "lint")
        echo "🔍 Running linters..."
        uv run ruff check .
        uv run black --check .
        ;;
    "format")
        echo "✨ Formatting code..."
        uv run black .
        uv run ruff check --fix .
        ;;
    "pre-commit")
        echo "🪝 Installing pre-commit hooks..."
        uv run pre-commit install
        ;;
    "dump")
        echo "📋 Creating codebase snapshot..."
        ./dumper.sh
        ;;
    *)
        echo "Usage: $0 {local|test|install|lint|format|pre-commit|dump}"
        echo ""
        echo "Commands:"
        echo "  local      - Run the example script locally"
        echo "  test       - Run the test suite"
        echo "  install    - Install all dependencies"
        echo "  lint       - Run linters"
        echo "  format     - Format code"
        echo "  pre-commit - Install pre-commit hooks"
        echo "  dump       - Create a complete codebase snapshot"
        exit 1
        ;;
esac
