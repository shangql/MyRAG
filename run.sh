#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 优先使用 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -f "$HOME/miniforge/bin/conda" ]; then
    source "$HOME/miniforge/bin/activate" myrag 2>/dev/null || true
    echo "Activated conda environment: myrag"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: No conda environment or virtual environment found."
    exit 1
fi

PORT="${API_PORT:-8000}"

echo "Checking for existing uvicorn process on port $PORT..."
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing existing process on port $PORT (PID: $EXISTING_PID)"
    kill $EXISTING_PID 2>/dev/null || true
    sleep 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

echo "Starting RAG API Server..."
cd src
uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "$PORT" --reload
