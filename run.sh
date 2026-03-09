#!/bin/bash

# RAG System Startup Script
# 使用方法: ./run.sh

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run 'uv sync' first."
    exit 1
fi

# 设置 Python 路径为 src 目录
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

echo "Starting RAG API Server..."
cd src
uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
