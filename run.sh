#!/bin/bash

# RAG System Startup Script
# 使用方法: ./run.sh [api|ui]

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

# 默认启动 API
MODE="${1:-api}"

case "$MODE" in
    api)
        echo "Starting RAG API Server..."
        cd src
        uvicorn api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --reload
        ;;
    ui)
        echo "Starting RAG Streamlit UI..."
        cd src
        streamlit run ui/app.py --server.port 8501
        ;;
    all)
        echo "Starting both API and UI..."
        echo "Please run 'uvicorn src.api.main:app' in one terminal"
        echo "And 'streamlit run src/ui/app.py' in another terminal"
        ;;
    *)
        echo "Usage: $0 [api|ui|all]"
        exit 1
        ;;
esac
