# MyRAG Dockerfile
# 基于 Python 3.10 轻量镜像构建

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml .

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/vector_store /app/logs

# 暴露端口
EXPOSE 8000 8501

# 启动命令（默认启动 API 服务）
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
