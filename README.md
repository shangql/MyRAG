# MyRAG - 大语言模型 RAG 搜索补充系统

基于 Python 的检索增强生成（RAG）系统，从数据库中检索知识，结合大语言模型生成准确、上下文相关的回答。

## 功能特性

- **混合检索**：向量检索 + 关键词检索（RRF 融合算法）
- **多模型支持**：OpenAI GPT、Claude、Ollama 本地模型
- **Web 界面**：原生 HTML/JS 聊天界面，支持流式输出
- **REST API**：FastAPI 高性能接口
- **向量存储**：支持 ChromaDB 和 FAISS
- **类型安全**：完整的类型注解和 Pydantic 验证

## 快速开始

### 环境要求

- Python 3.10+
- uv 包管理工具（推荐）

### 安装步骤

```bash
# 1. 克隆项目
cd MyRAG

# 2. 创建虚拟环境
uv venv .venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 安装依赖
uv pip install -e .

# 5. 复制环境配置
cp .env.example .env

# 6. 编辑 .env 填入你的 API 密钥
```

### 运行服务

```bash
# 方式一：使用 Python 直接运行

# 启动 API 服务
python main.py api --host 0.0.0.0 --port 8000

# 启动 Web 界面（新终端）
python main.py ui


# 方式二：使用 Docker

# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f


# 方式三：分别启动

# 启动 API
./run.sh
```

### 访问地址

| 服务 | 地址 |
|------|------|
| Web 界面 | http://localhost:8000 |
| API 文档 | http://localhost:8000/docs |

## 配置说明

### 环境变量

在 `.env` 文件中配置以下选项：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DATABASE_URL` | 数据库连接 URL | mysql+pymysql://user:password@localhost:3306/rag_db |
| `VECTOR_STORE_TYPE` | 向量存储类型 | chroma |
| `EMBEDDING_MODEL` | 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| `LLM_PROVIDER` | LLM 提供商 | openai |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `TOP_K` | 检索结果数量 | 5 |

### 支持的 LLM

| 提供商 | 模型示例 | 需要配置 |
|--------|----------|----------|
| OpenAI | gpt-3.5-turbo, gpt-4 | OPENAI_API_KEY |
| Anthropic | claude-3-sonnet | ANTHROPIC_API_KEY |
| Ollama | llama2, qwen | OLLAMA_BASE_URL |

## 项目结构

```
MyRAG/
├── src/
│   ├── core/           # 核心模块
│   │   ├── config.py  # 配置管理
│   │   ├── logger.py  # 日志系统
│   │   └── exceptions.py  # 异常定义
│   ├── data/          # 数据层
│   │   ├── database.py  # 数据库连接
│   │   ├── embedder.py  # 嵌入模型
│   │   └── vector_store.py  # 向量存储
│   ├── retrieval/     # 检索模块
│   │   └── hybrid_retriever.py  # 混合检索
│   ├── llm/          # LLM 模块
│   │   └── orchestrator.py  # 多模型调度
│   ├── application/  # 应用层
│   │   └── rag_pipeline.py  # RAG 流程
│   ├── api/          # API 层
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── ui/           # 前端界面
│       └── app.py
├── tests/            # 测试文件
├── docker-compose.yml  # Docker 编排
├── Dockerfile         # Docker 镜像
├── pyproject.toml     # 项目配置
└── main.py            # 入口文件
```

## API 使用

### 聊天接口

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "top_k": 5,
    "model": "gpt-3.5-turbo"
  }'
```

响应示例：

```json
{
  "answer": "机器学习是人工智能的一个分支...",
  "sources": [
    {
      "content": "机器学习是...",
      "score": 0.95,
      "metadata": {"source": "wiki"}
    }
  ],
  "model": "gpt-3.5-turbo",
  "query": "什么是机器学习？"
}
```

### 流式输出

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "解释一下深度学习",
    "stream": true
  }'
```

### 切换模型

```bash
curl -X POST http://localhost:8000/api/v1/model/switch \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4"
  }'
```

## 开发指南

### 代码检查

```bash
# 格式化代码
ruff format .

# 自动修复
ruff check --fix .

# 类型检查
mypy src/ --strict

# 全部检查
ruff check . && ruff format --check . && mypy src/
```

### 运行测试

```bash
# 所有测试
pytest

# 指定文件
pytest tests/test_config.py

# 指定函数
pytest tests/test_retriever.py::TestHybridRetriever::test_search_with_both

# 带覆盖率
pytest --cov=src --cov-report=html

# 仅失败项
pytest --lf
```

### 添加数据

```python
from src.data import get_embedder, get_vector_store
from src.application import create_rag_pipeline

# 初始化组件
embedder = get_embedder()
vector_store = get_vector_store()

# 添加文档
documents = [
    {"id": "doc1", "content": "Python 是一种高级编程语言...", "metadata": {"source": "wiki"}},
    {"id": "doc2", "content": "机器学习是人工智能的分支...", "metadata": {"source": "wiki"}},
]

# 创建 Pipeline 并添加
pipeline = create_rag_pipeline(vector_store, embedder)
await pipeline.add_documents(documents)
```

## 常见问题

### Q: 向量检索效果不好？

A: 尝试：
1. 调整嵌入模型（`EMBEDDING_MODEL`）
2. 修改 `TOP_K` 参数
3. 调整向量检索和关键词检索的权重

### Q: API 响应慢？

A: 优化建议：
1. 使用本地模型（Ollama）
2. 启用 Redis 缓存
3. 增加向量数据库索引

### Q: 如何添加更多数据源？

A: 参考 `src/data/database.py` 实现新的数据连接器。

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + 原生 HTML |
| 向量存储 | ChromaDB, FAISS |
| 嵌入模型 | sentence-transformers |
| LLM | OpenAI, Anthropic, Ollama |
| 数据库 | SQLAlchemy, MySQL, PostgreSQL |
| 测试 | pytest, pytest-asyncio |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
