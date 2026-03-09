# MyRAG - 大语言模型 RAG 搜索补充系统

基于 Python 的检索增强生成（RAG）系统，从数据库中检索知识，结合大语言模型生成准确、上下文相关的回答。

## 功能特性

- **文件解析**：支持 TXT、Markdown、PDF、DOCX、XLSX、CSV、HTML 等多种格式
- **多模型支持**：ModelScope（DeepSeek Qwen 系列）、OpenAI GPT、Anthropic Claude、Ollama 本地模型
- **Web 界面**：原生 HTML/JS 聊天界面，支持流式输出
- **REST API**：FastAPI 高性能接口
- **向量存储**：支持 ChromaDB 和 FAISS
- **类型安全**：完整的类型注解和 Pydantic 验证

## 快速开始

### 环境要求

- Python 3.12+
- uv 包管理工具（推荐）

### 安装步骤

```bash
# 1. 克隆项目
cd MyRAG

# 2. 创建虚拟环境（使用 Python 3.12）
uv venv .venv --python 3.12

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
# 启动 API 服务
./run.sh

# 访问 Web 界面
http://localhost:8000
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
| `DATABASE_URL` | 数据库连接 URL | mysql+pymysql://rag_db:password@192.168.102.45:3306/rag_db |
| `VECTOR_STORE_TYPE` | 向量存储类型 | chroma |
| `EMBEDDING_MODEL` | 嵌入模型 | sentence-transformers/all-MiniLM-L6-v2 |
| `LLM_PROVIDER` | LLM 默认提供商 | modelscope |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `MODELSCOPE_API_KEY` | ModelScope API 密钥 | - |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | - |
| `TOP_K` | 检索结果数量 | 5 |

### 支持的 LLM

| 提供商 | 模型示例 | 说明 |
|--------|----------|------|
| **ModelScope** | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 默认推荐 |
| **ModelScope** | Qwen/Qwen2-7B-Instruct | 支持 |
| OpenAI | gpt-3.5-turbo, gpt-4 | 需要 OPENAI_API_KEY |
| Anthropic | claude-3-sonnet | 需要 ANTHROPIC_API_KEY |
| Ollama | llama2, qwen | 需要 OLLAMA_BASE_URL |

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
│   │   ├── vector_store.py  # 向量存储
│   │   └── file_model.py  # 文件数据模型
│   ├── retrieval/     # 检索模块
│   │   └── hybrid_retriever.py  # 混合检索
│   ├── llm/          # LLM 模块
│   │   └── orchestrator.py  # 多模型调度
│   ├── application/  # 应用层
│   │   └── rag_pipeline.py  # RAG 流程
│   ├── api/          # API 层
│   │   ├── main.py   # FastAPI 应用
│   │   ├── routes.py # API 路由
│   │   ├── parser.py # 文件解析器
│   │   └── schemas.py # 数据模型
│   └── ui/           # 前端界面
│       └── templates/ # HTML 模板
├── data/
│   ├── uploads/      # 上传文件存储
│   └── vector_store/ # ChromaDB 向量存储
├── tests/            # 测试文件
├── run.sh            # 启动脚本
├── docker-compose.yml # Docker 编排
├── pyproject.toml    # 项目配置
└── .env              # 环境配置
```

## API 使用

### 聊天接口

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "provider": "modelscope",
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "top_k": 5
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
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "query": "什么是机器学习？"
}
```

### 流式输出

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "解释一下深度学习",
    "provider": "modelscope",
    "stream": true
  }'
```

### 文件上传

```bash
curl -X POST http://localhost:8000/api/v1/files/upload \
  -F "file=@/path/to/document.xlsx"
```

### 文件列表

```bash
curl http://localhost:8000/api/v1/files
```

### 导入向量库

```bash
curl -X POST http://localhost:8000/api/v1/files/1/import
```

## 支持的文件格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 纯文本 | .txt | 直接读取内容 |
| Markdown | .md, .markdown | 转换为 HTML 后提取文本 |
| PDF | .pdf | 使用 pypdf 提取文本 |
| Word | .docx | 使用 python-docx 提取文本 |
| Excel | .xlsx, .xls | 使用 pandas 提取所有 Sheet |
| CSV | .csv | 使用 pandas 解析 |
| HTML | .html, .htm | 使用 BeautifulSoup 提取文本 |
| 图片 | .jpg, .png | 使用 OCR 提取文字 |
| 邮件 | .eml | 提取邮件主题和正文 |

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

## 常见问题

### Q: LLM 调用失败？

A: 检查：
1. 是否配置了正确的 API Key（ModelScope / OpenAI / Anthropic）
2. 网络是否能访问 API 服务
3. 查看日志 `src/logs/app.log`

### Q: 文件上传失败？

A: 检查：
1. 数据库连接是否正常
2. 上传目录是否有写权限
3. 文件大小是否超出限制

### Q: Excel 文件解析失败？

A: 确保已安装 openpyxl：
```bash
uv pip install openpyxl
```

### Q: 向量检索效果不好？

A: 尝试：
1. 调整嵌入模型（`EMBEDDING_MODEL`）
2. 修改 `TOP_K` 参数
3. 调整向量检索和关键词检索的权重

## 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI + 原生 HTML |
| 向量存储 | ChromaDB, FAISS |
| 嵌入模型 | sentence-transformers |
| LLM | ModelScope, OpenAI, Anthropic, Ollama |
| 数据库 | SQLAlchemy, MySQL |
| 文件解析 | pandas, openpyxl, pypdf, python-docx |
| 测试 | pytest, pytest-asyncio |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
