# LLM RAG 搜索补充系统开发计划

## 一、项目概述

### 1.1 项目背景

本项目旨在构建一个基于 Python 的大语言模型 RAG（Retrieval-Augmented Generation，检索增强生成）搜索补充系统。系统从数据库中检索知识，结合大语言模型生成准确、上下文相关的回答。

### 1.2 需求分析

| 需求项 | 说明 |
|--------|------|
| **数据源** | 数据库（结构化数据） |
| **数据规模** | 小型（<10万条记录） |
| **交互方式** | Web 聊天界面 |
| **LLM 支持** | ModelScope（国内可用，每日 2000 次免费调用） |

### 1.3 核心特性

- 支持多数据源接入（MySQL、PostgreSQL、MongoDB）
- 混合检索：向量检索 + 关键词检索
- LLM：ModelScope（DeepSeek-R1、Qwen3 等开源大模型）
- Web 界面：流式输出、对话历史、引用来源展示
- 可扩展架构：插件化向量库、灵活的分词器

---

## 二、技术架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│ Web 层 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 原生 HTML/JS 单页应用 (SPA) │ │
│ │ • 聊天对话界面 • 流式输出 • 对话历史 │ │
│ └─────────────────────────────────────────────────────────┘ │
│ │
▼ ┌─────────────────────────────────────────────────────────┐ │
│ FastAPI Backend │ │
│ • REST API • SSE 流式输出 • 文件上传 │ │
└─────────────────────────────────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────────────┐
│                      应用层 (Application)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   对话管理器   │  │   检索引擎   │  │     LLM 调度器       │  │
│  │ (ChatManager)│  │(Retriever)   │  │   (LLMOrchestrator)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据层 (Data)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  数据库连接   │  │  向量数据库   │  │    嵌入模型          │  │
│  │(DBConnector) │  │ (VectorStore)│  │   (EmbeddingModel)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      基础设施层 (Infra)                          │
│         Python 3.10+  ·  Docker  ·  Redis (缓存)                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈选型

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **Web 框架** | FastAPI | 高性能异步 API，支持 SSE 流式输出 |
| **前端** | 原生 HTML/JS | 轻量级单页应用，无需构建工具 |
| **向量存储** | ChromaDB / FAISS | 轻量级向量数据库，支持本地部署 |
| **嵌入模型** | sentence-transformers | 开源中文/英文嵌入模型 |
| **LLM 客户端** | OpenAI 兼容 API | 通过 OpenAI 兼容接口接入 ModelScope / Ollama |
| **数据库驱动** | SQLAlchemy + pymysql | 关系型数据库 ORM |
| **配置管理** | Pydantic Settings | 类型安全的配置管理 |
| **日志** | Loguru | 美化的日志输出 |

---

## 三、功能模块设计

### 3.1 核心模块

```
rag_system/
├── core/ # 核心模块
│ ├── config.py # 配置管理
│ ├── logger.py # 日志配置
│ └── exceptions.py # 自定义异常
├── data/ # 数据层
│ ├── database.py # 数据库连接器
│ ├── vector_store.py # 向量存储管理
│ ├── embedder.py # 嵌入模型管理
│ └── file_model.py # 文件数据模型
├── retrieval/ # 检索模块
│ ├── hybrid_retriever.py # 混合检索实现 (向量 + BM25)
│ └── query_processor.py # 查询处理器
├── llm/ # LLM 模块
│ ├── orchestrator.py # LLM 调度器
│ └── modelscope_llm.py # ModelScope 实现 (OpenAI 兼容)
├── application/ # 应用层
│ └── rag_pipeline.py # RAG 流程编排
├── api/ # API 层
│ ├── main.py # FastAPI 应用入口
│ ├── routes.py # API 路由
│ ├── parser.py # 文件解析器
│ └── schemas.py # Pydantic 模型
└── ui/ # 前端界面
    └── templates/ # HTML 模板
        └── index.html # 聊天界面
```
rag_system/
├── core/                       # 核心模块
│   ├── config.py               # 配置管理
│   ├── logger.py               # 日志配置
│   └── exceptions.py           # 自定义异常
├── data/                       # 数据层
│   ├── database.py             # 数据库连接器
│   ├── vector_store.py         # 向量存储管理
│   └── embedder.py             # 嵌入模型管理
├── retrieval/                  # 检索模块
│   ├── hybrid_retriever.py     # 混合检索实现 (向量 + BM25)
│   └── query_processor.py      # 查询处理器
├── llm/                        # LLM 模块
│   ├── orchestrator.py         # LLM 调度器
│   └── modelscope_llm.py       # ModelScope 实现 (OpenAI 兼容)
├── application/                # 应用层
│   └── rag_pipeline.py         # RAG 流程编排
├── api/                        # API 层
│   ├── main.py                 # FastAPI 应用入口
│   ├── routes.py               # API 路由
│   └── schemas.py              # Pydantic 模型
└── ui/                         # 前端界面
    └── app.py                  # Streamlit 应用
```

### 3.2 核心流程

```
用户输入查询
     │
     ▼
┌─────────────────┐
│  查询预处理      │  → 分词、拼写纠错、意图识别
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  混合检索引擎    │  → 向量相似度 + 关键词匹配
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  结果重排序      │  → RRF 融合算法
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  上下文构建      │  → 选取 Top-K 片段拼接 prompt
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  LLM 生成回答    │  → 流式输出 + 引用来源
└─────────────────┘
```

---

## 四、开发阶段规划

### 阶段一：基础架构搭建 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 1.1 | 项目初始化、目录结构搭建 | ✅ 完成 |
| 1.2 | 配置管理模块（支持 .env） | ✅ 完成 |
| 1.3 | 日志系统（Loguru） | ✅ 完成 |
| 1.4 | 数据库连接器（MySQL） | ✅ 完成 |

### 阶段二：数据层开发 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 2.1 | 嵌入模型封装（sentence-transformers） | ✅ 完成 |
| 2.2 | 向量存储实现（ChromaDB / FAISS） | ✅ 完成 |
| 2.3 | 数据导入脚本 | ⏳ 待开发 |
| 2.4 | 数据增量更新 | ⏳ 待开发 |

### 阶段三：检索模块开发 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 3.1 | 基础检索器（向量相似度） | ✅ 完成 |
| 3.2 | 混合检索（向量 + BM25） | ✅ 完成 |
| 3.3 | 查询预处理 | ⏳ 待开发 |
| 3.4 | 结果重排序（RRF） | ✅ 完成 |

### 阶段四：LLM 模块开发 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 4.1 | LLM 调度器（ModelScope） | ✅ 完成 |
| 4.2 | ModelScope 集成（DeepSeek-R1、Qwen3） | ✅ 完成 |
| 4.3 | OpenAI 兼容接口 | ✅ 完成 |
| 4.4 | 流式输出（SSE） | ✅ 完成 |

### 阶段五：RAG 流程编排 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 5.1 | RAG Pipeline | ✅ 完成 |
| 5.2 | Prompt 模板（Jinja2） | ✅ 完成 |
| 5.3 | 引用来源追溯 | ✅ 完成 |
| 5.4 | 缓存优化 | ⏳ 待开发（Redis） |

### 阶段六：API 与 Web 界面开发 ✅ 完成

| 序号 | 任务 | 状态 |
|------|------|------|
| 6.1 | FastAPI 基础 | ✅ 完成 |
| 6.2 | REST API | ✅ 完成 |
| 6.3 | 流式响应 (SSE) | ✅ 完成 |
| 6.4 | 原生 HTML/JS 前端 | ✅ 完成 |
| 6.5 | 文件上传解析 | ✅ 完成 |

### 阶段七：测试与部署 ⏳ 待开发

| 序号 | 任务 | 状态 |
|------|------|------|
| 7.1 | 单元测试 | ⏳ 待开发 |
| 7.2 | 集成测试 | ⏳ 待开发 |
| 7.3 | 性能测试 | ⏳ 待开发 |
| 7.4 | Docker 镜像 | ⏳ 待开发 |
| 7.5 | 部署文档 | ⏳ 待开发 |

---

## 五、待开发功能清单

### 5.1 数据导入模块

```python
# scripts/import_data.py
"""数据导入脚本 - 从数据库同步到向量库"""
import sys
sys.path.insert(0, 'src')

from data import get_db, get_embedder, get_vector_store
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


def import_documents(table_name: str, batch_size: int = 100):
    """从数据库表导入文档到向量库

    Args:
        table_name: 数据库表名
        batch_size: 批处理大小
    """
    embedder = get_embedder()
    vector_store = get_vector_store()

    # TODO: 从数据库读取数据
    # TODO: 批量向量化
    # TODO: 添加到向量存储
    pass


if __name__ == "__main__":
    import_documents("your_table")
```

### 5.2 查询预处理模块

```python
# retrieval/query_processor.py
"""查询预处理模块 - 分词、同义词扩展、查询改写"""
import jieba
from typing import List


class QueryProcessor:
    """查询处理器

    负责对用户查询进行预处理，提升检索效果。
    """

    def __init__(self, use_synonym: bool = True):
        """初始化查询处理器

        Args:
            use_synonym: 是否使用同义词扩展
        """
        self.use_synonym = use_synonym

    def process(self, query: str) -> str:
        """处理查询

        Args:
            query: 原始查询文本

        Returns:
            处理后的查询文本
        """
        # 分词
        tokens = self.tokenize(query)
        # TODO: 同义词扩展
        # TODO: 查询改写
        return " ".join(tokens)

    def tokenize(self, query: str) -> List[str]:
        """中文分词

        Args:
            query: 查询文本

        Returns:
            分词后的词列表
        """
        return list(jieba.cut(query))
```

### 5.3 Redis 缓存模块

```python
# core/cache.py
"""Redis 缓存模块 - 缓存查询结果"""
import json
from typing import Optional
import redis

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """缓存管理器

    使用 Redis 缓存查询结果，提升响应速度。
    """

    def __init__(self, redis_url: Optional[str] = None):
        """初始化缓存管理器

        Args:
            redis_url: Redis 连接 URL
        """
        self.redis_url = redis_url or settings.redis_url
        self._client = None

    @property
    def client(self):
        """获取 Redis 客户端"""
        if self._client is None and self.redis_url:
            self._client = redis.from_url(self.redis_url)
        return self._client

    def get(self, key: str) -> Optional[dict]:
        """获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在返回 None
        """
        if not self.client:
            return None
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"缓存获取失败: {e}")
            return None

    def set(self, key: str, value: dict, ttl: int = 3600) -> bool:
        """设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        if not self.client:
            return False
        try:
            self.client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"缓存设置失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.client:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"缓存删除失败: {e}")
            return False


# 全局缓存管理器
cache_manager = CacheManager()
```

---

## 六、启动指南

### 6.1 环境配置

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填写必要的配置

# 3. 启动服务
./run.sh

# 4. 访问 Web 界面
http://localhost:8000
```

### 6.2 启动脚本说明

`run.sh` 会自动：
1. 激活 conda 环境 `myrag`（包含 PyTorch 2.10.0）
2. 启动 FastAPI 服务（端口 8000）
3. 启动原生 HTML 前端

### 6.2 API 文档

启动后访问：http://localhost:8000/docs

### 6.3 可用模型

| 提供商 | 模型 | 说明 |
|--------|------|------|
| **ModelScope** | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 推理模型（需要 API Key） |
| **ModelScope** | Qwen/Qwen3-8B | 通用模型 |
| **Ollama** (本地) | qwen:7b | 本地部署，无需外网 |
| **Ollama** (本地) | llama2 | 本地部署 |

---

## 七、系统当前状态

### 7.1 已完成 ✅

| 模块 | 状态 | 说明 |
|------|------|------|
| 项目初始化 | ✅ | 目录结构、配置管理 |
| PyTorch 2.10.0 | ✅ | 通过 conda 安装 |
| 嵌入模型 | ✅ | sentence-transformers/all-MiniLM-L6-v2 |
| 向量存储 | ✅ | ChromaDB 本地部署 |
| 文件解析 | ✅ | TXT/PDF/DOCX/XLSX/CSV/HTML |
| 文件上传 | ✅ | MySQL 持久化存储 |
| RAG Pipeline | ✅ | 混合检索 + LLM 生成 |
| API 接口 | ✅ | REST API + SSE 流式输出 |
| 前端界面 | ✅ | 原生 HTML/JS |

### 7.2 待解决 ⚠️

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| ModelScope API 超时 | ⚠️ 网络问题 | 使用 Ollama 本地模型 |
| 查询预处理 | ⏳ 待开发 | jieba 中文分词 |
| Redis 缓存 | ⏳ 待开发 | 缓存查询结果 |
| 单元测试 | ⏳ 待开发 | pytest 覆盖率测试 |

### 7.3 启动步骤（当前配置）

```bash
# 使用 conda 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate myrag

# 启动服务
cd /Users/sql/GitHub/MyRAG
./run.sh

# 访问
# 前端: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 7.4 启用本地 Ollama 模型（如 ModelScope 不可用）

```bash
# 1. 安装 Ollama
brew install ollama

# 2. 启动服务
ollama serve

# 3. 下载模型
ollama pull qwen:7b

# 4. 配置 .env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen:7b
```

---

## 八、关键风险与应对

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| ModelScope API 网络问题 | LLM 调用超时 | 切换到 Ollama 本地模型 |
| 向量检索效果不佳 | 回答质量低 | 调优嵌入模型、添加查询改写 |
| LLM API 配额不足 | 服务不可用 | 使用 Ollama 本地模型 |
| 中文分词效果 | 检索准确性 | 使用专业中文分词库（jieba） |
| 大并发性能 | 响应延迟 | 添加 Redis 缓存、异步队列 |
| 数据安全 | 隐私泄露 | 数据脱敏、访问控制 |

---

## 九、后续扩展方向

1. **Agent 能力**：支持工具调用、多轮对话
2. **个性化**：用户偏好学习、对话风格定制
3. **监控运维**：请求日志、指标监控、告警
4. **性能优化**：Redis 缓存、查询预处理

---

*文档版本: v3.0*
*更新日期: 2026-03-10*
*更新内容：移除 Streamlit UI，改为原生 HTML/JS 前端；添加系统当前状态*
