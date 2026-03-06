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
| **LLM 支持** | 多模型切换（OpenAI / 本地开源模型） |

### 1.3 核心特性

- 支持多数据源接入（MySQL、PostgreSQL、MongoDB）
- 混合检索：向量检索 + 关键词检索
- 多 LLM 切换：OpenAI GPT、Claude、本地 Llama2/Qwen
- Web 界面：流式输出、对话历史、引用来源展示
- 可扩展架构：插件化向量库、灵活的分词器

---

## 二、技术架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web 层                                   │
│  ┌─────────────────┐    ┌─────────────────────────────────┐   │
│  │   Streamlit UI   │    │        FastAPI Backend          │   │
│  │  (前端交互界面)   │◄──►│   (REST API / WebSocket)        │   │
│  └─────────────────┘    └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                    │
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
| **Web 框架** | FastAPI | 高性能异步 API，支持 WebSocket 流式输出 |
| **前端** | Streamlit | 快速构建数据应用的 UI 框架 |
| **向量存储** | ChromaDB / FAISS | 轻量级向量数据库，支持本地部署 |
| **嵌入模型** | sentence-transformers | 开源中文/英文嵌入模型 |
| **LLM 客户端** | LangChain | 统一的多模型接入框架 |
| **数据库驱动** | SQLAlchemy + pymysql | 关系型数据库 ORM |
| **配置管理** | Pydantic Settings | 类型安全的配置管理 |
| **日志** | Loguru | 美化的日志输出 |

---

## 三、功能模块设计

### 3.1 核心模块

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
│   ├── retriever.py            # 检索器基类
│   ├── hybrid_retriever.py     # 混合检索实现
│   └── query_processor.py      # 查询处理器
├── llm/                        # LLM 模块
│   ├── base.py                 # LLM 基类
│   ├── openai_llm.py           # OpenAI 实现
│   ├── anthropic_llm.py        # Claude 实现
│   └── local_llm.py            # 本地模型实现
├── application/                # 应用层
│   ├── chat_manager.py         # 对话管理器
│   ├── rag_pipeline.py         # RAG 流程编排
│   └── context_builder.py      # 上下文构建器
├── api/                        # API 层
│   ├── routes.py               # API 路由
│   ├── schemas.py              # Pydantic 模型
│   └── websocket.py            # WebSocket 处理
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

### 阶段一：基础架构搭建（第 1-2 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 1.1 | 项目初始化、目录结构搭建 | `pyproject.toml`、基础代码框架 |
| 1.2 | 配置管理模块 | `config.yaml`、环境变量加载 |
| 1.3 | 日志系统 | 统一日志配置、请求追踪 |
| 1.4 | 数据库连接器 | 支持 MySQL/PostgreSQL 的基础连接 |

**代码示例 - 配置管理：**

```python
# core/config.py
"""配置管理模块 - 使用 Pydantic Settings 进行类型安全的配置管理"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置类
    
    从环境变量加载配置，支持默认值
    """
    # 数据库配置
    database_url: str = "mysql+pymysql://user:pass@localhost:3306/rag_db"
    
    # 向量数据库配置
    vector_store_type: str = "chroma"  # chroma / faiss
    persist_directory: str = "./data/vector_store"
    
    # 嵌入模型配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # cpu / cuda
    
    # LLM 配置
    llm_provider: str = "openai"  # openai / anthropic / ollama
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    
    # 检索配置
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()
```

### 阶段二：数据层开发（第 3-4 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 2.1 | 嵌入模型封装 | `Embedder` 类，支持批量向量化 |
| 2.2 | 向量存储实现 | ChromaDB / FAISS 封装 |
| 2.3 | 数据导入脚本 | 从数据库同步到向量库 |
| 2.4 | 数据增量更新 | 增量索引、删除同步 |

**代码示例 - 嵌入模型：**

```python
# data/embedder.py
"""嵌入模型管理模块 - 负责将文本转换为向量表示"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch


class Embedder:
    """嵌入模型封装类
    
    使用 sentence-transformers 库加载预训练模型，
    将文本转换为高维向量用于相似度计算
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """初始化嵌入模型
        
        Args:
            model_name: HuggingFace 模型名称
            device: 运行环境 (cpu/cuda)
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 根据设备选择加载模型
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # 获取向量维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy 数组，shape = (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """将单个查询转换为向量（别名方法）
        
        Args:
            query: 查询文本
            
        Returns:
            向量数组
        """
        return self.embed_texts([query])[0]
```

### 阶段三：检索模块开发（第 5-6 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 3.1 | 基础检索器 | 向量相似度检索 |
| 3.2 | 混合检索 | 向量 + 关键词 BM25 融合 |
| 3.3 | 查询预处理 | 分词、同义词扩展 |
| 3.4 | 结果重排序 | RRF 排序算法 |

**代码示例 - 混合检索器：**

```python
# retrieval/hybrid_retriever.py
"""混合检索模块 - 结合向量检索与关键词检索"""
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    """检索结果数据类
    
    用于封装单条检索结果，包含文本内容、相似度分数、
    元数据和原始数据源信息
    """
    content: str           # 检索到的文本内容
    score: float           # 相似度分数 (0-1)
    metadata: Dict         # 元数据 (来源表、ID 等)
    rank: int              # 排名


class HybridRetriever:
    """混合检索器
    
    融合向量检索与关键词检索，通过 RRF 算法对结果进行重排序
    """
    
    def __init__(
        self,
        vector_store,
        keyword_index: Optional[Any] = None,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        """初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            keyword_index: 关键词索引 (BM25)
            top_k: 返回结果数量
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    def search(
        self, 
        query: str, 
        query_vector: np.ndarray,
        filter_condition: Optional[Dict] = None
    ) -> List[SearchResult]:
        """执行混合检索
        
        1. 并行执行向量检索和关键词检索
        2. 使用 RRF 算法融合结果
        3. 返回融合后的 Top-K 结果
        
        Args:
            query: 查询文本
            query_vector: 查询的向量表示
            filter_condition: 过滤条件 (如时间范围、分类)
            
        Returns:
            按相关性排序的检索结果列表
        """
        # 向量检索
        vector_results = self._vector_search(query_vector, filter_condition)
        
        # 关键词检索
        keyword_results = self._keyword_search(query, filter_condition)
        
        # RRF 融合
        fused_results = self._reciprocal_rank_fusion(
            vector_results, 
            keyword_results
        )
        
        return fused_results[:self.top_k]
    
    def _vector_search(
        self, 
        query_vector: np.ndarray,
        filter_condition: Optional[Dict]
    ) -> List[SearchResult]:
        """向量检索"""
        # 调用向量存储的相似度搜索
        results = self.vector_store.similarity_search(
            query_vector=query_vector,
            k=self.top_k * 2,  # 多取一些用于融合
            filter=filter_condition
        )
        return results
    
    def _keyword_search(
        self, 
        query: str,
        filter_condition: Optional[Dict]
    ) -> List[SearchResult]:
        """关键词检索 (BM25)"""
        if self.keyword_index is None:
            return []
        
        results = self.keyword_index.search(
            query=query,
            k=self.top_k * 2,
            filter=filter_condition
        )
        return results
    
    def _reciprocal_rank_fusion(
        self,
        results_a: List[SearchResult],
        results_b: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """倒数排名融合算法 (RRF)
        
        RRF 公式: score = Σ(1 / (k + rank))
        其中 k 为平滑参数，rank 为排名位置
        
        Args:
            results_a: 第一组检索结果
            results_b: 第二组检索结果
            k: 平滑参数 (默认 60)
            
        Returns:
            融合后的排序结果
        """
        # 构建排名字典
        doc_scores = {}
        
        # 处理向量检索结果
        for rank, result in enumerate(results_a):
            doc_id = result.metadata.get("id", str(rank))
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.vector_weight
            result._rrf_score = rrf_score * self.vector_weight
        
        # 处理关键词检索结果
        for rank, result in enumerate(results_b):
            doc_id = result.metadata.get("id", str(rank))
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.keyword_weight
        
        # 合并结果并按分数排序
        all_results = {**{r.metadata.get("id"): r for r in results_a}, 
                       **{r.metadata.get("id"): r for r in results_b}}
        
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        fused = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            if doc_id in all_results:
                result = all_results[doc_id]
                result.score = score
                result.rank = rank + 1
                fused.append(result)
        
        return fused
```

### 阶段四：LLM 模块开发（第 7-8 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 4.1 | LLM 基类与工厂 | 统一接口、多模型支持 |
| 4.2 | OpenAI 集成 | GPT-3.5/4 接口封装 |
| 4.3 | Claude 集成 | Anthropic Claude 接口 |
| 4.4 | 本地模型集成 | Ollama 本地模型支持 |
| 4.5 | 流式输出 | Server-Sent Events 支持 |

**代码示例 - LLM 调度器：**

```python
# llm/orchestrator.py
"""LLM 调度模块 - 统一管理多模型调用"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, List, Dict
from dataclasses import dataclass
import os


@dataclass
class LLMResponse:
    """LLM 响应数据类"""
    content: str           # 生成的文本内容
    model: str             # 使用的模型名称
    usage: Dict            # token 使用量
    finish_reason: str     # 结束原因


class BaseLLM(ABC):
    """LLM 基类 - 定义统一接口"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> LLMResponse:
        """同步生成"""
        pass
    
    @abstractmethod
    async def stream_generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现类"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """初始化 OpenAI LLM
        
        Args:
            api_key: OpenAI API 密钥，默认从环境变量读取
            model: 模型名称
            temperature: 采样温度 (0-2)
            max_tokens: 最大生成 token 数
        """
        from openai import AsyncOpenAI
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> LLMResponse:
        """同步生成"""
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens)
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=dict(response.usage),
            finish_reason=response.choices[0].finish_reason
        )
    
    async def stream_generate(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成"""
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMOrchestrator:
    """LLM 调度器 - 工厂模式管理多模型
    
    根据配置动态选择 LLM 提供商，支持模型热切换
    """
    
    _providers = {
        "openai": OpenAILLM,
        # "anthropic": AnthropicLLM,
        # "ollama": OllamaLLM,
    }
    
    def __init__(self, provider: str = "openai", **kwargs):
        """初始化调度器
        
        Args:
            provider: LLM 提供商名称
            **kwargs: 传递给具体 LLM 类的参数
        """
        self.provider = provider
        self.llm = self._create_llm(provider, **kwargs)
    
    def _create_llm(self, provider: str, **kwargs) -> BaseLLM:
        """创建 LLM 实例"""
        llm_class = self._providers.get(provider)
        if not llm_class:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
        return llm_class(**kwargs)
    
    def switch_model(self, provider: str, **kwargs):
        """切换 LLM 模型"""
        self.provider = provider
        self.llm = self._create_llm(provider, **kwargs)
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        return await self.llm.generate(prompt, **kwargs)
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成"""
        return await self.llm.stream_generate(prompt, **kwargs)
```

### 阶段五：RAG 流程编排（第 9 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 5.1 | RAG Pipeline | 完整的检索增强生成流程 |
| 5.2 | Prompt 模板 | 结构化的提示词模板 |
| 5.3 | 引用来源 | 追溯生成内容的来源 |
| 5.4 | 缓存优化 | Redis 缓存查询结果 |

**代码示例 - RAG Pipeline：**

```python
# application/rag_pipeline.py
"""RAG 流程编排模块 - 整合检索与生成"""
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from retrieval.hybrid_retriever import HybridRetriever, SearchResult
from llm.orchestrator import LLMOrchestrator
import jinja2


@dataclass
class RAGResponse:
    """RAG 响应数据类"""
    answer: str                     # 生成的回答
    sources: List[SearchResult]     # 引用的来源
    model: str                      # 使用的模型
    query: str                      # 原始查询


class RAGPipeline:
    """RAG 流程编排器
    
    整合检索、上下文构建、LLM 生成的全流程
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_orchestrator: LLMOrchestrator,
        embedder,
        prompt_template: Optional[str] = None
    ):
        """初始化 RAG Pipeline
        
        Args:
            retriever: 混合检索器实例
            llm_orchestrator: LLM 调度器实例
            embedder: 嵌入模型实例
            prompt_template: 自定义提示词模板
        """
        self.retriever = retriever
        self.llm = llm_orchestrator
        self.embedder = embedder
        
        # 默认提示词模板
        self.prompt_template = prompt_template or self._default_template()
    
    def _default_template(self) -> str:
        """默认提示词模板"""
        template_str = """你是一个专业的问答助手。请根据以下参考信息回答用户的问题。

## 参考信息
{% for source in sources %}
[{{ loop.index }}] {{ source.content }}
来源: {{ source.metadata }}
{% endfor %}

## 用户问题
{{ query }}

## 回答要求
1. 仅根据提供的参考信息回答，不要编造信息
2. 如果参考信息不足以回答问题，请明确说明
3. 回答要清晰、准确、简洁
4. 在回答中注明来源编号

## 回答
"""
        return template_str
    
    async def chat(
        self, 
        query: str, 
        top_k: int = 5,
        stream: bool = False
    ) -> RAGResponse | AsyncIterator[str]:
        """执行 RAG 对话
        
        Args:
            query: 用户查询
            top_k: 检索结果数量
            stream: 是否流式输出
            
        Returns:
            RAGResponse 对象或流式文本迭代器
        """
        # Step 1: 将查询向量化
        query_vector = self.embedder.embed_query(query)
        
        # Step 2: 检索相关文档
        search_results = self.retriever.search(
            query=query,
            query_vector=query_vector,
            filter_condition=None
        )[:top_k]
        
        # Step 3: 构建上下文
        context = self._build_context(search_results)
        
        # Step 4: 生成 prompt
        prompt = self._build_prompt(query, search_results)
        
        # Step 5: 调用 LLM 生成
        if stream:
            # 流式输出
            async def generate():
                async for chunk in self.llm.stream_generate(prompt):
                    yield chunk
            return generate()
        else:
            response = await self.llm.generate(prompt)
            return RAGResponse(
                answer=response.content,
                sources=search_results,
                model=response.model,
                query=query
            )
    
    def _build_context(self, sources: List[SearchResult]) -> str:
        """构建上下文文本"""
        return "\n\n".join([
            f"[{i+1}] {source.content}" 
            for i, source in enumerate(sources)
        ])
    
    def _build_prompt(self, query: str, sources: List[SearchResult]) -> str:
        """渲染 prompt 模板"""
        template = jinja2.Template(self.prompt_template)
        return template.render(query=query, sources=sources)
```

### 阶段六：API 与 Web 界面开发（第 10-11 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 6.1 | FastAPI 基础 | 路由、中间件、错误处理 |
| 6.2 | REST API | 对话、检索、模型切换接口 |
| 6.3 | WebSocket | 流式响应支持 |
| 6.4 | Streamlit UI | 聊天界面、对话历史 |
| 6.5 | 前端优化 | 样式、加载状态、引用展示 |

**代码示例 - API 路由：**

```python
# api/routes.py
"""API 路由模块 - FastAPI 路由定义"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from application.rag_pipeline import RAGPipeline


router = APIRouter(prefix="/api/v1", tags=["RAG"])


# ========== Pydantic 模型 ==========

class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str
    top_k: int = 5
    stream: bool = False
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    sources: List[dict]
    model: str


class SwitchModelRequest(BaseModel):
    """切换模型请求"""
    provider: str
    model: str


# ========== 路由端点 ==========

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 聊天接口
    
    接收用户查询，返回检索增强后的回答
    
    Args:
        request: 聊天请求，包含查询内容、top_k、stream 等参数
        
    Returns:
        包含回答内容和来源引用的响应
    """
    try:
        # 获取或创建 pipeline 实例
        pipeline = get_rag_pipeline()
        
        # 切换模型（如果指定）
        if request.model:
            pipeline.llm.switch_model(request.model)
        
        # 执行 RAG 对话
        response = await pipeline.chat(
            query=request.query,
            top_k=request.top_k,
            stream=request.stream
        )
        
        # 序列化来源
        sources = [
            {
                "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                "score": s.score,
                "metadata": s.metadata
            }
            for s in response.sources
        ]
        
        return ChatResponse(
            answer=response.answer,
            sources=sources,
            model=response.model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口
    
    使用 Server-Sent Events 实现流式输出
    """
    pipeline = get_rag_pipeline()
    
    if request.model:
        pipeline.llm.switch_model(request.model)
    
    async def event_generator():
        try:
            async for chunk in await pipeline.chat(
                query=request.query,
                top_k=request.top_k,
                stream=True
            ):
                # 发送数据帧
                yield f"data: {chunk}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: error: {str(e)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/model/switch")
async def switch_model(request: SwitchModelRequest):
    """切换 LLM 模型"""
    pipeline = get_rag_pipeline()
    pipeline.llm.switch_model(
        provider=request.provider,
        model=request.model
    )
    return {"status": "success", "model": request.model}


# ========== 依赖注入 ==========

_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """获取 RAG Pipeline 实例（单例）"""
    global _rag_pipeline
    if _rag_pipeline is None:
        # 实际应用中从依赖注入获取
        raise HTTPException(status_code=500, detail="Pipeline 未初始化")
    return _rag_pipeline
```

**代码示例 - Streamlit 前端：**

```python
# ui/app.py
"""Streamlit 前端应用 - Web 聊天界面"""
import streamlit as st
import asyncio
from typing import List, Dict
import time


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model" not in st.session_state:
        st.session_state.model = "openai"


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.title("⚙️ 系统设置")
    
    # 模型选择
    model_provider = st.sidebar.selectbox(
        "LLM 提供商",
        ["openai", "anthropic", "ollama"],
        index=0
    )
    
    model_name = st.sidebar.selectbox(
        "模型",
        ["gpt-3.5-turbo", "gpt-4"] if model_provider == "openai" 
        else ["claude-3-opus", "claude-3-sonnet"]
    )
    
    # 检索参数
    top_k = st.sidebar.slider("检索数量", 1, 10, 5)
    
    # 清空对话
    if st.sidebar.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()
    
    return model_provider, model_name, top_k


def render_chat_message(role: str, content: str, sources: List[Dict] = None):
    """渲染单条聊天消息
    
    Args:
        role: 角色 (user/assistant)
        content: 消息内容
        sources: 引用来源
    """
    with st.chat_message(role):
        st.markdown(content)
        
        # 显示引用来源
        if sources and role == "assistant":
            with st.expander("📚 参考来源"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    **{i}.** {source['content'][:150]}...
                    
                    相似度: `{source['score']:.2f}`
                    """)


async def call_api(query: str, model_provider: str, model_name: str, top_k: int):
    """调用后端 API"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/chat",
            json={
                "query": query,
                "model": model_name,
                "top_k": top_k,
                "stream": False
            }
        ) as resp:
            return await resp.json()


def main():
    """主函数"""
    st.set_page_config(
        page_title="RAG 智能问答系统",
        page_icon="💬",
        layout="wide"
    )
    
    # 初始化
    init_session_state()
    
    # 渲染侧边栏
    model_provider, model_name, top_k = render_sidebar()
    
    # 标题
    st.title("💬 RAG 智能问答系统")
    st.markdown("基于检索增强生成的大语言模型问答系统")
    
    # 渲染历史消息
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"], msg.get("sources"))
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        render_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 调用 API 获取回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                try:
                    # 同步调用（实际使用异步）
                    response = asyncio.run(
                        call_api(prompt, model_provider, model_name, top_k)
                    )
                    
                    answer = response.get("answer", "抱歉，生成回答失败")
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # 显示引用
                    if sources:
                        with st.expander("📚 参考来源"):
                            for i, s in enumerate(sources, 1):
                                st.markdown(f"**{i}.** {s['content'][:200]}...")
                    
                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")


if __name__ == "__main__":
    main()
```

### 阶段七：测试与部署（第 12 周）

| 序号 | 任务 | 交付物 |
|------|------|--------|
| 7.1 | 单元测试 | 核心模块测试用例 |
| 7.2 | 集成测试 | API 端到端测试 |
| 7.3 | 性能测试 | 检索速度、并发能力 |
| 7.4 | Docker 镜像 | 容器化部署配置 |
| 7.5 | 部署文档 | 环境搭建、运行说明 |

**Docker 配置示例：**

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  # RAG API 服务
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  # Streamlit 前端
  rag-ui:
    image: streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./ui:/app
    command: streamlit run app.py
    depends_on:
      - rag-api

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

---

## 五、开发时间线总览

```
┌────────────────────────────────────────────────────────────────────────┐
│                         开发时间线 (12 周)                              │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ W1   │ W2   │ W3   │ W4   │ W5   │ W6   │ W7   │ W8   │ W9   │ W10  │ W11  │ W12  │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ 基础架构 │      │ 数据层    │      │ 检索模块  │      │ LLM模块  │      │ 流程编排│ API/Web│    │ 测试部署│
│ 1.1-1.4    │  2.1-2.4  │  3.1-3.4  │  4.1-4.5 │  5.1-5.4 │  6.1-6.5  │ 7.1-7.5 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                                              ↑
                                         并行开发
```

---

## 六、关键风险与应对

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| 向量检索效果不佳 | 回答质量低 | 调优嵌入模型、添加查询改写 |
| LLM API 成本高 | 运营成本上升 | 添加缓存、本地模型备选 |
| 中文分词效果 | 检索准确性 | 使用专业中文分词库 |
| 大并发性能 | 响应延迟 | 添加 Redis 缓存、异步队列 |
| 数据安全 | 隐私泄露 | 数据脱敏、访问控制 |

---

## 七、后续扩展方向

1. **多模态检索**：支持图片、PDF 内容的检索
2. **Agent 能力**：支持工具调用、多轮对话
3. **个性化**：用户偏好学习、对话风格定制
4. **监控运维**：请求日志、指标监控、告警

---

*文档版本: v1.0*  
*创建日期: 2024*
