"""RAG 流程编排模块 - 整合检索与生成"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from jinja2 import Template

from core.config import settings
from core.exceptions import PipelineError
from core.logger import get_logger
from llm.orchestrator import LLMOrchestrator, LLMResponse, Message
from retrieval.hybrid_retriever import HybridRetriever, SearchResult

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """RAG 响应数据类
    
    Attributes:
        answer: 生成的回答文本
        sources: 检索到的来源列表
        model: 使用的模型名称
        query: 原始用户查询
        usage: token 使用量统计
    """
    answer: str
    sources: List[SearchResult]
    model: str
    query: str
    usage: Dict[str, int] = field(default_factory=dict)


class RAGPipeline:
    """RAG 流程编排器
    
    整合检索、上下文构建、LLM 生成的全流程。
    支持流式输出、引用来源追踪。
    
    Attributes:
        retriever: 混合检索器实例
        llm: LLM 调度器实例
        embedder: 嵌入模型实例
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMOrchestrator,
        embedder: Any,
        prompt_template: Optional[str] = None,
    ):
        """初始化 RAG Pipeline
        
        Args:
            retriever: 混合检索器实例
            llm: LLM 调度器实例
            embedder: 嵌入模型实例
            prompt_template: 自定义提示词模板
        """
        self.retriever = retriever
        self.llm = llm
        self.embedder = embedder
        
        # 默认提示词模板
        self.prompt_template = prompt_template or self._default_template()
        
        logger.info("RAG Pipeline 初始化完成")
    
    def _default_template(self) -> str:
        """获取默认提示词模板
        
        Returns:
            str: Jinja2 模板字符串
        """
        template_str = """你是一个专业的问答助手。请根据以下参考信息回答用户的问题。

## 参考信息
{% for source in sources %}
[{{ loop.index }}] {{ source.content }}
{% if source.metadata %}
来源: {{ source.metadata }}
{% endif %}
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
        stream: bool = False,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse | AsyncIterator[str]:
        """执行 RAG 对话
        
        流程：
        1. 将查询向量化
        2. 检索相关文档
        3. 构建上下文
        4. 生成 prompt
        5. 调用 LLM 生成回答
        
        Args:
            query: 用户查询
            top_k: 检索结果数量
            stream: 是否流式输出
            filter_condition: 检索过滤条件
            
        Returns:
            RAGResponse 对象或流式文本迭代器
        """
        logger.info(f"处理查询: {query[:50]}...")
        
        # Step 1: 将查询向量化
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            raise PipelineError(f"查询向量化失败: {str(e)}")
        
        # Step 2: 检索相关文档
        try:
            search_results = self.retriever.search(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                filter_condition=filter_condition,
            )
        except Exception as e:
            raise PipelineError(f"文档检索失败: {str(e)}")
        
        logger.debug(f"检索到 {len(search_results)} 条相关文档")
        
        # 检查是否有检索结果
        if not search_results:
            return RAGResponse(
                answer="抱歉，我没有找到相关的参考信息来回答这个问题。",
                sources=[],
                model=self.llm.model,
                query=query,
            )
        
        # Step 3 & 4: 构建 prompt
        prompt = self._build_prompt(query, search_results)
        
        # Step 5: 调用 LLM 生成
        if stream:
            return self._stream_response(prompt, search_results, query)
        else:
            return await self._generate_response(prompt, search_results, query)
    
    async def _generate_response(
        self,
        prompt: str,
        sources: List[SearchResult],
        query: str,
    ) -> RAGResponse:
        """同步生成回答
        
        Args:
            prompt: 构建好的提示词
            sources: 检索结果
            query: 原始查询
            
        Returns:
            RAGResponse: 响应对象
        """
        try:
            response = await self.llm.generate(prompt)
            
            return RAGResponse(
                answer=response.content,
                sources=sources,
                model=response.model,
                query=query,
                usage=response.usage,
            )
            
        except Exception as e:
            raise PipelineError(f"LLM 生成失败: {str(e)}")
    
    async def _stream_response(
        self,
        prompt: str,
        sources: List[SearchResult],
        query: str,
    ) -> AsyncIterator[str]:
        """流式生成回答
        
        Args:
            prompt: 构建好的提示词
            sources: 检索结果
            query: 原始查询
            
        Yields:
            str: 生成的文本片段
        """
        try:
            async for chunk in self.llm.stream_generate(prompt):
                yield chunk
        except Exception as e:
            raise PipelineError(f"LLM 流式生成失败: {str(e)}")
    
    def _build_prompt(
        self,
        query: str,
        sources: List[SearchResult],
    ) -> str:
        """渲染 prompt 模板
        
        Args:
            query: 用户查询
            sources: 检索结果
            
        Returns:
            str: 渲染后的 prompt
        """
        template = Template(self.prompt_template)
        return template.render(query=query, sources=sources)
    
    def _build_context(self, sources: List[SearchResult]) -> str:
        """构建上下文文本
        
        Args:
            sources: 检索结果
            
        Returns:
            str: 上下文文本
        """
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[{i}] {source.content}")
        return "\n\n".join(context_parts)
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """向向量存储添加文档
        
        Args:
            documents: 文档列表，每项包含 id, content, metadata
            batch_size: 批处理大小
            
        Returns:
            int: 添加的文档数量
        """
        total = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 提取文本内容
            texts = [doc["content"] for doc in batch]
            
            # 批量向量化
            embeddings = self.embedder.embed_texts(texts)
            
            # 准备向量存储所需数据
            ids = [doc["id"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]
            
            # 添加到向量存储（需要向量存储实例支持）
            # 这里假设 retriever.vector_retriever.vector_store 有 add 方法
            if hasattr(self.retriever.vector_retriever, "vector_store"):
                vector_store = self.retriever.vector_retriever.vector_store
                if hasattr(vector_store, "add"):
                    vector_store.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                    )
                    total += len(batch)
                    logger.debug(f"已添加 {len(batch)} 条文档")
        
        logger.info(f"文档添加完成，总计 {total} 条")
        return total


class SimpleRAGPipeline:
    """简化版 RAG Pipeline
    
    用于快速测试和简单场景。
    """
    
    def __init__(
        self,
        vector_store: Any,
        llm_orchestrator: LLMOrchestrator,
        embedder: Any,
    ):
        """初始化简化版 RAG Pipeline
        
        Args:
            vector_store: 向量存储实例
            llm_orchestrator: LLM 调度器
            embedder: 嵌入模型
        """
        from retrieval.hybrid_retriever import create_hybrid_retriever
        
        # 创建检索器
        retriever = create_hybrid_retriever(
            vector_store=vector_store,
        )
        
        # 创建完整 Pipeline
        self.pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm_orchestrator,
            embedder=embedder,
        )
    
    async def chat(
        self,
        query: str,
        top_k: int = 5,
        stream: bool = False,
    ) -> RAGResponse | AsyncIterator[str]:
        """执行对话"""
        return await self.pipeline.chat(
            query=query,
            top_k=top_k,
            stream=stream,
        )


def create_rag_pipeline(
    vector_store: Any,
    embedder: Any,
    llm_provider: Optional[str] = None,
    model: Optional[str] = None,
) -> RAGPipeline:
    """创建 RAG Pipeline 的工厂函数

    Args:
        vector_store: 向量存储实例
        embedder: 嵌入模型实例
        llm_provider: LLM 提供商（默认从配置读取）
        model: 模型名称（默认从配置读取）

    Returns:
        RAGPipeline: RAG Pipeline 实例
    """
    from retrieval.hybrid_retriever import create_hybrid_retriever
    from llm.orchestrator import LLMOrchestrator

    # 使用配置中的默认值
    provider = llm_provider or settings.llm_provider

    # 创建检索器
    retriever = create_hybrid_retriever(
        vector_store=vector_store,
    )

    # 创建 LLM 调度器
    llm = LLMOrchestrator(
        provider=provider,
        model=model,
    )

    # 创建 Pipeline
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        embedder=embedder,
    )
