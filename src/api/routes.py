"""API 路由 - FastAPI 路由定义"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SwitchModelRequest,
    SwitchModelResponse,
    ImportRequest,
    ImportResponse,
    TextAddRequest,
    TextAddResponse,
    VectorStatsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["RAG"])

# 全局 pipeline 变量（实际应使用依赖注入）
_rag_pipeline = None


def set_rag_pipeline(pipeline):
    """设置 RAG Pipeline 实例"""
    global _rag_pipeline
    _rag_pipeline = pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 聊天接口

    接收用户查询，返回检索增强后的回答
    """
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    # 切换模型（如果提供了 provider 或 model）
    if request.provider or request.model:
        try:
            _rag_pipeline.llm.switch_model(
                provider=request.provider,
                model=request.model,
            )
        except Exception as e:
            logger.warning(f"模型切换失败: {e}")

    try:
        response = await _rag_pipeline.chat(
            query=request.query,
            top_k=request.top_k,
            stream=request.stream,
        )
        
        # 序列化来源
        sources = [
            {
                "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                "score": s.score,
                "metadata": s.metadata,
            }
            for s in response.sources
        ]
        
        return ChatResponse(
            answer=response.answer,
            sources=sources,
            model=response.model,
            query=response.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    # 切换模型（如果提供了 provider 或 model）
    if request.provider or request.model:
        try:
            _rag_pipeline.llm.switch_model(
                provider=request.provider,
                model=request.model,
            )
        except Exception as e:
            logger.warning(f"模型切换失败: {e}")

    async def event_generator():
        try:
            async for chunk in await _rag_pipeline.chat(
                query=request.query,
                top_k=request.top_k,
                stream=True,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: error: {str(e)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/model/switch", response_model=SwitchModelResponse)
async def switch_model(request: SwitchModelRequest):
    """切换 LLM 模型"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")
    
    _rag_pipeline.llm.switch_model(provider=request.provider, model=request.model)
    return SwitchModelResponse(status="success", model=request.model)


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口"""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/models")
async def list_models():
    """获取可用模型列表"""
    models = [
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek-R1"},
        {"id": "Qwen/Qwen3-8B", "name": "Qwen3-8B"},
        {"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B"},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B"},
    ]
    return {"models": models}


# ========== 向量库管理接口 ==========

@router.get("/vectors/stats", response_model=VectorStatsResponse)
async def get_vector_stats():
    """获取向量库统计信息"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        count = _rag_pipeline.vector_store.count()
        return VectorStatsResponse(
            total_count=count,
            collections=[{"name": "default", "count": count}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/vectors/add", response_model=TextAddResponse)
async def add_texts(request: TextAddRequest):
    """直接添加文本到向量库"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        import uuid
        ids = [str(uuid.uuid4()) for _ in request.texts]
        # 确保 metadata 是字典列表
        if request.metadata:
            metadatas = request.metadata
        else:
            metadatas = [{}] * len(request.texts)

        # 使用 embedder 生成向量
        embeddings = _rag_pipeline.embedder.embed_texts(request.texts)

        # 获取 vector_store 并添加
        vector_store = _rag_pipeline.retriever.vector_retriever.vector_store
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=request.texts,
            metadatas=metadatas,
        )

        return TextAddResponse(
            status="success",
            count=len(request.texts),
            ids=ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文本失败: {str(e)}")


@router.post("/vectors/import", response_model=ImportResponse)
async def import_from_db(request: ImportRequest):
    """从MySQL数据库导入数据到向量库"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        from data.importer import DataImporter, ImportConfig
        from data import get_embedder, get_vector_store

        # 创建导入器
        importer = DataImporter(
            db_manager=_rag_pipeline.db_manager,
            embedder=_rag_pipeline.embedder,
            vector_store=_rag_pipeline.vector_store,
        )

        # 配置导入
        config = ImportConfig(
            table_name=request.table_name,
            content_column=request.content_column,
            id_column=request.id_column,
            metadata_columns=request.metadata_columns,
            batch_size=request.batch_size,
        )

        # 执行导入
        result = importer.import_table(
            config=config,
            filter_condition=request.filter_condition,
        )

        return ImportResponse(
            status="success",
            imported=result.get("imported", 0),
            failed=result.get("failed", 0),
            total=result.get("total", 0),
            message=f"成功导入 {result.get('imported', 0)} 条记录"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")
