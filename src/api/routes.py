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
