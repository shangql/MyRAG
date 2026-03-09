"""API 数据模型 - Pydantic 请求/响应模型"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., description="用户查询文本")
    top_k: int = Field(5, description="检索结果数量")
    stream: bool = Field(False, description="是否流式输出")
    model: Optional[str] = Field(None, description="指定模型名称")
    provider: Optional[str] = Field(None, description="LLM 提供商 (openai/anthropic/ollama/modelscope)")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    query: str


class SwitchModelRequest(BaseModel):
    """切换模型请求"""
    provider: str
    model: str


class SwitchModelResponse(BaseModel):
    """切换模型响应"""
    status: str
    model: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
