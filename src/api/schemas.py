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


class ImportRequest(BaseModel):
    """数据导入请求"""
    table_name: str = Field(..., description="数据库表名")
    content_column: str = Field("content", description="内容列名")
    id_column: str = Field("id", description="ID列名")
    metadata_columns: Optional[List[str]] = Field(None, description="元数据列名")
    batch_size: int = Field(100, description="批处理大小")
    filter_condition: Optional[str] = Field(None, description="SQL WHERE条件")


class ImportResponse(BaseModel):
    """数据导入响应"""
    status: str
    imported: int
    failed: int
    total: int
    message: str


class TextAddRequest(BaseModel):
    """添加文本请求"""
    texts: List[str] = Field(..., description="文本列表")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="元数据列表")


class TextAddResponse(BaseModel):
    """添加文本响应"""
    status: str
    count: int
    ids: List[str]


class VectorStatsResponse(BaseModel):
    """向量库统计"""
    total_count: int
    collections: List[Dict[str, Any]]
