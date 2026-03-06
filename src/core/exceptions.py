"""自定义异常模块 - 定义系统中使用的各类异常"""
from typing import Any, Optional


class RAGError(Exception):
    """RAG 系统基础异常类
    
    所有自定义异常的基类，提供统一的异常处理方式。
    """
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        """初始化异常
        
        Args:
            message: 异常消息
            details: 额外的详细信息字典
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """返回异常字符串表示"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} | {details_str}"
        return self.message


# ========== 配置相关异常 ==========

class ConfigError(RAGError):
    """配置错误异常"""
    pass


# ========== 数据层异常 ==========

class DataError(RAGError):
    """数据层基础异常"""
    pass


class DatabaseError(DataError):
    """数据库操作错误
    
    Attributes:
        operation: 失败的数据库操作 (connect/query/insert/update/delete)
        original_error: 原始数据库错误
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if original_error:
            details["original_error"] = str(original_error)
        
        super().__init__(message, details)
        self.operation = operation
        self.original_error = original_error


class VectorStoreError(DataError):
    """向量存储操作错误
    
    Attributes:
        operation: 失败的操作 (init/search/add/delete)
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
    ):
        details = {"operation": operation} if operation else {}
        super().__init__(message, details)
        self.operation = operation


# ========== 嵌入模型异常 ==========

class EmbeddingError(DataError):
    """嵌入模型相关错误
    
    Attributes:
        model_name: 失败的模型名称
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
    ):
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, details)
        self.model_name = model_name


class ModelLoadError(EmbeddingError):
    """模型加载失败异常"""
    pass


# ========== 检索模块异常 ==========

class RetrievalError(RAGError):
    """检索模块基础异常"""
    pass


class QueryProcessingError(RetrievalError):
    """查询处理错误
    
    Attributes:
        query: 原始查询文本
    """
    
    def __init__(self, message: str, query: Optional[str] = None):
        details = {"query": query} if query else {}
        super().__init__(message, details)
        self.query = query


# ========== LLM 模块异常 ==========

class LLMError(RAGError):
    """LLM 模块基础异常"""
    pass


class LLMAPIError(LLMError):
    """LLM API 调用错误
    
    Attributes:
        provider: LLM 提供商名称
        status_code: HTTP 状态码
    """
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        details = {}
        if provider:
            details["provider"] = provider
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMAPIError):
    """LLM API 速率限制异常"""
    pass


class LLMResponseError(LLMError):
    """LLM 响应解析错误"""
    pass


# ========== API 相关异常 ==========

class APIError(RAGError):
    """API 层基础异常"""
    pass


class ValidationError(APIError):
    """请求验证错误
    
    Attributes:
        field: 验证失败的字段名
    """
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(message, details)
        self.field = field


class AuthenticationError(APIError):
    """认证错误"""
    pass


class AuthorizationError(APIError):
    """授权错误"""
    pass


# ========== RAG Pipeline 异常 ==========

class PipelineError(RAGError):
    """RAG Pipeline 执行错误"""
    pass


class ContextBuilderError(PipelineError):
    """上下文构建错误"""
    pass
