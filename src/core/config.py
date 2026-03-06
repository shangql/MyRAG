"""配置管理模块 - 使用 Pydantic Settings 进行类型安全的配置管理"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类

    从环境变量加载配置，支持默认值。配置优先级：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 项目根目录
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # 数据库配置
    database_url: str = "mysql+pymysql://user:password@localhost:3306/rag_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # 向量数据库配置
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    persist_directory: str = "./data/vector_store"

    # 嵌入模型配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda"] = "cpu"
    embedding_batch_size: int = 32

    # LLM 配置
    llm_provider: Literal["openai", "anthropic", "ollama", "modelscope"] = "openai"
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-3.5-turbo"
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-3-sonnet-20240229"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    modelscope_api_key: Optional[str] = Field(default=None, validation_alias="MODELSCOPE_API_KEY")
    modelscope_model: str = "qwen-turbo"
    modelscope_base_url: str = "https://api-inference.modelscope.cn/v1"

    # LLM 生成参数
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # 检索配置
    top_k: int = 5
    similarity_threshold: float = 0.7
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 1

    # CORS 配置
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Redis 配置 (可选)
    redis_url: Optional[str] = None
    redis_ttl: int = 3600

    # 日志配置
    log_level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    log_file: Optional[str] = "./logs/app.log"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例

    使用 lru_cache 缓存配置实例，避免重复读取环境变量

    Returns:
        Settings: 应用配置实例
    """
    return Settings()


# 全局配置实例
settings = get_settings()
