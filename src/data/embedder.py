"""嵌入模型管理模块 - 负责将文本转换为向量表示"""

import hashlib
import os
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from core.config import settings
from core.exceptions import ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)

# 尝试加载 sentence-transformers，如果失败则使用 fallback
_TORCH_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_SentenceTransformer = None

try:
    import torch

    _TORCH_AVAILABLE = True
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info(f"PyTorch 版本: {torch.__version__}, sentence-transformers 已加载")
except Exception as e:
    logger.warning(f"PyTorch/sentence-transformers 加载失败: {e}，将使用简单哈希向量")
    _SentenceTransformer = None
    torch = None  # type: ignore


class EmbedderConfig(BaseModel):
    """嵌入模型配置类

    使用 Pydantic 进行参数验证，确保配置的正确性。

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备 (cpu/cuda/mps)
        batch_size: 批处理大小
        max_length: 最大序列长度
        normalize: 是否归一化向量
    """

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace 模型名称",
    )
    device: str = Field(
        default="auto",
        description="运行设备 (cpu/cuda/mps/auto)",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="批处理大小",
    )
    max_length: int = Field(
        default=384,
        ge=64,
        le=2048,
        description="最大序列长度",
    )
    normalize: bool = Field(
        default=True,
        description="是否归一化向量",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """验证设备参数并自动检测可用设备"""
        if v == "auto":
            if _TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            return "cpu"

        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"不支持的设备 '{v}'，可选值: {valid_devices}")
        return v


class Embedder:
    """嵌入模型封装类

    将文本转换为向量表示，支持批量处理。

    Attributes:
        config: 嵌入模型配置
        model: SentenceTransformer 模型实例
        embedding_dim: 向量维度
    """

    def __init__(self, config: Optional[EmbedderConfig] = None):
        """初始化嵌入模型

        Args:
            config: 嵌入模型配置，默认使用 settings 中的配置

        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        self.config = config or self._create_config_from_settings()
        self.model = None
        self.embedding_dim: int = 384  # 默认维度

        # 尝试加载模型
        if _SENTENCE_TRANSFORMERS_AVAILABLE and _SentenceTransformer is not None:
            try:
                self.model = _SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                )
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(
                    f"嵌入模型加载成功: {self.config.model_name} | "
                    f"设备: {self.config.device} | 向量维度: {self.embedding_dim}"
                )
            except Exception as e:
                logger.warning(f"嵌入模型加载失败: {e}，使用简单哈希向量")
                self.model = None
        else:
            logger.info("使用简单哈希向量作为 fallback")

    def _create_config_from_settings(self) -> EmbedderConfig:
        """从 settings 创建配置"""
        return EmbedderConfig(
            model_name=settings.embedding_model,
            device="cpu",  # fallback 模式
            batch_size=settings.embedding_batch_size,
            max_length=settings.embedding_max_length,
            normalize=True,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """将单个文本转换为向量

        Args:
            text: 输入文本

        Returns:
            文本的向量表示
        """
        if self.model is not None:
            embedding = self.model.encode(
                text,
                batch_size=1,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embedding

        # Fallback: 使用文本哈希生成固定向量
        return self._hash_embedding(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将多个文本转换为向量

        Args:
            texts: 输入文本列表

        Returns:
            二维 numpy 数组，每行是一个文本的向量表示
        """
        if self.model is not None and texts:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
            )
            return embeddings

        # Fallback: 使用文本哈希生成固定向量
        return np.array([self._hash_embedding(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """将查询文本转换为向量（兼容 LangChain 接口）

        Args:
            text: 查询文本

        Returns:
            查询文本的向量表示
        """
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """将多个文档转换为向量（兼容 LangChain 接口）

        Args:
            texts: 文档列表

        Returns:
            二维 numpy 数组，每行是一个文档的向量表示
        """
        return self.embed_texts(texts)

    def _hash_embedding(self, text: str) -> np.ndarray:
        """使用哈希生成固定向量（fallback 方案）

        将文本的 SHA256 哈希值转换为固定维度的向量。
        这是一个简单的 fallback，用于在模型不可用时提供基本功能。

        Args:
            text: 输入文本

        Returns:
            固定维度的向量
        """
        # 生成 SHA256 哈希
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        # 将哈希转换为固定长度的数值序列
        vectors = []
        for i in range(0, min(len(hash_hex), self.embedding_dim * 2), 2):
            # 每两个十六进制字符转换为一个 0-255 的值
            value = int(hash_hex[i : i + 2], 16)
            vectors.append(value)

        # 填充或截断到固定维度
        while len(vectors) < self.embedding_dim:
            vectors.append(0)

        # 归一化
        vector = np.array(vectors[: self.embedding_dim], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.embedding_dim


# 全局单例
_embedder: Optional[Embedder] = None


def get_embedder(config: Optional[EmbedderConfig] = None) -> Embedder:
    """获取全局嵌入模型实例（单例模式）

    Args:
        config: 嵌入模型配置

    Returns:
        Embedder 实例
    """
    global _embedder

    if _embedder is None:
        _embedder = Embedder(config)

    return _embedder


def reset_embedder() -> None:
    """重置嵌入模型实例（用于测试或重新加载模型）"""
    global _embedder
    _embedder = None
