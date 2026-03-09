"""嵌入模型管理模块 - 负责将文本转换为向量表示"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.exceptions import ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)

# torch 为可选依赖，用于设备检测
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


class EmbedderConfig(BaseModel):
    """嵌入模型配置类

    使用 Pydantic 进行参数验证，确保配置的正确性。

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备 (cpu/cuda/mps)
        batch_size: 批处理大小
    """

    model_name: str = Field(
        default_factory=lambda: settings.embedding_model,
        description="HuggingFace 模型名称",
    )
    device: str = Field(
        default="auto",
        description="运行设备 (cpu/cuda/mps/auto)",
    )
    batch_size: int = Field(
        default_factory=lambda: settings.embedding_batch_size,
        ge=1,
        le=256,
        description="批处理大小，范围 1-256",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
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

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """验证批处理大小

        Args:
            v: 批处理大小

        Returns:
            验证后的批处理大小

        Raises:
            ValueError: 批处理大小超出范围时抛出
        """
        if v < 1 or v > 256:
            raise ValueError(f"批处理大小必须在 1-256 之间，当前值: {v}")
        return v


class Embedder:
    """嵌入模型封装类

    使用 sentence-transformers 库加载预训练模型，
    将文本转换为高维向量用于相似度计算。

    Attributes:
        model_name: HuggingFace 模型名称
        device: 运行设备 (cpu/cuda/mps)
        batch_size: 批处理大小
    """

    _instance: Optional["Embedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """初始化嵌入模型

        Args:
            model_name: HuggingFace 模型名称，默认使用配置
            device: 运行环境 (cpu/cuda/mps/auto)，默认自动检测
            batch_size: 批处理大小，默认使用配置

        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        # 使用 Pydantic 进行参数验证
        config = EmbedderConfig(
            model_name=model_name or settings.embedding_model,
            device=device or "auto",
            batch_size=batch_size or settings.embedding_batch_size,
        )

        self.model_name = config.model_name
        self.device = config.device
        self.batch_size = config.batch_size

        # 单例模式：避免重复加载模型
        if Embedder._model is None:
            self._load_model()
        else:
            self._model = Embedder._model

        self._embedding_dim: Optional[int] = None

    def _load_model(self) -> None:
        """加载预训练嵌入模型

        Raises:
            ModelLoadError: 模型加载失败时抛出
        """
        try:
            logger.info(f"加载嵌入模型: {self.model_name} | 设备: {self.device}")
            Embedder._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("嵌入模型加载完成")
        except Exception as e:
            raise ModelLoadError(
                f"无法加载嵌入模型 '{self.model_name}': {str(e)}",
                model_name=self.model_name,
            )

    @property
    def embedding_dim(self) -> int:
        """获取向量维度

        Returns:
            int: 嵌入向量的维度
        """
        if self._embedding_dim is None:
            self._embedding_dim = Embedder._model.get_sentence_embedding_dimension()
        return self._embedding_dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量列表

        Args:
            texts: 待编码的文本列表

        Returns:
            numpy 数组，形状为 (len(texts), embedding_dim)

        Raises:
            ValueError: 当文本列表为空时抛出
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        embeddings = Embedder._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 归一化，提升相似度计算效率
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """将单个查询转换为向量

        Args:
            query: 查询文本

        Returns:
            向量数组
        """
        return self.embed_texts([query])[0]

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """批量编码查询（embed_query 的别名）

        Args:
            queries: 查询文本列表

        Returns:
            numpy 数组
        """
        return self.embed_texts(queries)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """批量编码文档（embed_texts 的别名）

        Args:
            documents: 文档文本列表

        Returns:
            numpy 数组
        """
        return self.embed_texts(documents)


def get_embedder() -> Embedder:
    """获取嵌入模型单例

    Returns:
        Embedder: 嵌入模型实例
    """
    if Embedder._instance is None:
        Embedder._instance = Embedder()
    return Embedder._instance
