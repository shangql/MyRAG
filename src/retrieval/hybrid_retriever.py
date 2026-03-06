"""检索模块 - 包含混合检索器和查询处理器"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import settings
from core.exceptions import RetrievalError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """检索结果数据类
    
    用于封装单条检索结果，包含文本内容、相似度分数、
    元数据和原始数据源信息。
    
    Attributes:
        id: 文档唯一标识符
        content: 检索到的文本内容
        score: 相似度分数 (0-1)
        metadata: 元数据字典
        rank: 排名位置
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


class BaseRetriever:
    """检索器基类
    
    定义检索器的统一接口。
    """
    
    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """执行检索
        
        Args:
            query: 查询文本
            query_vector: 查询的向量表示
            top_k: 返回结果数量
            filter_condition: 过滤条件
            
        Returns:
            按相关性排序的检索结果列表
        """
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """向量检索器
    
    基于向量相似度进行文档检索。
    """
    
    def __init__(
        self,
        vector_store: Any,
        top_k: int = 5,
    ):
        """初始化向量检索器
        
        Args:
            vector_store: 向量存储实例
            top_k: 默认返回结果数量
        """
        self.vector_store = vector_store
        self.top_k = top_k
    
    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """执行向量相似度检索
        
        Args:
            query: 查询文本
            query_vector: 查询向量
            top_k: 返回结果数量
            filter_condition: 过滤条件
            
        Returns:
            检索结果列表
        """
        results = self.vector_store.search(
            query_embedding=query_vector,
            k=top_k,
            filter_condition=filter_condition,
        )
        
        search_results = []
        for rank, result in enumerate(results, 1):
            search_results.append(
                SearchResult(
                    id=result.get("id", ""),
                    content=result.get("document", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    rank=rank,
                )
            )
        
        logger.debug(f"向量检索返回 {len(search_results)} 条结果")
        return search_results


class KeywordRetriever(BaseRetriever):
    """关键词检索器
    
    基于 BM25 算法进行关键词检索。
    """
    
    def __init__(
        self,
        documents: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
    ):
        """初始化关键词检索器
        
        Args:
            documents: 文档列表，每项包含 id 和 content
            top_k: 默认返回结果数量
        """
        self.top_k = top_k
        self._index = None
        self._documents: Dict[str, Dict[str, str]] = {}
        
        if documents:
            self.build_index(documents)
    
    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """构建 BM25 索引
        
        Args:
            documents: 文档列表，每项包含 id 和 content
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 未安装，关键词检索将不可用")
            return
        
        # 保存文档
        for doc in documents:
            self._documents[doc["id"]] = doc
        
        # 分词
        tokenized_corpus = [doc["content"].split() for doc in documents]
        
        # 构建 BM25 索引
        self._index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成，包含 {len(documents)} 篇文档")
    
    def search(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """执行关键词检索
        
        Args:
            query: 查询文本
            query_vector: 保留参数，此检索器不使用
            top_k: 返回结果数量
            filter_condition: 保留参数
            
        Returns:
            检索结果列表
        """
        if self._index is None:
            return []
        
        # 分词查询
        tokenized_query = query.split()
        
        # 计算 BM25 分数
        scores = self._index.get_scores(tokenized_query)
        
        # 获取 Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                doc_id = list(self._documents.keys())[idx]
                results.append(
                    SearchResult(
                        id=doc_id,
                        content=self._documents[doc_id]["content"],
                        score=float(scores[idx]),
                        metadata=self._documents[doc_id].get("metadata", {}),
                        rank=rank,
                    )
                )
        
        logger.debug(f"关键词检索返回 {len(results)} 条结果")
        return results


class HybridRetriever(BaseRetriever):
    """混合检索器
    
    融合向量检索与关键词检索，通过 RRF (Reciprocal Rank Fusion) 
    算法对结果进行重排序，提升检索的准确性和召回率。
    
    Attributes:
        vector_retriever: 向量检索器
        keyword_retriever: 关键词检索器
        top_k: 返回结果数量
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
    """
    
    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        top_k: int = 5,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        """初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器实例
            keyword_retriever: 关键词检索器实例
            top_k: 返回结果数量
            vector_weight: 向量检索权重 (0-1)
            keyword_weight: 关键词检索权重 (0-1)
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # 验证权重
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            logger.warning(
                f"权重之和不为 1 ({vector_weight} + {keyword_weight})，将进行归一化"
            )
            total = vector_weight + keyword_weight
            self.vector_weight = vector_weight / total
            self.keyword_weight = keyword_weight / total
    
    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """执行混合检索
        
        流程：
        1. 并行执行向量检索和关键词检索
        2. 使用 RRF 算法融合结果
        3. 返回融合后的 Top-K 结果
        
        Args:
            query: 查询文本
            query_vector: 查询的向量表示
            top_k: 返回结果数量
            filter_condition: 过滤条件
            
        Returns:
            按相关性排序的检索结果列表
        """
        # 向量检索
        vector_results = []
        if self.vector_retriever:
            vector_results = self.vector_retriever.search(
                query=query,
                query_vector=query_vector,
                top_k=top_k * 2,
                filter_condition=filter_condition,
            )
        
        # 关键词检索
        keyword_results = []
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.search(
                query=query,
                query_vector=None,
                top_k=top_k * 2,
            )
        
        # RRF 融合
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
        )
        
        # 返回 Top-K
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        results_a: List[SearchResult],
        results_b: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """倒数排名融合算法 (RRF)
        
        RRF 公式: score = Σ(1 / (k + rank))
        其中 k 为平滑参数（通常设为 60），rank 为排名位置
        
        该算法优势：
        - 不需要校准不同检索器的分数
        - 对排名敏感而非分数敏感
        - 简单有效，广泛使用
        
        Args:
            results_a: 第一组检索结果（向量检索）
            results_b: 第二组检索结果（关键词检索）
            k: 平滑参数，默认 60
            
        Returns:
            融合后的排序结果
        """
        doc_scores: Dict[str, float] = {}
        
        # 构建文档映射
        all_docs: Dict[str, SearchResult] = {}
        for result in results_a:
            all_docs[result.id] = result
        for result in results_b:
            if result.id not in all_docs:
                all_docs[result.id] = result
        
        # 计算向量检索的 RRF 分数
        for rank, result in enumerate(results_a, 1):
            rrf_score = 1.0 / (k + rank)
            doc_scores[result.id] = doc_scores.get(result.id, 0) + rrf_score * self.vector_weight
        
        # 计算关键词检索的 RRF 分数
        for rank, result in enumerate(results_b, 1):
            rrf_score = 1.0 / (k + rank)
            doc_scores[result.id] = doc_scores.get(result.id, 0) + rrf_score * self.keyword_weight
        
        # 按分数排序
        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建结果列表
        fused_results = []
        for rank, (doc_id, score) in enumerate(sorted_ids, 1):
            if doc_id in all_docs:
                result = all_docs[doc_id]
                result.score = score
                result.rank = rank
                fused_results.append(result)
        
        logger.debug(
            f"RRF 融合完成 | 向量结果: {len(results_a)} | "
            f"关键词结果: {len(results_b)} | 融合结果: {len(fused_results)}"
        )
        return fused_results


def create_hybrid_retriever(
    vector_store: Any,
    documents: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> HybridRetriever:
    """创建混合检索器的工厂函数
    
    Args:
        vector_store: 向量存储实例
        documents: 文档列表（用于关键词检索）
        top_k: 返回结果数量
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
        
    Returns:
        HybridRetriever: 混合检索器实例
    """
    # 创建向量检索器
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        top_k=top_k,
    )
    
    # 创建关键词检索器
    keyword_retriever = None
    if documents:
        keyword_retriever = KeywordRetriever(
            documents=documents,
            top_k=top_k,
        )
    
    return HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
