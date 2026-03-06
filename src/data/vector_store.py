"""向量存储管理模块 - 提供统一的向量存储和检索接口"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import settings
from core.exceptions import VectorStoreError
from core.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """向量存储基类
    
    提供统一的向量存储和检索接口，支持 ChromaDB 和 FAISS 两种后端。
    
    Attributes:
        collection_name: 集合名称
        embedding_dim: 向量维度
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        embedding_dim: Optional[int] = None,
    ):
        """初始化向量存储
        
        Args:
            collection_name: 集合名称
            embedding_dim: 向量维度（可选，某些后端需要）
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client = None
    
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """添加向量到存储
        
        Args:
            ids: 文档 ID 列表
            embeddings: 向量数组
            documents: 原始文档文本列表
            metadatas: 元数据列表
        """
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """相似度搜索
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数量
            filter_condition: 过滤条件
            
        Returns:
            检索结果列表，每项包含 id, score, document, metadata
        """
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> None:
        """删除指定 ID 的向量
        
        Args:
            ids: 要删除的 ID 列表
        """
        raise NotImplementedError
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """更新向量
        
        Args:
            ids: 文档 ID 列表
            embeddings: 新的向量
            documents: 新的文档文本
            metadatas: 新的元数据
        """
        raise NotImplementedError
    
    def get_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """根据 ID 获取向量
        
        Args:
            ids: 文档 ID 列表
            
        Returns:
            向量数据列表
        """
        raise NotImplementedError
    
    def count(self) -> int:
        """获取向量总数
        
        Returns:
            int: 向量数量
        """
        raise NotImplementedError
    
    def reset(self) -> None:
        """清空所有向量"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储实现
    
    使用 ChromaDB 作为向量数据库后端。
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        embedding_function: Optional[Any] = None,
        persist_directory: Optional[str] = None,
    ):
        """初始化 ChromaDB 向量存储
        
        Args:
            collection_name: 集合名称
            embedding_function: 嵌入函数（可选）
            persist_directory: 持久化目录
        """
        super().__init__(collection_name=collection_name)
        
        self.persist_directory = persist_directory or settings.persist_directory
        self.embedding_function = embedding_function
        
        # 延迟导入，避免未安装时报错
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise VectorStoreError(
                "ChromaDB 未安装，请运行: pip install chromadb",
                operation="import",
            )
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """初始化 ChromaDB 客户端"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # 创建持久化客户端
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
            
            logger.info(f"ChromaDB 集合初始化完成: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(
                f"ChromaDB 初始化失败: {str(e)}",
                operation="init",
            )
    
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """添加向量到 ChromaDB"""
        try:
            # 转换 numpy 数组为列表
            embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
            self._collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(f"添加 {len(ids)} 条向量到 ChromaDB")
            
        except Exception as e:
            raise VectorStoreError(
                f"添加向量失败: {str(e)}",
                operation="add",
            )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        try:
            query_embedding = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_condition,
                include=["documents", "metadatas", "distances"],
            )
            
            # 格式化结果
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "id": doc_id,
                        "score": 1 - results["distances"][0][i],  # 转换为相似度
                        "document": results["documents"][0][i] if results["documents"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                    })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(
                f"搜索失败: {str(e)}",
                operation="search",
            )
    
    def delete(self, ids: List[str]) -> None:
        """删除向量"""
        try:
            self._collection.delete(ids=ids)
            logger.debug(f"删除 {len(ids)} 条向量")
        except Exception as e:
            raise VectorStoreError(
                f"删除向量失败: {str(e)}",
                operation="delete",
            )
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """更新向量"""
        try:
            embeddings_list = embeddings.tolist() if embeddings is not None else None
            
            self._collection.update(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug(f"更新 {len(ids)} 条向量")
            
        except Exception as e:
            raise VectorStoreError(
                f"更新向量失败: {str(e)}",
                operation="update",
            )
    
    def get_by_id(self, ids: List[str]) -> List[Dict[str, Any]]:
        """根据 ID 获取向量"""
        try:
            results = self._collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"],
            )
            
            formatted_results = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    formatted_results.append({
                        "id": doc_id,
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "embedding": results["embeddings"][i],
                    })
            
            return formatted_results
            
        except Exception as e:
            raise VectorStoreError(
                f"获取向量失败: {str(e)}",
                operation="get",
            )
    
    def count(self) -> int:
        """获取向量总数"""
        return self._collection.count()
    
    def reset(self) -> None:
        """清空所有向量"""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
        )
        logger.info(f"集合 {self.collection_name} 已清空")


class FAISSVectorStore(VectorStore):
    """FAISS 向量存储实现
    
    使用 Facebook FAISS 作为向量检索后端。
    适用于需要高性能向量检索的场景。
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IVF_FLAT",
        metric: str = "cosine",
    ):
        """初始化 FAISS 向量存储
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 (FLAT/IVF_FLAT/HNSW)
            metric: 距离度量方式 (cosine/l2/ip)
        """
        super().__init__(embedding_dim=dimension)
        
        self.index_type = index_type
        self.metric = metric
        
        # 延迟导入
        try:
            import faiss
        except ImportError:
            raise VectorStoreError(
                "FAISS 未安装，请运行: pip install faiss-cpu",
                operation="import",
            )
        
        self._index = None
        self._id_to_doc: Dict[str, Dict[str, Any]] = {}
        self._doc_to_id: Dict[str, str] = {}
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """初始化 FAISS 索引"""
        import faiss
        
        if self.metric == "cosine":
            # 余弦相似度需要先归一化，使用内积代替
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
        
        logger.info(f"FAISS 索引初始化完成 | 维度: {self.embedding_dim} | 类型: {self.index_type}")
    
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """添加向量到 FAISS 索引"""
        import faiss
        
        try:
            # 确保向量是 float32 类型
            embeddings = embeddings.astype("float32")
            
            # 归一化（用于余弦相似度）
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)
            
            # 添加到索引
            self._index.add(embeddings)
            
            # 保存文档映射
            for i, doc_id in enumerate(ids):
                self._id_to_doc[doc_id] = {
                    "document": documents[i] if documents else None,
                    "metadata": metadatas[i] if metadatas else None,
                }
            
            logger.debug(f"添加 {len(ids)} 条向量到 FAISS 索引")
            
        except Exception as e:
            raise VectorStoreError(
                f"添加向量失败: {str(e)}",
                operation="add",
            )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_condition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """相似度搜索"""
        import faiss
        
        try:
            query_embedding = query_embedding.astype("float32").reshape(1, -1)
            
            if self.metric == "cosine":
                faiss.normalize_L2(query_embedding)
            
            # 搜索
            scores, indices = self._index.search(query_embedding, k)
            
            # 格式化结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # 有效索引
                    doc_id = f"doc_{idx}"
                    if doc_id in self._id_to_doc:
                        results.append({
                            "id": doc_id,
                            "score": float(score),
                            "document": self._id_to_doc[doc_id].get("document"),
                            "metadata": self._id_to_doc[doc_id].get("metadata"),
                        })
            
            return results
            
        except Exception as e:
            raise VectorStoreError(
                f"搜索失败: {str(e)}",
                operation="search",
            )
    
    def delete(self, ids: List[str]) -> None:
        """删除向量（FAISS 不支持直接删除，需要重建索引）"""
        logger.warning("FAISS 不支持直接删除向量，建议使用标记删除或重建索引")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """更新向量"""
        logger.warning("FAISS 不支持直接更新向量，建议重建索引")
    
    def count(self) -> int:
        """获取向量总数"""
        return self._index.ntotal
    
    def reset(self) -> None:
        """清空所有向量"""
        import faiss
        
        self._initialize_index()
        self._id_to_doc.clear()
        logger.info("FAISS 索引已清空")


def get_vector_store(
    store_type: Optional[str] = None,
    **kwargs,
) -> VectorStore:
    """获取向量存储实例的工厂函数
    
    Args:
        store_type: 存储类型 (chroma/faiss)，默认使用配置
        **kwargs: 传递给向量存储的额外参数
        
    Returns:
        VectorStore: 向量存储实例
    """
    store_type = store_type or settings.vector_store_type
    
    if store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise VectorStoreError(
            f"不支持的向量存储类型: {store_type}",
            operation="init",
        )
