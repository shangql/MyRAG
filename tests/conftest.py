"""Pytest 配置文件 - 测试夹具和公共配置"""
import sys
from pathlib import Path
from typing import Generator

import pytest

# 将 src 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_documents() -> list[dict]:
    """示例文档数据
    
    Returns:
        list: 包含 id, content, metadata 的文档列表
    """
    return [
        {
            "id": "doc1",
            "content": "Python 是一种高级编程语言，",
            "metadata": {"source": "wiki", "category": "programming"},
        },
        {
            "id": "doc2",
            "content": "机器学习是人工智能的一个分支，",
            "metadata": {"source": "wiki", "category": "ai"},
        },
        {
            "id": "doc3",
            "content": "深度学习是机器学习的子领域，",
            "metadata": {"source": "wiki", "category": "ai"},
        },
        {
            "id": "doc4",
            "content": "自然语言处理用于处理文本数据，",
            "metadata": {"source": "wiki", "category": "nlp"},
        },
        {
            "id": "doc5",
            "content": "向量数据库用于存储高维向量，",
            "metadata": {"source": "wiki", "category": "database"},
        },
    ]


@pytest.fixture
def sample_query() -> str:
    """示例查询文本
    
    Returns:
        str: 查询文本
    """
    return "什么是机器学习？"


@pytest.fixture
def sample_query_vector() -> list[float]:
    """示例查询向量
    
    Returns:
        list: 查询向量（模拟）
    """
    import numpy as np
    return np.random.randn(384).tolist()


@pytest.fixture
def mock_embedder():
    """模拟嵌入模型
    
    Returns:
        Mock 嵌入模型对象
    """
    import numpy as np
    
    class MockEmbedder:
        def __init__(self):
            self.embedding_dim = 384
        
        def embed_texts(self, texts):
            """返回随机向量"""
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        def embed_query(self, query):
            """返回单个随机向量"""
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    return MockEmbedder()


@pytest.fixture
def mock_vector_store(sample_documents, mock_embedder):
    """模拟向量存储
    
    Args:
        sample_documents: 示例文档
        mock_embedder: 模拟嵌入模型
        
    Returns:
        Mock 向量存储对象
    """
    import numpy as np
    
    class MockVectorStore:
        def __init__(self):
            self.documents = {}
            self.embeddings = {}
        
        def add(self, ids, embeddings, documents, metadatas):
            """添加文档"""
            for i, doc_id in enumerate(ids):
                self.documents[doc_id] = {
                    "content": documents[i],
                    "metadata": metadatas[i] if metadatas else {},
                }
                self.embeddings[doc_id] = embeddings[i]
        
        def search(self, query_embedding, k=5, filter_condition=None):
            """模拟搜索返回随机结果"""
            import random
            
            results = []
            doc_items = list(self.documents.items())
            
            for i in range(min(k, len(doc_items))):
                if doc_items:
                    doc_id, doc = random.choice(doc_items)
                    results.append({
                        "id": doc_id,
                        "score": random.uniform(0.5, 1.0),
                        "document": doc["content"],
                        "metadata": doc["metadata"],
                    })
            
            return results
        
        def delete(self, ids):
            """删除文档"""
            for doc_id in ids:
                self.documents.pop(doc_id, None)
                self.embeddings.pop(doc_id, None)
        
        def count(self):
            """返回文档数量"""
            return len(self.documents)
    
    store = MockVectorStore()
    
    # 添加示例文档
    if sample_documents:
        ids = [doc["id"] for doc in sample_documents]
        texts = [doc["content"] for doc in sample_documents]
        metadatas = [doc["metadata"] for doc in sample_documents]
        embeddings = mock_embedder.embed_texts(texts)
        store.add(ids, embeddings, texts, metadatas)
    
    return store
