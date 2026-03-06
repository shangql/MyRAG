"""检索模块测试"""
import numpy as np
import pytest

from retrieval.hybrid_retriever import (
    BaseRetriever,
    HybridRetriever,
    KeywordRetriever,
    SearchResult,
    VectorRetriever,
    create_hybrid_retriever,
)


class TestSearchResult:
    """SearchResult 数据类测试"""
    
    def test_creation(self):
        """测试创建"""
        result = SearchResult(
            id="test1",
            content="测试内容",
            score=0.95,
            metadata={"source": "test"},
            rank=1,
        )
        
        assert result.id == "test1"
        assert result.content == "测试内容"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}
        assert result.rank == 1


class TestVectorRetriever:
    """向量检索器测试"""
    
    def test_initialization(self, mock_vector_store):
        """测试初始化"""
        retriever = VectorRetriever(vector_store=mock_vector_store, top_k=5)
        
        assert retriever.vector_store == mock_vector_store
        assert retriever.top_k == 5
    
    def test_search(self, mock_vector_store, sample_query, sample_query_vector):
        """测试搜索功能"""
        retriever = VectorRetriever(vector_store=mock_vector_store, top_k=3)
        
        results = retriever.search(
            query=sample_query,
            query_vector=np.array(sample_query_vector),
            top_k=3,
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, SearchResult)


class TestKeywordRetriever:
    """关键词检索器测试"""
    
    def test_initialization(self, sample_documents):
        """测试初始化和索引构建"""
        retriever = KeywordRetriever(documents=sample_documents, top_k=5)
        
        assert retriever.top_k == 5
        assert retriever._index is not None
    
    def test_search(self, sample_documents, sample_query):
        """测试搜索功能"""
        retriever = KeywordRetriever(documents=sample_documents, top_k=3)
        
        results = retriever.search(
            query=sample_query,
            query_vector=None,
            top_k=3,
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_empty_index(self):
        """测试空索引"""
        retriever = KeywordRetriever()
        
        results = retriever.search(query="test", query_vector=None)
        
        assert results == []


class TestHybridRetriever:
    """混合检索器测试"""
    
    def test_initialization(self, mock_vector_store, sample_documents):
        """测试初始化"""
        vector_retriever = VectorRetriever(vector_store=mock_vector_store)
        keyword_retriever = KeywordRetriever(documents=sample_documents)
        
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            top_k=5,
            vector_weight=0.6,
            keyword_weight=0.4,
        )
        
        assert retriever.vector_retriever == vector_retriever
        assert retriever.keyword_retriever == keyword_retriever
        assert retriever.top_k == 5
        assert retriever.vector_weight == 0.6
        assert retriever.keyword_weight == 0.4
    
    def test_weight_normalization(self, mock_vector_store):
        """测试权重归一化"""
        retriever = HybridRetriever(
            vector_retriever=VectorRetriever(vector_store=mock_vector_store),
            keyword_retriever=None,
            top_k=5,
            vector_weight=0.6,
            keyword_weight=0.4,
        )
        
        # 验证权重已被归一化
        assert abs(retriever.vector_weight + retriever.keyword_weight - 1.0) < 0.01
    
    def test_search_vector_only(self, mock_vector_store, sample_query, sample_query_vector):
        """测试仅向量检索"""
        retriever = HybridRetriever(
            vector_retriever=VectorRetriever(vector_store=mock_vector_store),
            keyword_retriever=None,
            top_k=3,
        )
        
        results = retriever.search(
            query=sample_query,
            query_vector=np.array(sample_query_vector),
            top_k=3,
        )
        
        assert isinstance(results, list)
    
    def test_search_with_both(self, mock_vector_store, sample_documents, sample_query, sample_query_vector):
        """测试向量+关键词混合检索"""
        vector_retriever = VectorRetriever(vector_store=mock_vector_store)
        keyword_retriever = KeywordRetriever(documents=sample_documents)
        
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            top_k=3,
        )
        
        results = retriever.search(
            query=sample_query,
            query_vector=np.array(sample_query_vector),
            top_k=3,
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # 验证结果已排序
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestRRF:
    """RRF 算法测试"""
    
    def test_rrf_fusion(self, mock_vector_store, sample_query_vector):
        """测试 RRF 融合"""
        vector_retriever = VectorRetriever(vector_store=mock_vector_store)
        
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            keyword_retriever=None,
            top_k=5,
        )
        
        # 创建测试结果
        results_a = [
            SearchResult(id="1", content="内容1", score=0.9, rank=1),
            SearchResult(id="2", content="内容2", score=0.8, rank=2),
        ]
        results_b = [
            SearchResult(id="2", content="内容2", score=0.85, rank=1),
            SearchResult(id="3", content="内容3", score=0.75, rank=2),
        ]
        
        fused = retriever._reciprocal_rank_fusion(results_a, results_b, k=60)
        
        # 验证融合结果
        assert len(fused) > 0
        assert fused[0].id in ["1", "2", "3"]
    
    def test_empty_results(self, mock_vector_store):
        """测试空结果"""
        retriever = HybridRetriever(
            vector_retriever=VectorRetriever(vector_store=mock_vector_store),
            keyword_retriever=None,
            top_k=5,
        )
        
        fused = retriever._reciprocal_rank_fusion([], [], k=60)
        
        assert fused == []


class TestCreateHybridRetriever:
    """create_hybrid_retriever 工厂函数测试"""
    
    def test_create_with_vector_only(self, mock_vector_store):
        """测试仅创建向量检索器"""
        retriever = create_hybrid_retriever(
            vector_store=mock_vector_store,
            documents=None,
            top_k=5,
        )
        
        assert isinstance(retriever, HybridRetriever)
        assert retriever.vector_retriever is not None
        assert retriever.keyword_retriever is None
    
    def test_create_with_documents(self, mock_vector_store, sample_documents):
        """测试创建包含关键词检索的混合检索器"""
        retriever = create_hybrid_retriever(
            vector_store=mock_vector_store,
            documents=sample_documents,
            top_k=5,
        )
        
        assert isinstance(retriever, HybridRetriever)
        assert retriever.vector_retriever is not None
        assert retriever.keyword_retriever is not None
