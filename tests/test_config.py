"""配置模块测试"""
import os
from pathlib import Path

import pytest

from core.config import Settings, get_settings


class TestSettings:
    """Settings 配置类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        settings = Settings()
        
        assert settings.database_url == "mysql+pymysql://user:password@localhost:3306/rag_db"
        assert settings.vector_store_type == "chroma"
        assert settings.top_k == 5
        assert settings.llm_provider == "openai"
    
    def test_environment_variable_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
        monkeypatch.setenv("TOP_K", "10")
        
        settings = Settings()
        
        assert settings.database_url == "postgresql://test:test@localhost/testdb"
        assert settings.top_k == 10
    
    def test_get_settings_cached(self):
        """测试配置缓存"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # 验证返回同一实例
        assert settings1 is settings2
    
    def test_pydantic_validation(self):
        """测试 Pydantic 验证"""
        with pytest.raises(Exception):
            # 无效的向量存储类型
            Settings(vector_store_type="invalid")


class TestSettingsFields:
    """Settings 字段测试"""
    
    def test_llm_provider_options(self):
        """测试 LLM 提供商选项"""
        settings = Settings(llm_provider="openai")
        assert settings.llm_provider == "openai"
        
        settings = Settings(llm_provider="anthropic")
        assert settings.llm_provider == "anthropic"
        
        settings = Settings(llm_provider="ollama")
        assert settings.llm_provider == "ollama"
    
    def test_vector_store_type_options(self):
        """测试向量存储类型选项"""
        settings = Settings(vector_store_type="chroma")
        assert settings.vector_store_type == "chroma"
        
        settings = Settings(vector_store_type="faiss")
        assert settings.vector_store_type == "faiss"
    
    def test_embedding_device_options(self):
        """测试嵌入设备选项"""
        settings = Settings(embedding_device="cpu")
        assert settings.embedding_device == "cpu"
        
        settings = Settings(embedding_device="cuda")
        assert settings.embedding_device == "cuda"
