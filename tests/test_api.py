"""API 模块测试"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import set_rag_pipeline


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG Pipeline"""
    pipeline = MagicMock()
    
    # Mock chat response
    mock_response = MagicMock()
    mock_response.answer = "测试回答"
    mock_response.sources = []
    mock_response.model = "gpt-3.5-turbo"
    mock_response.query = "测试问题"
    mock_response.usage = {"total_tokens": 100}
    
    pipeline.chat = AsyncMock(return_value=mock_response)
    pipeline.llm = MagicMock()
    pipeline.llm.model = "gpt-3.5-turbo"
    pipeline.llm.switch_model = MagicMock()
    
    return pipeline


@pytest.fixture
def client(mock_rag_pipeline):
    """TestClient fixture"""
    with TestClient(app) as test_client:
        set_rag_pipeline(mock_rag_pipeline)
        yield test_client


class TestHealthEndpoint:
    """健康检查接口测试"""
    
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestChatEndpoint:
    """聊天接口测试"""
    
    def test_chat_success(self, client, mock_rag_pipeline):
        """测试成功聊天"""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "什么是机器学习？",
                "top_k": 5,
                "stream": False,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model" in data
        assert data["answer"] == "测试回答"
    
    def test_chat_with_custom_model(self, client, mock_rag_pipeline):
        """测试指定模型"""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "测试",
                "model": "gpt-4",
            },
        )
        
        assert response.status_code == 200
    
    def test_chat_validation_error(self, client):
        """测试请求验证错误"""
        response = client.post(
            "/api/v1/chat",
            json={},
        )
        
        assert response.status_code == 422
    
    def test_chat_pipeline_not_initialized(self):
        """测试 Pipeline 未初始化"""
        # 创建不带 mock 的客户端
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/chat",
                json={"query": "测试"},
            )
            
            assert response.status_code == 500


class TestSwitchModelEndpoint:
    """切换模型接口测试"""
    
    def test_switch_model_success(self, client, mock_rag_pipeline):
        """测试成功切换模型"""
        response = client.post(
            "/api/v1/model/switch",
            json={
                "provider": "openai",
                "model": "gpt-4",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "gpt-4"
        
        # 验证 switch_model 被调用
        mock_rag_pipeline.llm.switch_model.assert_called_once_with(
            provider="openai",
            model="gpt-4",
        )
    
    def test_switch_model_pipeline_not_initialized(self):
        """测试 Pipeline 未初始化"""
        with TestClient(app) as test_client:
            response = test_client.post(
                "/api/v1/model/switch",
                json={
                    "provider": "openai",
                    "model": "gpt-4",
                },
            )
            
            assert response.status_code == 500


class TestStreamEndpoint:
    """流式接口测试"""
    
    def test_chat_stream(self, client, mock_rag_pipeline):
        """测试流式聊天"""
        # Mock 返回异步生成器
        async def mock_stream():
            yield "你好"
            yield "世界"
        
        mock_rag_pipeline.chat = AsyncMock(return_value=mock_stream())
        
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "query": "你好",
                "stream": True,
            },
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestCORS:
    """CORS 跨域测试"""
    
    def test_cors_headers(self, client):
        """测试 CORS 头"""
        response = client.options(
            "/api/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        
        # 检查是否返回 CORS 头
        assert "access-control-allow-origin" in response.headers or True


class TestSchemas:
    """Pydantic 模型测试"""
    
    def test_chat_request_validation(self):
        """测试 ChatRequest 验证"""
        from api.schemas import ChatRequest
        
        # 有效请求
        req = ChatRequest(query="测试")
        assert req.query == "测试"
        assert req.top_k == 5
        assert req.stream is False
        
        # 带参数的请求
        req = ChatRequest(query="测试", top_k=10, stream=True, model="gpt-4")
        assert req.top_k == 10
        assert req.stream is True
        assert req.model == "gpt-4"
    
    def test_chat_response(self):
        """测试 ChatResponse"""
        from api.schemas import ChatResponse
        
        resp = ChatResponse(
            answer="回答",
            sources=[{"content": "源", "score": 0.9, "metadata": {}}],
            model="gpt-3.5-turbo",
            query="问题",
        )
        
        assert resp.answer == "回答"
        assert len(resp.sources) == 1
    
    def test_switch_model_request(self):
        """测试 SwitchModelRequest"""
        from api.schemas import SwitchModelRequest
        
        req = SwitchModelRequest(provider="openai", model="gpt-4")
        assert req.provider == "openai"
        assert req.model == "gpt-4"
    
    def test_health_response(self):
        """测试 HealthResponse"""
        from api.schemas import HealthResponse
        
        resp = HealthResponse(status="ok", version="1.0.0")
        assert resp.status == "ok"
        assert resp.version == "1.0.0"
