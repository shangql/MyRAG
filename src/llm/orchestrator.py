"""LLM 模块 - 统一管理多模型调用"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from core.config import settings
from core.exceptions import LLMAPIError, LLMRateLimitError, LLMResponseError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应数据类
    
    Attributes:
        content: 生成的文本内容
        model: 使用的模型名称
        usage: token 使用量统计
        finish_reason: 结束原因 (stop/length)
    """
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


@dataclass
class Message:
    """聊天消息数据类
    
    Attributes:
        role: 角色 (system/user/assistant)
        content: 消息内容
    """
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {"role": self.role, "content": self.content}


class BaseLLM(ABC):
    """LLM 基类 - 定义统一接口"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """同步生成
        
        Args:
            prompt: 提示词
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 响应对象
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """聊天模式生成
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 响应对象
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成
        
        Args:
            prompt: 提示词
            **kwargs: 额外参数
            
        Yields:
            str: 生成的文本片段
        """
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式聊天
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Yields:
            str: 生成的文本片段
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现类
    
    支持 GPT-3.5、GPT-4 等模型。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        base_url: Optional[str] = None,
    ):
        """初始化 OpenAI LLM
        
        Args:
            api_key: OpenAI API 密钥，默认从环境变量读取
            model: 模型名称
            temperature: 采样温度 (0-2)
            max_tokens: 最大生成 token 数
            base_url: 自定义 API 地址（用于代理或兼容接口）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        if not self.api_key:
            raise LLMAPIError("OpenAI API 密钥未设置", provider="openai")
        
        self._client = None
    
    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client
    
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """同步生成"""
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """聊天模式生成"""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[msg.to_dict() for msg in messages],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False,
            )
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
            )
            
        except Exception as e:
            self._handle_error(e)
    
    async def stream_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成"""
        messages = [Message(role="user", content=prompt)]
        async for chunk in self.stream_chat(messages, **kwargs):
            yield chunk
    
    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式聊天"""
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[msg.to_dict() for msg in messages],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> None:
        """处理 API 错误"""
        error_msg = str(error)
        
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise LLMRateLimitError(
                "OpenAI API 速率限制",
                provider="openai",
            )
        elif "authentication" in error_msg.lower() or "401" in error_msg:
            raise LLMAPIError(
                "OpenAI API 认证失败",
                provider="openai",
                status_code=401,
            )
        elif "insufficient_quota" in error_msg.lower():
            raise LLMAPIError(
                "OpenAI API 配额不足",
                provider="openai",
            )
        else:
            raise LLMAPIError(
                f"OpenAI API 调用失败: {error_msg}",
                provider="openai",
            )


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM 实现类
    
    支持 Claude 3 系列模型。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """初始化 Anthropic LLM
        
        Args:
            api_key: Anthropic API 密钥，默认从环境变量读取
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise LLMAPIError("Anthropic API 密钥未设置", provider="anthropic")
        
        self._client = None
    
    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
            )
        return self._client
    
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """同步生成"""
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """聊天模式生成"""
        try:
            # 转换消息格式
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    converted_messages.append({
                        "role": "user",
                        "content": f"System: {msg.content}"
                    })
                else:
                    converted_messages.append({"role": msg.role, "content": msg.content})
            
            response = await self.client.messages.create(
                model=kwargs.get("model", self.model),
                messages=converted_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=False,
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                finish_reason="stop",
            )
            
        except Exception as e:
            self._handle_error(e)
    
    async def stream_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成"""
        messages = [Message(role="user", content=prompt)]
        async for chunk in self.stream_chat(messages, **kwargs):
            yield chunk
    
    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式聊天"""
        try:
            converted_messages = []
            for msg in messages:
                if msg.role == "system":
                    converted_messages.append({
                        "role": "user",
                        "content": f"System: {msg.content}"
                    })
                else:
                    converted_messages.append({"role": msg.role, "content": msg.content})
            
            async with self.client.messages.stream(
                model=kwargs.get("model", self.model),
                messages=converted_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            self._handle_error(e)
    
    def _handle_error(self, error: Exception) -> None:
        """处理 API 错误"""
        error_msg = str(error)
        
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            raise LLMRateLimitError(
                "Anthropic API 速率限制",
                provider="anthropic",
            )
        else:
            raise LLMAPIError(
                f"Anthropic API 调用失败: {error_msg}",
                provider="anthropic",
            )


class OllamaLLM(BaseLLM):
    """Ollama 本地模型实现类
    
    支持通过 Ollama 运行本地开源模型（Llama2、Qwen 等）。
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """初始化 Ollama LLM
        
        Args:
            base_url: Ollama 服务地址
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """同步生成"""
        import aiohttp
        
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "options": {"num_predict": max_tokens},
                        "stream": False,
                    },
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise LLMAPIError(
                            f"Ollama API 调用失败: {text}",
                            provider="ollama",
                            status_code=response.status,
                        )
                    
                    result = await response.json()
                    return LLMResponse(
                        content=result.get("response", ""),
                        model=model,
                        usage={},
                        finish_reason="stop",
                    )
        except aiohttp.ClientError as e:
            raise LLMAPIError(
                f"无法连接到 Ollama 服务: {str(e)}",
                provider="ollama",
            )
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> LLMResponse:
        """聊天模式生成"""
        import aiohttp
        
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # 转换消息格式
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": ollama_messages,
                        "temperature": temperature,
                        "options": {"num_predict": max_tokens},
                        "stream": False,
                    },
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise LLMAPIError(
                            f"Ollama API 调用失败: {text}",
                            provider="ollama",
                            status_code=response.status,
                        )
                    
                    result = await response.json()
                    return LLMResponse(
                        content=result["message"]["content"],
                        model=model,
                        usage={},
                        finish_reason=result.get("done_reason", "stop"),
                    )
        except aiohttp.ClientError as e:
            raise LLMAPIError(
                f"无法连接到 Ollama 服务: {str(e)}",
                provider="ollama",
            )
    
    async def stream_generate(
        self,
        prompt: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成"""
        import aiohttp
        
        model = kwargs.get("model", self.model)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.content:
                        if line:
                            data = line.decode().strip()
                            if data.startswith("data:"):
                                import json
                                try:
                                    result = json.loads(data[5:])
                                    if "response" in result:
                                        yield result["response"]
                                except json.JSONDecodeError:
                                    continue
        except aiohttp.ClientError as e:
            raise LLMAPIError(
                f"无法连接到 Ollama 服务: {str(e)}",
                provider="ollama",
            )
    
    async def stream_chat(
        self,
        messages: List[Message],
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式聊天"""
        import aiohttp
        
        model = kwargs.get("model", self.model)
        
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": ollama_messages,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.content:
                        if line:
                            data = line.decode().strip()
                            if data.startswith("data:"):
                                import json
                                try:
                                    result = json.loads(data[5:])
                                    if "message" in result and "content" in result["message"]:
                                        yield result["message"]["content"]
                                except json.JSONDecodeError:
                                    continue
        except aiohttp.ClientError as e:
            raise LLMAPIError(
                f"无法连接到 Ollama 服务: {str(e)}",
                provider="ollama",
            )


class LLMOrchestrator:
    """LLM 调度器 - 工厂模式管理多模型
    
    根据配置动态选择 LLM 提供商，支持模型热切换。
    
    Attributes:
        provider: 当前 LLM 提供商
        model: 当前模型名称
    """
    
    # 提供商映射
    _providers: Dict[str, type] = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "ollama": OllamaLLM,
    }
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        """初始化调度器
        
        Args:
            provider: LLM 提供商名称 (openai/anthropic/ollama)
            model: 模型名称
            **kwargs: 传递给具体 LLM 类的参数
        """
        self.provider = provider or settings.llm_provider
        self.model = model or self._get_default_model()
        self.llm = self._create_llm(self.provider, self.model, **kwargs)
        logger.info(f"LLM 调度器初始化完成 | 提供商: {self.provider} | 模型: {self.model}")
    
    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "openai":
            return settings.openai_model
        elif self.provider == "anthropic":
            return settings.anthropic_model
        elif self.provider == "ollama":
            return settings.ollama_model
        return "gpt-3.5-turbo"
    
    def _create_llm(
        self,
        provider: str,
        model: str,
        **kwargs,
    ) -> BaseLLM:
        """创建 LLM 实例"""
        llm_class = self._providers.get(provider)
        if not llm_class:
            raise LLMAPIError(
                f"不支持的 LLM 提供商: {provider}",
                provider=provider,
            )
        
        # 合并配置
        config = {
            "model": model,
            "temperature": kwargs.get("temperature", settings.llm_temperature),
            "max_tokens": kwargs.get("max_tokens", settings.llm_max_tokens),
        }
        
        if provider == "openai":
            config["api_key"] = kwargs.get("api_key") or settings.openai_api_key
        elif provider == "anthropic":
            config["api_key"] = kwargs.get("api_key") or settings.anthropic_api_key
        elif provider == "ollama":
            config["base_url"] = kwargs.get("base_url") or settings.ollama_base_url
        
        return llm_class(**config)
    
    def switch_model(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """切换 LLM 模型
        
        Args:
            provider: 新的 LLM 提供商
            model: 新的模型名称
            **kwargs: 额外参数
        """
        self.provider = provider or self.provider
        self.model = model or self._get_default_model()
        self.llm = self._create_llm(self.provider, self.model, **kwargs)
        logger.info(f"模型切换完成 | 提供商: {self.provider} | 模型: {self.model}")
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """生成响应"""
        return await self.llm.generate(prompt, **kwargs)
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """聊天模式生成"""
        return await self.llm.chat(messages, **kwargs)
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成"""
        return await self.llm.stream_generate(prompt, **kwargs)
    
    async def stream_chat(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """流式聊天"""
        return await self.llm.stream_chat(messages, **kwargs)


def get_llm_orchestrator() -> LLMOrchestrator:
    """获取 LLM 调度器单例
    
    Returns:
        LLMOrchestrator: LLM 调度器实例
    """
    return LLMOrchestrator()
