"""Redis 缓存模块 - 缓存查询结果和向量"""
import json
import hashlib
from typing import Any, Optional
from dataclasses import asdict

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """缓存管理器

    使用 Redis 缓存查询结果和向量数据，提升响应速度。
    支持向量缓存、查询结果缓存。

    Attributes:
        redis_url: Redis 连接 URL
        ttl: 默认过期时间（秒）
    """

    # 缓存键前缀
    PREFIX_QUERY = "rag:query:"
    PREFIX_VECTOR = "rag:vector:"
    PREFIX_RESULT = "rag:result:"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        """初始化缓存管理器

        Args:
            redis_url: Redis 连接 URL，默认从配置读取
            default_ttl: 默认过期时间（秒）
        """
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl or settings.redis_ttl
        self._client = None

        if not self.redis_url:
            logger.warning("Redis URL 未配置，缓存功能将不可用")

    @property
    def client(self):
        """获取 Redis 客户端（延迟初始化）

        Returns:
            Redis 客户端实例
        """
        if self._client is None:
            if not self.redis_url:
                return None

            try:
                import redis

                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )
                # 测试连接
                self._client.ping()
                logger.info("Redis 缓存连接成功")

            except ImportError:
                logger.warning("redis 库未安装，缓存功能不可用")
                return None
            except Exception as e:
                logger.warning(f"Redis 连接失败: {e}")
                return None

        return self._client

    def _generate_key(self, prefix: str, value: str) -> str:
        """生成缓存键

        Args:
            prefix: 键前缀
            value: 键值

        Returns:
            完整的缓存键
        """
        # 对长值进行 hash
        if len(value) > 100:
            hash_value = hashlib.md5(value.encode()).hexdigest()
            return f"{prefix}{hash_value}"
        return f"{prefix}{value}"

    # ========== 查询结果缓存 ==========

    def get_query_result(self, query: str) -> Optional[dict]:
        """获取查询结果缓存

        Args:
            query: 查询文本

        Returns:
            缓存的查询结果，不存在返回 None
        """
        if not self.client:
            return None

        try:
            key = self._generate_key(self.PREFIX_QUERY, query)
            data = self.client.get(key)

            if data:
                logger.debug(f"查询缓存命中: {query[:30]}...")
                return json.loads(data)

            return None

        except Exception as e:
            logger.warning(f"查询缓存获取失败: {e}")
            return None

    def set_query_result(
        self,
        query: str,
        result: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置查询结果缓存

        Args:
            query: 查询文本
            result: 查询结果
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        if not self.client:
            return False

        try:
            key = self._generate_key(self.PREFIX_QUERY, query)
            ttl = ttl or self.default_ttl

            self.client.setex(
                key,
                ttl,
                json.dumps(result, ensure_ascii=False),
            )

            logger.debug(f"查询结果已缓存: {query[:30]}...")
            return True

        except Exception as e:
            logger.warning(f"查询缓存设置失败: {e}")
            return False

    # ========== 向量缓存 ==========

    def get_vector(self, text: str) -> Optional[list]:
        """获取向量缓存

        Args:
            text: 文本

        Returns:
            缓存的向量，不存在返回 None
        """
        if not self.client:
            return None

        try:
            key = self._generate_key(self.PREFIX_VECTOR, text)
            data = self.client.get(key)

            if data:
                logger.debug(f"向量缓存命中: {text[:30]}...")
                return json.loads(data)

            return None

        except Exception as e:
            logger.warning(f"向量缓存获取失败: {e}")
            return None

    def set_vector(
        self,
        text: str,
        vector: list,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置向量缓存

        Args:
            text: 文本
            vector: 向量列表
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        if not self.client:
            return False

        try:
            key = self._generate_key(self.PREFIX_VECTOR, text)
            ttl = ttl or self.default_ttl * 24  # 向量缓存更久

            self.client.setex(
                key,
                ttl,
                json.dumps(vector),
            )

            logger.debug(f"向量已缓存: {text[:30]}...")
            return True

        except Exception as e:
            logger.warning(f"向量缓存设置失败: {e}")
            return False

    # ========== 通用缓存 ==========

    def get(self, key: str) -> Optional[Any]:
        """获取通用缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在返回 None
        """
        if not self.client:
            return None

        try:
            data = self.client.get(key)
            return json.loads(data) if data else None

        except Exception as e:
            logger.warning(f"缓存获取失败: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """设置通用缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        if not self.client:
            return False

        try:
            ttl = ttl or self.default_ttl
            self.client.setex(
                key,
                ttl,
                json.dumps(value, ensure_ascii=False),
            )
            return True

        except Exception as e:
            logger.warning(f"缓存设置失败: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        if not self.client:
            return False

        try:
            self.client.delete(key)
            return True

        except Exception as e:
            logger.warning(f"缓存删除失败: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """删除匹配模式的所有缓存

        Args:
            pattern: 匹配模式

        Returns:
            删除的键数量
        """
        if not self.client:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0

        except Exception as e:
            logger.warning(f"批量删除缓存失败: {e}")
            return 0

    def clear_all(self) -> int:
        """清空所有 RAG 相关缓存

        Returns:
            删除的键数量
        """
        count = 0
        for prefix in [self.PREFIX_QUERY, self.PREFIX_VECTOR, self.PREFIX_RESULT]:
            count += self.clear_pattern(f"{prefix}*")
        logger.info(f"已清空 {count} 个缓存键")
        return count


# 全局缓存管理器实例
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例（单例）

    Returns:
        CacheManager: 缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
