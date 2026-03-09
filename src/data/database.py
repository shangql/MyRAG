"""数据库连接器模块 - 提供统一的数据库连接和会话管理"""

from contextlib import contextmanager
from typing import Any, Generator, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from core.config import settings
from core.exceptions import DatabaseError
from core.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

# 创建 SQLAlchemy 声明式基类
Base = declarative_base()

# 模型导入延迟到使用时（避免启动时导入 embedder）


class DatabaseManager:
    """数据库管理器

    负责数据库连接池的创建、会话管理和表结构初始化。
    支持 MySQL、PostgreSQL 等主流关系型数据库。

    Attributes:
        engine: SQLAlchemy 引擎实例
        session_factory: 会话工厂
    """

    def __init__(self, database_url: Optional[str] = None):
        """初始化数据库管理器

        Args:
            database_url: 数据库连接 URL，默认使用配置中的 URL
        """
        self.database_url = database_url or settings.database_url
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._initialize()

    def _initialize(self) -> None:
        """初始化数据库引擎和会话工厂"""
        try:
            # 解析数据库 URL 获取数据库类型
            parsed = urlparse(self.database_url)
            db_type = parsed.scheme.split("+")[0]

            logger.info(
                f"初始化数据库连接 | 类型: {db_type} | URL: {parsed.hostname}:{parsed.port}"
            )

            # 创建引擎，配置连接池
            self.engine = create_engine(
                self.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_pre_ping=True,  # 连接前先检查
                pool_recycle=3600,  # 连接回收时间（1小时）
                echo=False,  # 是否打印 SQL 语句
            )

            # 创建会话工厂
            self.session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

            logger.info("数据库连接池初始化完成")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise DatabaseError(
                f"数据库初始化失败: {str(e)}",
                operation="init",
                original_error=e,
            )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话的上下文管理器

        使用方法:
            with db_manager.get_session() as session:
                session.query(Model).all()

        Yields:
            Session: SQLAlchemy 会话对象

        Raises:
            DatabaseError: 数据库操作失败时抛出
        """
        if not self.session_factory:
            raise DatabaseError("数据库会话工厂未初始化", operation="session")

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise DatabaseError(
                f"数据库操作失败: {str(e)}",
                operation="query",
                original_error=e,
            )
        finally:
            session.close()

    def execute_raw(self, sql: str, params: Optional[dict[str, Any]] = None) -> Any:
        """执行原始 SQL 语句

        Args:
            sql: SQL 语句
            params: SQL 参数

        Returns:
            查询结果

        Raises:
            DatabaseError: SQL 执行失败时抛出
        """
        with self.get_session() as session:
            try:
                result = session.execute(text(sql), params or {})
                # 如果是 SELECT 语句，返回结果
                if result.returns_rows:
                    return result.fetchall()
                return result.rowcount
            except Exception as e:
                raise DatabaseError(
                    f"原始 SQL 执行失败: {str(e)}",
                    operation="raw_sql",
                    original_error=e,
                )

    def test_connection(self) -> bool:
        """测试数据库连接

        Returns:
            bool: 连接是否成功
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False

    def create_tables(self) -> None:
        """创建所有定义的表

        注意：仅用于开发环境，生产环境应使用迁移工具
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("数据库表创建完成")
        except Exception as e:
            raise DatabaseError(
                f"创建数据库表失败: {str(e)}",
                operation="create_tables",
                original_error=e,
            )

    def close(self) -> None:
        """关闭数据库连接池"""
        if self.engine:
            self.engine.dispose()
            logger.info("数据库连接池已关闭")


# 创建全局数据库管理器实例
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入函数

    用于在 API 路由中获取数据库会话。

    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...

    Yields:
        Session: SQLAlchemy 会话对象
    """
    with db_manager.get_session() as session:
        yield session
