"""FastAPI 应用入口"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router, set_rag_pipeline
from core.config import settings
from core.logger import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理

    启动时初始化 RAG Pipeline，关闭时清理资源
    """
    # 启动时
    setup_logging()
    logger.info("RAG API 服务启动")

    # 创建数据库表
    try:
        from data.database import Base
        from data.database import db_manager

        # 直接导入模型文件，避免触发 data/__init__.py 中的 embedder 导入
        import importlib.util

        # 修复路径：src/api/ -> src/
        file_model_path = Path(__file__).parent.parent / "data" / "file_model.py"
        spec = importlib.util.spec_from_file_location("file_model", file_model_path)
        file_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(file_model)

        Base.metadata.create_all(bind=db_manager.engine)
        logger.info("数据库表创建完成")
    except Exception as e:
        logger.warning(f"数据库表创建失败: {e}")

    # 懒加载 RAG Pipeline
    try:
        from application import create_rag_pipeline

        # 懒加载 embedder 和 vector_store，避免启动时导入 torch
        from data.database import db_manager
        from data.vector_store import get_vector_store

        # 尝试加载 embedder，如果失败则跳过 RAG 功能
        try:
            from data.embedder import get_embedder

            embedder = get_embedder()
        except ImportError as e:
            logger.warning(f"嵌入模型加载失败: {e}")
            embedder = None

        vector_store = get_vector_store()

        if embedder is not None:
            pipeline = create_rag_pipeline(
                vector_store=vector_store,
                embedder=embedder,
                db_manager=db_manager,
            )
            set_rag_pipeline(pipeline)
            logger.info("RAG Pipeline 初始化完成")
        else:
            logger.warning("RAG Pipeline 未初始化（嵌入模型不可用）")
    except Exception as e:
        logger.warning(f"RAG Pipeline 初始化失败: {e}")

    yield

    # 关闭时
    logger.info("RAG API 服务关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用

    Returns:
        FastAPI: 配置好的 FastAPI 应用实例
    """
    app = FastAPI(
        title="RAG API",
        description="RAG 搜索补充系统 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS 中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # 注册路由
    app.include_router(router)

    # 静态文件服务
    ui_path = Path(__file__).parent.parent / "ui"
    if (ui_path / "templates").exists():

        @app.get("/")
        async def serve_index():
            return FileResponse(ui_path / "templates" / "index.html")

    return app


# 创建应用实例
app = create_app()
