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
    
    # 懒加载 RAG Pipeline
    try:
        from application import create_rag_pipeline
        from data import get_embedder, get_vector_store
        
        embedder = get_embedder()
        vector_store = get_vector_store()
        pipeline = create_rag_pipeline(
            vector_store=vector_store,
            embedder=embedder,
        )
        set_rag_pipeline(pipeline)
        logger.info("RAG Pipeline 初始化完成")
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
