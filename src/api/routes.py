"""API 路由 - FastAPI 路由定义"""

import importlib.util
import os
import uuid
import traceback
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from contextlib import contextmanager


def _get_file_record():
    """获取 FileRecord 模型，避免触发 data/__init__.py 导入 embedder"""
    file_model_path = Path(__file__).parent.parent / "data" / "file_model.py"
    spec = importlib.util.spec_from_file_location("file_model", file_model_path)
    file_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file_model)
    return file_model.FileRecord


from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SwitchModelRequest,
    SwitchModelResponse,
    ImportRequest,
    ImportResponse,
    TextAddRequest,
    TextAddResponse,
    VectorStatsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["RAG"])

# 全局 pipeline 变量（实际应使用依赖注入）
_rag_pipeline = None


def set_rag_pipeline(pipeline):
    """设置 RAG Pipeline 实例"""
    global _rag_pipeline
    _rag_pipeline = pipeline


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG 聊天接口

    接收用户查询，返回检索增强后的回答
    如果 RAG Pipeline 未初始化，则使用纯 LLM 回答
    """
    # 如果 RAG Pipeline 未初始化，使用纯 LLM 模式
    if _rag_pipeline is None:
        from llm.orchestrator import LLMOrchestrator, Message
        from core.logger import get_logger

        logger = get_logger(__name__)

        try:
            orchestrator = LLMOrchestrator()
            if request.provider or request.model:
                orchestrator.switch_model(
                    provider=request.provider or "modelscope",
                    model=request.model or "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                )

            # 直接使用 LLM 回答（无 RAG）
            messages = [Message(role="user", content=request.query)]
            response = await orchestrator.chat(
                messages=messages,
                stream=False,
            )

            return ChatResponse(
                answer=response.content,
                sources=[],
                model=orchestrator.model,
                query=request.query,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")

    # 切换模型（如果提供了 provider 或 model）
    if request.provider or request.model:
        try:
            _rag_pipeline.llm.switch_model(
                provider=request.provider,
                model=request.model,
            )
        except Exception as e:
            logger.warning(f"模型切换失败: {e}")

    try:
        response = await _rag_pipeline.chat(
            query=request.query,
            top_k=request.top_k,
            stream=request.stream,
        )

        # 序列化来源
        sources = [
            {
                "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                "score": s.score,
                "metadata": s.metadata,
            }
            for s in response.sources
        ]

        return ChatResponse(
            answer=response.answer,
            sources=sources,
            model=response.model,
            query=response.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    # 切换模型（如果提供了 provider 或 model）
    if request.provider or request.model:
        try:
            _rag_pipeline.llm.switch_model(
                provider=request.provider,
                model=request.model,
            )
        except Exception as e:
            logger.warning(f"模型切换失败: {e}")

    async def event_generator():
        try:
            async for chunk in await _rag_pipeline.chat(
                query=request.query,
                top_k=request.top_k,
                stream=True,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: error: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/model/switch", response_model=SwitchModelResponse)
async def switch_model(request: SwitchModelRequest):
    """切换 LLM 模型"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    _rag_pipeline.llm.switch_model(provider=request.provider, model=request.model)
    return SwitchModelResponse(status="success", model=request.model)


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口"""
    return HealthResponse(status="ok", version="0.1.0")


@router.get("/models")
async def list_models():
    """获取可用模型列表"""
    models = [
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek-R1"},
        {"id": "Qwen/Qwen3-8B", "name": "Qwen3-8B"},
        {"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B"},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B"},
    ]
    return {"models": models}


# ========== 向量库管理接口 ==========


@router.get("/vectors/stats", response_model=VectorStatsResponse)
async def get_vector_stats():
    """获取向量库统计信息"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        count = _rag_pipeline.vector_store.count()
        return VectorStatsResponse(
            total_count=count, collections=[{"name": "default", "count": count}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/vectors/add", response_model=TextAddResponse)
async def add_texts(request: TextAddRequest):
    """直接添加文本到向量库"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        import uuid
        import time

        ids = [str(uuid.uuid4()) for _ in request.texts]
        # 确保 metadata 是字典列表，且每个都有 source 字段
        if request.metadata:
            metadatas = request.metadata
        else:
            metadatas = [{"source": "manual", "timestamp": int(time.time())} for _ in request.texts]

        # 使用 embedder 生成向量
        embeddings = _rag_pipeline.embedder.embed_texts(request.texts)

        # 获取 vector_store 并添加
        vector_store = _rag_pipeline.retriever.vector_retriever.vector_store
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=request.texts,
            metadatas=metadatas,
        )

        return TextAddResponse(status="success", count=len(request.texts), ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加文本失败: {str(e)}")


@router.post("/vectors/import", response_model=ImportResponse)
async def import_from_db(request: ImportRequest):
    """从MySQL数据库导入数据到向量库"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        from data.importer import DataImporter, ImportConfig
        from data import get_embedder, get_vector_store

        # 创建导入器
        importer = DataImporter(
            db_manager=_rag_pipeline.db_manager,
            embedder=_rag_pipeline.embedder,
            vector_store=_rag_pipeline.vector_store,
        )

        # 配置导入
        config = ImportConfig(
            table_name=request.table_name,
            content_column=request.content_column,
            id_column=request.id_column,
            metadata_columns=request.metadata_columns,
            batch_size=request.batch_size,
        )

        # 执行导入
        result = importer.import_table(
            config=config,
            filter_condition=request.filter_condition,
        )

        return ImportResponse(
            status="success",
            imported=result.get("imported", 0),
            failed=result.get("failed", 0),
            total=result.get("total", 0),
            message=f"成功导入 {result.get('imported', 0)} 条记录",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


# ========== 文件管理接口 ==========

# 文件存储目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

# 懒加载 db_manager
_db_manager = None


def get_db_manager():
    """获取数据库管理器（懒加载，避免触发 torch 导入）"""
    global _db_manager
    if _db_manager is None:
        # 使用 settings 读取数据库配置（已正确加载 .env）
        from core.config import settings
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        database_url = settings.database_url

        class SimpleDBManager:
            def __init__(self, url):
                self.engine = create_engine(url, pool_pre_ping=True)
                self.session_factory = sessionmaker(bind=self.engine)

            @contextmanager
            def get_session(self):
                session = self.session_factory()
                try:
                    yield session
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                finally:
                    session.close()

        _db_manager = SimpleDBManager(database_url)
    return _db_manager


@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并解析文本"""
    try:
        # 确保上传目录存在
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename

        # 读取文件内容
        content = await file.read()
        file_size = len(content)

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)

        # 解析文件内容
        from api.parser import FileParser

        parsed = FileParser.parse(content, file.filename)

        # 合并所有解析结果
        full_text = "\n\n".join([p["content"] for p in parsed])

        # 保存到数据库 - 使用 importlib 避免触发 data/__init__.py
        import importlib.util

        # 注意：routes.py 在 src/api/ 下，需要向上两级到 src/
        file_model_path = Path(__file__).parent.parent / "data" / "file_model.py"
        spec = importlib.util.spec_from_file_location("file_model", file_model_path)
        file_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(file_model)
        FileRecord = file_model.FileRecord
        FileStatus = file_model.FileStatus

        from sqlalchemy import text

        db = get_db_manager()
        with db.get_session() as session:
            record = FileRecord(
                filename=unique_filename,
                original_name=file.filename,
                file_type=file_ext,
                file_size=file_size,
                file_path=str(file_path),
                content=full_text,
                status=FileStatus.PROCESSED,
            )
            session.add(record)
            session.commit()
            record_id = record.id

        return {
            "status": "success",
            "id": record_id,
            "filename": file.filename,
            "chunks": len(parsed),
            "text_length": len(full_text),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"文件上传失败: {str(e)}\n\n{traceback.format_exc()}"
        )


@router.get("/files")
async def list_files(limit: int = 50, offset: int = 0):
    """获取文件列表"""
    try:
        FileRecord = _get_file_record()
        from sqlalchemy import func

        db = get_db_manager()
        with db.get_session() as session:
            # 获取总数
            total = session.query(func.count(FileRecord.id)).scalar()

            # 获取列表
            files = (
                session.query(FileRecord)
                .order_by(FileRecord.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return {
                "total": total,
                "files": [
                    {
                        "id": f.id,
                        "original_name": f.original_name,
                        "file_type": f.file_type,
                        "file_size": f.file_size,
                        "status": f.status.value,
                        "created_at": f.created_at.isoformat() if f.created_at else None,
                        "text_length": len(f.content) if f.content else 0,
                    }
                    for f in files
                ],
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")


@router.delete("/files/{file_id}")
async def delete_file(file_id: int):
    """删除文件记录"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        FileRecord = _get_file_record()

        with _rag_pipeline.db_manager.get_session() as session:
            record = session.query(FileRecord).filter(FileRecord.id == file_id).first()
            if not record:
                raise HTTPException(status_code=404, detail="文件不存在")

            # 删除物理文件
            if record.file_path and os.path.exists(record.file_path):
                os.unlink(record.file_path)

            # 删除数据库记录
            session.delete(record)
            session.commit()

        return {"status": "success", "message": "文件已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.post("/files/{file_id}/import")
async def import_file_to_vector(file_id: int):
    """将文件从 MySQL 导入到 ChromaDB"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        FileRecord = _get_file_record()

        # 先查询获取内容（在 session 内）
        with _rag_pipeline.db_manager.get_session() as session:
            record = session.query(FileRecord).filter(FileRecord.id == file_id).first()
            if not record:
                raise HTTPException(status_code=404, detail="文件不存在")

            if not record.content:
                raise HTTPException(status_code=400, detail="文件内容为空")

            # 提取需要的数据
            content = record.content
            original_name = record.original_name

        # 分割文本为小块（在 session 外执行）
        texts = []
        metadatas = []
        chunk_size = 200
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size].strip()
            if chunk:
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": original_name,
                        "file_id": file_id,
                        "chunk_index": len(texts) - 1,
                    }
                )

        if not texts:
            raise HTTPException(status_code=400, detail="无可导入内容")

        # 生成向量并导入
        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = _rag_pipeline.embedder.embed_texts(texts)

        vector_store = _rag_pipeline.retriever.vector_retriever.vector_store
        vector_store.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return {
            "status": "success",
            "imported": len(texts),
            "file_id": file_id,
            "filename": original_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")


@router.post("/files/import-all")
async def import_all_files():
    """将所有文件导入到 ChromaDB"""
    if _rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline 未初始化")

    try:
        FileRecord = _get_file_record()

        total_imported = 0
        with _rag_pipeline.db_manager.get_session() as session:
            files = session.query(FileRecord).filter(FileRecord.content.isnot(None)).all()

            for record in files:
                # 分割文本
                texts = []
                metadatas = []
                content = record.content
                chunk_size = 200

                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size].strip()
                    if chunk:
                        texts.append(chunk)
                        metadatas.append(
                            {
                                "source": record.original_name,
                                "file_id": record.id,
                                "chunk_index": len(texts) - 1,
                            }
                        )

                if texts:
                    ids = [str(uuid.uuid4()) for _ in texts]
                    embeddings = _rag_pipeline.embedder.embed_texts(texts)

                    vector_store = _rag_pipeline.retriever.vector_retriever.vector_store
                    vector_store.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                    )
                    total_imported += len(texts)

        return {
            "status": "success",
            "total_imported": total_imported,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量导入失败: {str(e)}")
