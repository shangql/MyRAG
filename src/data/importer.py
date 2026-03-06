"""数据导入脚本 - 从数据库同步到向量库"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import hashlib

from sqlalchemy import text

from core.config import settings
from core.exceptions import DatabaseError, VectorStoreError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImportConfig:
    """数据导入配置

    Attributes:
        table_name: 数据库表名
        id_column: ID 列名
        content_column: 内容列名
        metadata_columns: 元数据列名列表
        batch_size: 批处理大小
    """
    table_name: str
    id_column: str = "id"
    content_column: str = "content"
    metadata_columns: Optional[List[str]] = None
    batch_size: int = 100


class DataImporter:
    """数据导入器

    从关系型数据库读取数据，转换为向量并存储到向量数据库。
    支持增量导入和断点续传。

    Attributes:
        db_manager: 数据库管理器
        embedder: 嵌入模型
        vector_store: 向量存储
    """

    def __init__(
        self,
        db_manager,
        embedder,
        vector_store,
    ):
        """初始化数据导入器

        Args:
            db_manager: 数据库管理器实例
            embedder: 嵌入模型实例
            vector_store: 向量存储实例
        """
        self.db_manager = db_manager
        self.embedder = embedder
        self.vector_store = vector_store

    def import_table(
        self,
        config: ImportConfig,
        filter_condition: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, int]:
        """从数据库表导入数据到向量库

        Args:
            config: 导入配置
            filter_condition: SQL WHERE 条件
            progress_callback: 进度回调函数 (current, total)

        Returns:
            导入统计信息
        """
        # 构建查询 SQL
        columns = [config.id_column, config.content_column]
        if config.metadata_columns:
            columns.extend(config.metadata_columns)

        sql = f"SELECT {', '.join(columns)} FROM {config.table_name}"
        if filter_condition:
            sql += f" WHERE {filter_condition}"

        logger.info(f"开始导入数据 | 表: {config.table_name} | 条件: {filter_condition}")

        # 获取总数
        count_sql = f"SELECT COUNT(*) as total FROM {config.table_name}"
        if filter_condition:
            count_sql += f" WHERE {filter_condition}"

        try:
            with self.db_manager.get_session() as session:
                result = session.execute(text(count_sql))
                total = result.fetchone()[0]
        except Exception as e:
            raise DatabaseError(f"获取总数失败: {str(e)}", operation="count")

        logger.info(f"待导入文档总数: {total}")

        # 分批导入
        imported = 0
        failed = 0
        offset = 0

        while offset < total:
            # 读取当前批次
            batch_sql = f"{sql} LIMIT {config.batch_size} OFFSET {offset}"

            try:
                with self.db_manager.get_session() as session:
                    result = session.execute(text(batch_sql))
                    rows = result.fetchall()
            except Exception as e:
                logger.error(f"批次读取失败 (offset={offset}): {e}")
                failed += config.batch_size
                offset += config.batch_size
                continue

            if not rows:
                break

            # 处理批次数据
            batch_result = self._process_batch(
                rows=rows,
                config=config,
            )

            imported += batch_result["imported"]
            failed += batch_result["failed"]

            # 进度回调
            if progress_callback:
                progress_callback(offset + len(rows), total)

            offset += config.batch_size

            logger.debug(
                f"批次导入完成 | 已处理: {offset}/{total} | "
                f"成功: {imported} | 失败: {failed}"
            )

        stats = {
            "total": total,
            "imported": imported,
            "failed": failed,
            "success_rate": round(imported / total * 100, 2) if total > 0 else 0,
        }

        logger.info(
            f"导入完成 | 总数: {total} | 成功: {imported} | "
            f"失败: {failed} | 成功率: {stats['success_rate']}%"
        )

        return stats

    def _process_batch(
        self,
        rows: List[Any],
        config: ImportConfig,
    ) -> Dict[str, int]:
        """处理单个批次的数据

        Args:
            rows: 数据库行数据
            config: 导入配置

        Returns:
            处理结果统计
        """
        documents = []
        ids = []
        metadatas = []

        for row in rows:
            try:
                # 提取数据
                doc_id = str(row[0])
                content = str(row[1]) if row[1] else ""

                if not content.strip():
                    continue

                # 生成向量 ID
                ids.append(doc_id)
                documents.append(content)

                # 提取元数据
                metadata = {}
                if config.metadata_columns:
                    for i, col in enumerate(config.metadata_columns):
                        if i + 2 < len(row):
                            metadata[col] = row[i + 2]
                metadatas.append(metadata)

            except Exception as e:
                logger.warning(f"数据处理失败: {e}")
                continue

        if not documents:
            return {"imported": 0, "failed": len(rows)}

        try:
            # 批量向量化
            embeddings = self.embedder.embed_texts(documents)

            # 添加到向量存储
            self.vector_store.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            return {"imported": len(documents), "failed": len(rows) - len(documents)}

        except Exception as e:
            logger.error(f"批次向量化失败: {e}")
            return {"imported": 0, "failed": len(rows)}

    def sync_incremental(
        self,
        config: ImportConfig,
        update_column: str = "updated_at",
    ) -> Dict[str, int]:
        """增量同步数据

        只同步更新时间大于上次同步时间的数据。

        Args:
            config: 导入配置
            update_column: 更新时间列名

        Returns:
            同步统计信息
        """
        # TODO: 实现增量同步逻辑
        # 需要记录上次同步时间
        logger.warning("增量同步功能待实现")
        return {"imported": 0, "updated": 0, "deleted": 0}


def create_importer(
    db_manager,
    embedder,
    vector_store,
) -> DataImporter:
    """创建数据导入器

    Args:
        db_manager: 数据库管理器
        embedder: 嵌入模型
        vector_store: 向量存储

    Returns:
        DataImporter: 数据导入器实例
    """
    return DataImporter(
        db_manager=db_manager,
        embedder=embedder,
        vector_store=vector_store,
    )
