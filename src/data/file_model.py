"""文件存储数据模型"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
import enum

from data.database import Base


class FileStatus(enum.Enum):
    """文件状态"""

    PENDING = "pending"  # 待处理
    PROCESSED = "processed"  # 已处理
    FAILED = "failed"  # 处理失败


class FileRecord(Base):
    """文件记录表 - 存储上传的文件信息"""

    __tablename__ = "file_records"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False, comment="文件名")
    original_name = Column(String(500), nullable=False, comment="原始文件名")
    file_type = Column(String(50), nullable=False, comment="文件类型/扩展名")
    file_size = Column(Integer, nullable=False, comment="文件大小(字节)")
    file_path = Column(String(1000), nullable=False, comment="存储路径")
    content = Column(Text, nullable=True, comment="解析后的文本内容")

    status = Column(SQLEnum(FileStatus), default=FileStatus.PENDING, comment="处理状态")
    error_message = Column(Text, nullable=True, comment="错误信息")

    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    def __repr__(self):
        return f"<FileRecord {self.id}: {self.filename}>"
