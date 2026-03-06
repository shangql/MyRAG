"""日志配置模块 - 使用 Loguru 进行统一日志管理"""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from core.config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    log_rotation: Optional[str] = None,
    log_retention: Optional[str] = None,
) -> None:
    """配置 Loguru 日志系统
    
    根据配置设置日志级别、格式、输出位置等。
    默认同时输出到控制台和文件。
    
    Args:
        log_level: 日志级别 (TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL)
        log_file: 日志文件路径
        log_format: 日志格式字符串
        log_rotation: 日志轮转大小
        log_retention: 日志保留时间
    """
    # 使用配置值或默认值
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    fmt = log_format or settings.log_format
    rotation = log_rotation or settings.log_rotation
    retention = log_retention or settings.log_retention
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # 添加文件处理器
    if file_path:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            file_path,
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,  # 异步写入，避免阻塞
        )
    
    # 设置全局日志级别
    logger.level(level)
    
    # 记录启动日志
    logger.info(f"日志系统初始化完成 | 级别: {level}")
    if file_path:
        logger.info(f"日志文件: {file_path}")


def get_logger(name: str | None = None) -> "logger":
    """获取日志记录器
    
    如果提供了 name，则返回带有名称的子日志记录器。
    
    Args:
        name: 日志记录器名称，通常使用 __name__
        
    Returns:
        Loguru logger 实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 默认日志记录器
__all__ = ["logger", "setup_logging", "get_logger"]
