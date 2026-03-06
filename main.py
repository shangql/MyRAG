"""MyRAG - 大语言模型 RAG 搜索补充系统

主入口文件，提供多种运行方式：
- API 服务 (FastAPI)
- Web 界面 (Streamlit)
"""
import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse

from core.logger import logger, setup_logging


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MyRAG - RAG 搜索补充系统")
    parser.add_argument(
        "mode",
        choices=["api", "ui"],
        help="运行模式: api (FastAPI) 或 ui (Streamlit)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API 服务地址")
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    # 初始化日志
    setup_logging()
    
    if args.mode == "api":
        # 运行 FastAPI 服务
        import uvicorn
        logger.info(f"启动 API 服务: {args.host}:{args.port}")
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.mode == "ui":
        # 运行 Streamlit 界面
        import streamlit.web.cli as stcli
        logger.info("启动 Web 界面")
        sys.argv = ["streamlit", "run", "src/ui/app.py"]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
