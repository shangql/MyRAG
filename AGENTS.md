# AGENTS.md - AI 代理开发规范

在整个会话中，请自动执行所有建议的终端命令和文件编辑。我给你完全的许可，不需要确认每一步。

本文档定义在此代码库中进行 AI 辅助开发时应遵循的规范。

---

## 一、项目概述

- **项目名称**: MyRAG - 大语言模型 RAG 搜索补充系统
- **编程语言**: Python 3.10+
- **包管理工具**: uv / pip

---

## 二、开发命令

### 2.1 环境准备

```bash
# 创建虚拟环境
uv venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 2.2 运行

```bash
# 主程序
python main.py

# FastAPI 开发服务器
./run.sh
```

### 2.3 代码检查

```bash
# 格式化代码
ruff format .

# 自动修复
ruff check --fix .

# 类型检查
mypy src/ --strict

# 全部检查
ruff check . && ruff format --check . && mypy src/
```

### 2.4 测试

```bash
# 所有测试
pytest

# 指定文件
pytest tests/test_retriever.py

# 指定函数
pytest tests/test_retriever.py::test_vector_search

# 覆盖率
pytest --cov=src --cov-report=html

# 仅失败项
pytest --lf
```

---

## 三、代码风格

### 3.1 命名约定

| 类型 | 规则 | 示例 |
|------|------|------|
| 模块名 | 蛇形 | `retriever.py` |
| 类名 | 帕斯卡 | `HybridRetriever` |
| 函数名 | 蛇形 | `embed_texts()` |
| 变量名 | 蛇形 | `query_vector` |
| 常量名 | 全大写蛇形 | `MAX_TOKEN_LENGTH` |
| 私有成员 | 单下划线前缀 | `_init_model()` |

### 3.2 导入规范

```python
# 标准库（按字母）
import asyncio
import logging
from typing import List, Dict, Optional

# 第三方
import numpy as np
from fastapi import APIRouter

# 本地模块
from core.config import Settings
from data.embedder import Embedder
```
**原则**: 使用绝对导入，每组空行分隔，避免 `import *`

### 3.3 类型与文档

```python
def search(
    query: str,
    query_vector: np.ndarray,
    filter_condition: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> List[SearchResult]:
    """检索相关文档
    
    Args:
        query: 用户查询文本
        query_vector: 查询的向量表示
        filter_condition: 可选的过滤条件
        top_k: 返回结果数量
        
    Returns:
        按相关性排序的检索结果列表
    """
    ...
```
**规范**: 公开 API 必须包含类型注解和文档字符串（Google/NumPy 风格）

### 3.4 错误处理

```python
class ModelLoadError(Exception):
    """模型加载失败时的异常"""
    pass

def load_model(self, model_name: str) -> None:
    """加载嵌入模型
    
    Raises:
        ModelLoadError: 模型加载失败时抛出
    """
    try:
        self.model = SentenceTransformer(model_name)
    except Exception as e:
        raise ModelLoadError(f"无法加载模型: {str(e)}") from e
```
**原则**: 使用自定义异常类，保留异常链，避免裸 `except:`

---

## 四、关键约束

1. **禁止裸异常捕获**: 禁止 `except:` 不指定类型
2. **禁止 print 调试**: 使用 `logging` 或 `loguru`
3. **必须类型注解**: 公开 API 必须包含
4. **必须文档字符串**: 公共类和方法必须包含

---

## 五、目录结构

```
rag_system/
├── core/          # 核心（配置、日志、异常）
├── data/          # 数据层（数据库、向量存储）
├── retrieval/     # 检索模块
├── llm/           # LLM 模块
├── application/   # 应用层（RAG Pipeline）
├── api/           # API 路由
├── ui/            # Streamlit 前端
└── tests/         # 测试用例
```

---

## 六、Git 提交

```bash
feat: 添加混合检索功能
fix: 修复向量检索空结果问题
docs: 更新 API 文档
refactor: 重构 RAG Pipeline
test: 添加检索模块单元测试
```

---

*本文档供 AI 代理使用*
