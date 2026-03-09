"""Streamlit 前端应用 - Web 聊天界面"""
import streamlit as st
from typing import List, Dict, Optional


@st.cache_data(ttl=3600)
def fetch_modelscope_models(api_key: str) -> List[Dict[str, str]]:
    """从 ModelScope API 获取可用模型列表

    Args:
        api_key: ModelScope API 密钥

    Returns:
        模型列表，每个元素包含 id 和显示名称
    """
    import requests

    try:
        response = requests.get(
            "https://api-inference.modelscope.cn/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            models = []
            for item in data.get("data", []):
                model_id = item.get("id", "")
                # 提取模型简称用于显示
                display_name = model_id.split("/")[-1] if "/" in model_id else model_id
                models.append({
                    "id": model_id,
                    "display": display_name,
                })
            return models
        else:
            st.error(f"获取模型列表失败: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"连接 ModelScope API 失败: {e}")
        return []


def get_default_models() -> List[Dict[str, str]]:
    """获取默认模型列表（API 失败时的备用）"""
    return [
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "display": "DeepSeek-R1-Distill-Qwen-7B"},
        {"id": "Qwen/Qwen3-8B", "display": "Qwen3-8B"},
        {"id": "Qwen/Qwen3-4B", "display": "Qwen3-4B"},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "display": "Qwen2.5-7B-Instruct"},
        {"id": "Qwen/Qwen2.5-4B-Instruct", "display": "Qwen2.5-4B-Instruct"},
    ]


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def render_sidebar():
    """渲染侧边栏

    Returns:
        tuple: (model_name, top_k)
    """
    st.sidebar.title("⚙️ 系统设置")

    # 从环境变量获取 API Key（不硬编码在代码中）
    import os
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")

    # 如果环境变量没有，尝试从 st.secrets 获取
    if not api_key:
        try:
            api_key = st.secrets.get("MODELSCOPE_API_KEY", "")
        except:
            pass

    # 如果都没有，提示用户输入
    if not api_key:
        api_key = st.sidebar.text_input(
            "请输入 ModelScope API Key",
            type="password",
            help="在 https://modelscope.cn 获取 API Key"
        )
        if not api_key:
            st.sidebar.warning("未配置 API Key，部分功能可能不可用")
            models = get_default_models()
        else:
            models = fetch_modelscope_models(api_key)
    else:
        # 获取模型列表
        models = fetch_modelscope_models(api_key)
        if not models:
            models = get_default_models()

    # 按模型类型分组
    model_groups: Dict[str, List[Dict]] = {}
    for model in models:
        model_id = model["id"]
        if "/" in model_id:
            group = model_id.split("/")[0]
        else:
            group = "other"
        if group not in model_groups:
            model_groups[group] = []
        model_groups[group].append(model)

    # 创建扁平化的选项列表（保留原始 id）
    all_model_ids = [m["id"] for m in models]

    # 格式化显示函数
    def format_model(model_id: str) -> str:
        """格式化模型显示名称"""
        if "/" in model_id:
            owner, name = model_id.split("/", 1)
            return f"{name} ({owner})"
        return model_id

    # 默认选中的模型
    default_index = 0
    if st.session_state.model in all_model_ids:
        default_index = all_model_ids.index(st.session_state.model)

    # 显示刷新按钮
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        model_name = st.selectbox(
            "选择模型",
            options=all_model_ids,
            index=default_index,
            format_func=format_model,
            key="model_selector",
        )
    with col2:
        if st.button("🔄", help="刷新模型列表"):
            st.cache_data.clear()
            st.rerun()

    # 保存选中的模型
    st.session_state.model = model_name

    # 检索参数
    top_k = st.sidebar.slider("检索数量", 1, 10, 5)

    # 清空对话
    if st.sidebar.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    return model_name, top_k


def render_chat_message(role: str, content: str, sources: list = None):
    """渲染单条聊天消息
    
    Args:
        role: 角色 (user/assistant)
        content: 消息内容
        sources: 引用来源列表
    """
    with st.chat_message(role):
        st.markdown(content)
        
        # 显示引用来源
        if sources and role == "assistant":
            with st.expander("📚 参考来源"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {s.get('content', '')[:200]}...")
                    st.caption(f"相似度: {s.get('score', 0):.2f}")


def call_api(query: str, model: str, top_k: int, provider: str = "modelscope"):
    """调用后端 API

    Args:
        query: 查询文本
        model: 模型名称
        top_k: 检索数量
        provider: LLM 提供商

    Returns:
        dict: API 响应
    """
    import requests

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/chat",
            json={"query": query, "model": model, "provider": provider, "top_k": top_k},
            timeout=30,
        )
        return response.json()
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return None


def main():
    """主函数"""
    st.set_page_config(
        page_title="RAG 智能问答系统",
        page_icon="💬",
        layout="wide",
    )
    
    # 初始化
    init_session_state()
    
    # 渲染侧边栏
    model_name, top_k = render_sidebar()
    
    # 标题
    st.title("💬 RAG 智能问答系统")
    st.markdown("基于检索增强生成的大语言模型问答系统")
    
    # 渲染历史消息
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"], msg.get("sources"))
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        render_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 调用 API 获取回复
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                response = call_api(prompt, model_name, top_k)
                
                if response:
                    answer = response.get("answer", "抱歉，生成回答失败")
                    sources = response.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # 显示引用
                    if sources:
                        with st.expander("📚 参考来源"):
                            for i, s in enumerate(sources, 1):
                                st.markdown(f"**{i}.** {s.get('content', '')[:200]}...")
                    
                    # 保存到历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })


if __name__ == "__main__":
    main()
