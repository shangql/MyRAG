"""Streamlit 前端应用 - Web 聊天界面"""
import streamlit as st


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = "openai"


def render_sidebar():
    """渲染侧边栏
    
    Returns:
        tuple: (model_provider, model_name, top_k)
    """
    st.sidebar.title("⚙️ 系统设置")
    
    # 模型选择 - 只支持 ModelScope
    model_provider = "modelscope"

    model_options = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-4B-Instruct",
    ]

    model_name = st.sidebar.selectbox("模型", model_options, index=0)
    
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
