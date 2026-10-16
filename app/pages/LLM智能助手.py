"""
LLM 智能助手
配置指引 + 聊天界面（DeepSeek API）
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ========== 配置检查 ==========
def check_api_config():
    """返回 (api_key, api_base, model, is_configured)"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 尝试用 python-dotenv 从 .env 读
    if not api_key or api_key == "sk-your-api-key-here":
        try:
            from dotenv import load_dotenv
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", ".env")
            load_dotenv(env_path)
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            api_base = os.getenv("DEEPSEEK_API_BASE", api_base)
            model = os.getenv("DEEPSEEK_MODEL", model)
        except Exception:
            pass

    is_configured = bool(api_key) and api_key != "sk-your-api-key-here" and len(api_key) > 10
    return api_key, api_base, model, is_configured

# ========== 聊天调用 ==========
def chat_with_llm(prompt: str, history: list, api_key: str, api_base: str, model: str) -> str:
    try:
        import requests

        messages = []
        # 系统角色
        messages.append({
            "role": "system",
            "content": "你是一个资深信贷风控专家助手，擅长回答关于信贷风控、机器学习模型、特征工程、XGBoost、SHAP解释等方面的问题。回答要专业、简洁、实用。",
        })
        # 历史对话
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        # 当前提问
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            timeout=60,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ 调用失败：{e}\n\n请检查 API Key 是否正确、网络是否通畅。"

# ========== 页面渲染 ==========
def render():
    st.title("🤖 LLM 智能助手")
    st.caption("由 DeepSeek 驱动的风控专家助手，解答风控建模、特征工程、模型解释等问题")

    # 初始化 session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    api_key, api_base, model, is_configured = check_api_config()

    # ========== 配置状态 ==========
    if not is_configured:
        st.warning("⚠️ **LLM API 尚未配置！** 请按以下步骤配置：")

        st.markdown(
            """
            <div style="background:#f8f9fa;border-radius:10px;padding:1rem 1.5rem;border-left:4px solid #4A90E2;">
            <h4 style="margin-top:0;">📋 配置步骤：</h4>
            <ol>
                <li>访问 <a href="https://platform.deepseek.com" target="_blank">https://platform.deepseek.com</a> 注册账号</li>
                <li>登录后在「API Keys」页面生成一个新的 Key</li>
                <li>打开项目根目录下的 <code>.env</code> 文件，修改这一行：<br>
                    <code>DEEPSEEK_API_KEY=sk-your-real-key-here</code></li>
                <li>保存 <code>.env</code> 文件</li>
                <li>**重启 Streamlit 应用**（Ctrl+C → 重新 run）</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📝 .env 文件完整示例"):
            st.code(
                """

# ===== DeepSeek API 配置 =====
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ===== 应用配置 =====
APP_ENV=development
LOG_LEVEL=INFO
DATA_DIR=./data
MODEL_DIR=./data/models

# ===== Streamlit 配置 =====
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
                """,
                language="bash",
            )

        # 继续显示一个 Demo 聊天区域
        st.divider()
        st.info("💡 虽然没配置 API，你仍然可以查看界面效果（但调用会失败）")

    else:
        st.success("✅ **LLM API 已就绪！** 模型：" + model)

        # 显示当前配置（打码）
        masked = api_key[:8] + "..." + api_key[-4:]
        st.caption(f"API Key: {masked} | Base: {api_base}")

    # ========== 聊天界面 ==========
    st.divider()
    st.markdown("### 💬 与风控专家对话")

    # 快捷问题按钮
    quick_qs = [
        "XGBoost 和 LightGBM 在信贷风控中怎么选？",
        "类别不平衡时除了 scale_pos_weight 还有什么方法？",
        "SHAP 值怎么解读？举个例子",
        "为什么我的模型 AUC 高但 Recall 低？",
        "特征工程时 employmentLength 怎么编码比较好？",
    ]

    col_per_row = 2
    for i in range(0, len(quick_qs), col_per_row):
        row = st.columns(col_per_row)
        for j, q in enumerate(quick_qs[i:i+col_per_row]):
            if row[j].button(f"💡 {q}", use_container_width=True, key=f"q_{i}_{j}"):
                st.session_state["quick_input"] = q

    st.divider()

    # 历史消息展示
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    # 输入框
    placeholder = "问我关于风控建模、特征工程、XGBoost、SHAP 解释的问题..."
    default_text = st.session_state.get("quick_input", "")
    user_input = st.chat_input(placeholder)

    # 处理快捷按钮
    if default_text and not user_input:
        user_input = default_text
        del st.session_state["quick_input"]

    if user_input:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(user_input)

        # 调用 LLM
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                if is_configured:
                    reply = chat_with_llm(
                        user_input,
                        st.session_state.chat_history,
                        api_key, api_base, model,
                    )
                else:
                    reply = (
                        "⚠️ **API 未配置，这是本地模拟回复**\n\n"
                        f"你的问题：*{user_input}*\n\n"
                        "要获得真实 AI 回答，请先按上方步骤配置 `.env` 里的 `DEEPSEEK_API_KEY`。"
                    )

            st.write(reply)

        # 追加历史（最多保留 20 轮对话）
        st.session_state.chat_history.append((user_input, reply))
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    # 清空按钮
    if len(st.session_state.chat_history) > 0:
        if st.button("🗑️ 清空对话历史"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    render()
