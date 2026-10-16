"""
系统配置页面
数据路径 / 模型参数 / API Key / 通用设置
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

def read_env():
    """读取 .env 文件为字典"""
    result = {}
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    result[k.strip()] = v.strip()
    return result

def render():
    st.title("⚙️ 系统配置")
    st.caption("管理数据路径、模型参数、LLM API 等全局设置")

    env = read_env()

    # ========== Tab ==========
    t_data, t_model, t_llm, t_about = st.tabs(
        ["📂 数据路径", "🧠 模型参数", "🔑 LLM API", "ℹ️ 关于系统"]
    )

    # ---------- Tab 1: 数据路径 ----------
    with t_data:
        st.markdown("### 📂 数据与模型目录")

        col1, col2 = st.columns(2)
        data_dir = col1.text_input("数据目录 (DATA_DIR)",
                                  value=env.get("DATA_DIR", "./data"))
        model_dir = col2.text_input("模型目录 (MODEL_DIR)",
                                    value=env.get("MODEL_DIR", "./data/models"))

        st.divider()

        # 验证路径
        base = os.path.join(PROJECT_ROOT)
        actual_data = os.path.join(base, data_dir.lstrip("./"))
        actual_model = os.path.join(base, model_dir.lstrip("./"))

        c1, c2 = st.columns(2)
        with c1:
            if os.path.isdir(actual_data):
                st.success(f"✅ 数据目录存在（{actual_data}）")
                files = [f for f in os.listdir(actual_data) if f.endswith(".csv")]
                st.caption(f"CSV 文件：{', '.join(files)}")
            else:
                st.error(f"❌ 数据目录不存在：{actual_data}")

        with c2:
            os.makedirs(actual_model, exist_ok=True)
            st.success(f"✅ 模型目录已就绪（{actual_model}）")

        st.divider()

        # Streamlit 端口
        st.markdown("### 🖥️ Streamlit 服务配置")
        col1, col2 = st.columns(2)
        port = col1.number_input("服务端口", value=int(env.get("STREAMLIT_SERVER_PORT", 8501)))
        addr = col2.text_input("监听地址", value=env.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"))
        st.caption(f"当前访问地址：http://localhost:{port}")

    # ---------- Tab 2: 模型参数 ----------
    with t_model:
        st.markdown("### 🧠 默认模型超参数")
        st.caption("这些参数会在模型训练页作为默认值使用")

        c1, c2 = st.columns(2)
        n_est = c1.slider("n_estimators", 50, 500, 200)
        max_d = c2.slider("max_depth", 2, 12, 6)
        lr = st.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01)

        st.divider()
        st.markdown("### ⚠️ 类别不平衡处理")
        st.info("✅ 已自动启用 scale_pos_weight 计算，无需手动设置")

        st.metric("scale_pos_weight", "自动按训练集比例计算")

    # ---------- Tab 3: LLM API ----------
    with t_llm:
        st.markdown("### 🔑 DeepSeek API 配置")

        api_key = st.text_input("API Key", value=env.get("DEEPSEEK_API_KEY", ""),
                                type="password", help="sk- 开头的 API Key")
        api_base = st.text_input("API Base URL", value=env.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
        model = st.text_input("模型名称", value=env.get("DEEPSEEK_MODEL", "deepseek-chat"))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if api_key and len(api_key) > 10 and api_key != "sk-your-api-key-here":
                st.success("✅ API Key 已配置")
                st.caption(f"Key 预览：{api_key[:8]}...{api_key[-4:]}")
            else:
                st.warning("⚠️ API Key 未配置")

        with col2:
            if st.button("🧪 测试连接", key="test_llm"):
                with st.spinner("测试 API 连通性..."):
                    try:
                        import requests
                        resp = requests.post(
                            f"{api_base}/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                            json={"model": model, "messages": [{"role": "user", "content": "你好"}], "max_tokens": 20},
                            timeout=15,
                        )
                        if resp.status_code == 200:
                            st.success("✅ API 连接成功！")
                        else:
                            st.error(f"❌ 返回 HTTP {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ 连接失败：{e}")

        st.divider()

        if st.button("💾 保存配置到 .env", type="primary"):
            try:
                # 读取现有 .env，更新对应行
                lines = []
                if os.path.exists(ENV_PATH):
                    with open(ENV_PATH, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                def upsert(key, val):
                    for i, ln in enumerate(lines):
                        if ln.strip().startswith(key + "="):
                            lines[i] = f"{key}={val}\n"
                            return
                    lines.append(f"{key}={val}\n")

                upsert("DEEPSEEK_API_KEY", api_key)
                upsert("DEEPSEEK_API_BASE", api_base)
                upsert("DEEPSEEK_MODEL", model)

                with open(ENV_PATH, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                st.success("✅ .env 已更新！请重启 Streamlit 使配置生效")
            except Exception as e:
                st.error(f"❌ 保存失败：{e}")

    # ---------- Tab 4: 关于 ----------
    with t_about:
        st.markdown("### ℹ️ 系统信息")

        c1, c2 = st.columns(2)
        c1.metric("系统版本", "v1.0.0")
        c2.metric("Python", __import__("sys").version.split()[0])

        st.divider()
        st.markdown("### 🧩 技术栈")
        st.markdown("""
        | 模块 | 技术 |
        |------|------|
        | Web 框架 | Streamlit |
        | 机器学习 | XGBoost / Scikit-learn |
        | 深度学习 | PyTorch (预留) |
        | 模型解释 | SHAP |
        | 超参优化 | Optuna (预留) |
        | LLM | DeepSeek API |
        | 可视化 | Plotly / Matplotlib |
        """)

        st.divider()
        st.markdown("### 📄 项目说明")
        st.info("""
        智能信贷风控分析系统 —— 基于 sample.csv（Lending Club 贷款数据）
        实现从数据探索、特征工程、模型训练、单笔预测、模型解释到 LLM 报告的
        完整端到端风控解决方案。

        🔗 数据源：`./data/sample.csv`（50,000 行 × 47 列）
        📌 目标变量：`isDefault`（1=违约，0=正常）
        """)

if __name__ == "__main__":
    render()