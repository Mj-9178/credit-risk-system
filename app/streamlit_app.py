"""
智能信贷风控分析系统 - 主程序入口（首页）

运行方式：
    streamlit run app/streamlit_app.py --server.port 8501
"""

import os
import sys
import streamlit as st
from streamlit_option_menu import option_menu

# ---- 确保能 import 项目根目录的模块 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能信贷风控分析系统",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 加载自定义 CSS =====
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ===== 侧边栏导航 =====
with st.sidebar:
    st.markdown("### 📊 Streamlit App")

    selected = option_menu(
        menu_title=None,
        options=[
            "数据探索",
            "特征工程",
            "模型训练",
            "单笔预测",
            "模型解释",
            "LLM智能助手",
            "系统配置",
        ],
        icons=[
            "clipboard-data",
            "tools",
            "diagram-3",
            "123",
            "search",
            "robot",
            "gear",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "6px 0",
                "padding": "8px 14px",
                "border-radius": "8px",
                "color": "#333",
                "transition": "all 0.2s",
            },
            "nav-link-hover": {
                "background-color": "#e8f0fe",
                "color": "#1a73e8",
            },
            "nav-link-selected": {
                "background-color": "#4A90E2",
                "color": "white",
                "font-weight": "bold",
            },
        },
    )

# ===== 路由：根据侧边栏选择，渲染首页或跳转到对应页面 =====

if selected == "数据探索":
    from app.pages.数据探索 import render
    render()
    st.stop()

# 如果点击了其他页面，也加同样的路由（暂放占位）
elif selected == "特征工程":
    from app.pages.特征工程 import render
    render()
    st.stop()
elif selected == "模型训练":
    from app.pages.模型训练 import render
    render()
    st.stop()
elif selected == "单笔预测":
    from app.pages.单笔预测 import render
    render()
    st.stop()
elif selected == "模型解释":
    from app.pages.模型解释 import render
    render()
    st.stop()
elif selected == "LLM智能助手":
    from app.pages.LLM智能助手 import render
    render()
    st.stop()
elif selected == "系统配置":
    from app.pages.系统配置 import render
    render()
    st.stop()

# ===== 下面是首页内容（只有在没点击侧边栏任何项时才显示）=====

# ===== 顶部大标题 =====
st.markdown(
    """
    <div style="padding: 1rem 0 2rem 0;">
        <div class="main-title">💳 智能信贷风控分析系统</div>
        <div class="subtitle">
            基于 <b style="color:#4A90E2">XGBoost</b> + 
            <b style="color:#e67e22">PyTorch MLP</b> + 
            <b style="color:#28a745">SHAP</b> + 
            <b style="color:#8e44ad">DeepSeek LLM</b> 的端到端风控解决方案
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===== 四个状态卡片（分 4 列）=====
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="status-card">
            <div class="card-title"><span class="card-icon">📦</span> 数据状态</div>
            <div class="card-value">已就绪</div>
            <div><span class="tag-ok">↑ ✅ OK</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="status-card">
            <div class="card-title"><span class="card-icon">🧠</span> 模型状态</div>
            <div class="card-value">已训练</div>
            <div><span class="tag-ok">↑ ✅ OK</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="status-card">
            <div class="card-title"><span class="card-icon">🔑</span> LLM API</div>
            <div class="card-value">未配置</div>
            <div><span class="tag-warn">↑ ⚠️ 需API...</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="status-card">
            <div class="card-title"><span class="card-icon">⚙️</span> 系统版本</div>
            <div class="card-value">v1.0.0</div>
            <div><span class="tag-info">↑ 教学版</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.success("✨ 首页已渲染成功！点击左侧侧边栏的选项进入对应功能页面")