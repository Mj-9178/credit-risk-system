"""
数据探索页面
对应效果图：饼图 + 三个数字卡 + 警告框 + 缺失值表 + describe 表 + 特征分布图
"""

import os
import sys

import pandas as pd
import streamlit as st
import plotly.express as px

# 确保能 import 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "sample.csv")

# ========== 数据加载（带缓存，避免每次刷新都重读）==========
@st.cache_data
def load_data():
    """加载 sample.csv 并做基础类型清洗"""
    df = pd.read_csv(DATA_PATH)

    # employmentLength 是字符串（"2 years"），提取数字方便分组
    def parse_emp_len(x):
        if pd.isna(x):
            return None
        s = str(x)
        if s == "< 1 year":
            return 0.5
        if s == "10+ years":
            return 10
        try:
            return float(s.split()[0])
        except Exception:
            return None

    if "employmentLength" in df.columns:
        df["employmentLength_num"] = df["employmentLength"].apply(parse_emp_len)

    return df

# ========== 页面主渲染函数 ==========
def render():
    st.title("📊 数据探索")
    st.caption("快速理解信贷数据集的分布、缺失、类别不平衡情况")

    # ---------- 加载数据 ----------
    df = load_data()

    # ========== 1. 类别分布（饼图 + 数字卡片）==========
    st.markdown("### 📌 违约类别分布")

    col_pie, col_stats = st.columns([2, 1])

    with col_pie:
        # 饼图
        class_counts = df["isDefault"].value_counts().rename({0: "正常", 1: "违约"})
        fig_pie = px.pie(
            names=class_counts.index,
            values=class_counts.values,
            color=class_counts.index,
            color_discrete_map={"正常": "#28a745", "违约": "#dc3545"},
            hole=0.4,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            textfont=dict(size=14),
            pull=[0.02, 0.05],
        )
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_stats:
        total = len(df)
        default_n = int(df["isDefault"].sum())
        normal_n = total - default_n
        ratio = round(normal_n / default_n, 2) if default_n > 0 else 0

        st.metric("样本总数", f"{total:,}")
        st.metric("违约样本数", f"{default_n:,}")
        st.metric("不平衡比例", f"1 : {ratio}")

    # ---------- 警告提示 ----------
    st.warning(
        "⚠️ **数据存在类别不平衡**，建议在模型训练时启用：\n\n"
        "- **XGBoost**: `scale_pos_weight` 参数设为 " + str(round(ratio, 2)) + "\n"
        "- **MLP**: 使用 `class_weight` 或 Focal Loss\n"
        "- **评估**: 重点看 AUC、F1、Recall，而非 Accuracy"
    )

    st.divider()

    # ========== 2. 缺失值统计 ==========
    st.markdown("### 🧩 缺失值统计")

    missing_df = pd.DataFrame({
        "缺失数": df.isnull().sum(),
        "缺失率": (df.isnull().sum() / len(df) * 100).round(2),
    })
    missing_df = missing_df[missing_df["缺失数"] > 0].sort_values("缺失率", ascending=False)
    missing_df["字段"] = missing_df.index

    if len(missing_df) > 0:
        # 只展示前 15 个最严重的缺失字段
        st.dataframe(
            missing_df[["字段", "缺失数", "缺失率"]].head(15),
            use_container_width=True,
            hide_index=True,
            column_config={
                "缺失率": st.column_config.NumberColumn(format="%.2f %%"),
            },
        )
    else:
        st.success("✅ 数据无缺失值！")

    st.divider()

    # ========== 3. 数值特征统计（describe）==========
    with st.expander("📈 数值特征统计（describe）", expanded=False):
        numeric_df = df.select_dtypes(include=["number"]).drop(columns=["isDefault"], errors="ignore")
        st.dataframe(
            numeric_df.describe().round(2),
            use_container_width=True,
        )

    st.divider()

    # ========== 4. 特征分布图（按 isDefault 区分）==========
    st.markdown("### 📉 特征分布（按违约标签区分）")

    numeric_cols = df.select_dtypes(include=["number"]).drop(
        columns=["isDefault", "id"], errors="ignore"
    ).columns.tolist()

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_feature = st.selectbox(
            "选择特征",
            options=numeric_cols,
            index=numeric_cols.index("loanAmnt") if "loanAmnt" in numeric_cols else 0,
        )

    fig_hist = px.histogram(
        df,
        x=selected_feature,
        color="isDefault",
        barmode="overlay",
        nbins=50,
        color_discrete_map={0: "#4A90E2", 1: "#e67e22"},
        labels={"isDefault": "isDefault", selected_feature: selected_feature},
        title=f"{selected_feature} 的分布（按 isDefault 区分）",
    )
    fig_hist.update_layout(height=400, barmode="overlay")
    st.plotly_chart(fig_hist, use_container_width=True)

# ========== 独立运行入口 ==========
if __name__ == "__main__":
    render()