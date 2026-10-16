"""
特征工程页面
展示：缺失填充 → 类别编码 → 特征选择 → 相关性分析 → 处理后预览
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "sample.csv")

# ========== 数据加载 ==========
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)

# ========== 预处理后的完整特征表 ==========
@st.cache_data
def build_engineered_features():
    """返回 (engineered_df, feature_importance_df)"""
    df = pd.read_csv(DATA_PATH)
    target = df["isDefault"].values

    work = df.drop(columns=["isDefault", "id", "issueDate", "earliesCreditLine", "postCode", "title"], errors="ignore").copy()

    # --- 1. 缺失值填充 ---
    for col in work.select_dtypes(include=[np.number]).columns:
        work[col] = work[col].fillna(work[col].median())
    for col in work.select_dtypes(include=["object"]).columns:
        work[col] = work[col].fillna("Unknown")

    # --- 2. 类别编码（LabelEncoder）---
    cat_cols = work.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    # --- 3. SelectKBest 特征选择（取 top 20）---
    selector = SelectKBest(score_func=f_classif, k=min(20, len(work.columns)))
    selector.fit(work, target)
    selected_mask = selector.get_support()
    selected_features = work.columns[selected_mask].tolist()

    # --- 4. 计算重要性得分 ---
    scores = selector.scores_
    importance_df = pd.DataFrame({
        "feature": work.columns,
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    return work, importance_df, cat_cols, selected_features

# ========== 页面渲染 ==========
def render():
    st.title("🔧 特征工程")
    st.caption("从原始信贷数据中挖掘、清洗、筛选出适合建模的特征")

    # 加载数据
    raw_df = load_raw_data()
    work_df, imp_df, cat_cols, top_features = build_engineered_features()

    # ========== 1. 数据概览（Before vs After）==========
    st.markdown("### 📦 输入数据 → 特征工程 → 建模数据")

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**🧾 原始数据**")
        st.metric("行数", f"{len(raw_df):,}")
        st.metric("列数", raw_df.shape[1])
        st.metric("缺失字段", raw_df.isnull().sum().gt(0).sum())
        st.metric("类别字段", raw_df.select_dtypes(include=["object"]).shape[1])

    with col_after:
        st.markdown("**✨ 特征工程后**")
        st.metric("行数", f"{len(work_df):,}")
        st.metric("列数", len(work_df.columns))
        st.metric("缺失字段", 0)
        st.metric("已编码字段", len(cat_cols))
        st.metric("Top20 选中特征", len(top_features))

    st.divider()

    # ========== 2. 缺失值处理策略 ==========
    st.markdown("### 🧩 缺失值填充策略")

    missing_counts = raw_df.isnull().sum()
    missing_info = pd.DataFrame({
        "缺失数": missing_counts,
        "缺失率": (missing_counts / len(raw_df) * 100).round(2),
    })
    missing_info["填充策略"] = missing_info["缺失数"].apply(
        lambda x: "中位数填充" if x > 0 else "无缺失"
    )
    missing_info = missing_info[missing_info["缺失数"] > 0].sort_values("缺失率", ascending=False)
    missing_info["字段"] = missing_info.index

    if len(missing_info) > 0:
        st.dataframe(
            missing_info[["字段", "缺失数", "缺失率", "填充策略"]].head(15),
            use_container_width=True,
            hide_index=True,
            column_config={
                "缺失率": st.column_config.NumberColumn(format="%.2f %%"),
            },
        )
        st.success("✅ 所有数值字段使用 **中位数** 填充，避免极端值影响；类别字段用 `Unknown` 标记")

    st.divider()

    # ========== 3. 类别特征编码 ==========
    st.markdown("### 🏷️ 类别特征编码（LabelEncoder）")

    if len(cat_cols) > 0:
        # 展示第一个类别字段的编码映射
        demo_col = cat_cols[0]
        st.markdown(f"**示例：`{demo_col}` 的编码映射**")
        le_demo = LabelEncoder()
        le_demo.fit_transform(raw_df[demo_col].fillna("Unknown").astype(str).values)
        mapping_df = pd.DataFrame({
            "原始值": le_demo.classes_,
            "编码后": le_demo.transform(le_demo.classes_),
        })
        st.dataframe(mapping_df.head(20), use_container_width=True, hide_index=True)

        st.info(f"共 **{len(cat_cols)}** 个类别字段已做 LabelEncoder：{', '.join(cat_cols)}")

    st.divider()

    # ========== 4. 特征重要性排名（SelectKBest F-score）==========
    st.markdown("### 🏆 特征重要性排名（SelectKBest F-score）")

    col_top, col_chart = st.columns([1, 2])
    with col_top:
        st.dataframe(
            imp_df.head(20).round(2),
            use_container_width=True,
            hide_index=True,
        )

    with col_chart:
        fig_bar = px.bar(
            imp_df.head(20),
            x="score",
            y="feature",
            orientation="h",
            color="score",
            color_continuous_scale="Blues",
            title="Top20 特征重要性",
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=520)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ========== 5. 相关性热力图（Top 10）==========
    st.markdown("### 🔥 Top10 特征相关性热力图")

    top10 = imp_df.head(10)["feature"].tolist()
    corr_matrix = work_df[top10].corr()

    fig_heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values.round(2),
        x=top10,
        y=top10,
        colorscale="Blues",
        showscale=True,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 11},
    ))
    fig_heatmap.update_layout(height=520, width=780)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # ========== 6. 处理后数据预览 ==========
    with st.expander("👀 处理后数据预览（前 5 行）", expanded=False):
        st.dataframe(work_df.head(), use_container_width=True)

if __name__ == "__main__":
    render()