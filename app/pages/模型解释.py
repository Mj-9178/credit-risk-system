"""
模型解释页面 - 精简版
Tab: 全局解释 / 单样本解释 / 模型自带重要性
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "sample.csv")

@st.cache_data
def prepare_data():
    df = pd.read_csv(DATA_PATH)
    target = df["isDefault"].values
    work = df.drop(columns=["isDefault","id","issueDate","earliesCreditLine","postCode","title"], errors="ignore").copy()
    for col in work.select_dtypes(include=[np.number]).columns:
        work[col] = work[col].fillna(work[col].median())
    for col in work.select_dtypes(include=["object"]).columns:
        work[col] = work[col].fillna("Unknown")
    from sklearn.preprocessing import LabelEncoder
    for col in work.select_dtypes(include=["object"]).columns:
        work[col] = LabelEncoder().fit_transform(work[col].astype(str))
    return work.values, target, work.columns.tolist()

@st.cache_resource
def get_model():
    X, y, names = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, random_state=42,
                          use_label_encoder=False, eval_metric="logloss")
    m.fit(X_train, y_train)
    return m, X_test, y_test, names

def render():
    st.title("🔍 模型解释")
    st.caption("全局重要性 + 局部样本解释 + 模型自带重要性")

    with st.spinner("加载模型..."):
        model, X_test, y_test, feat_names = get_model()

    t1, t2, t3 = st.tabs(["🔵 SHAP 全局", "🟠 单样本解释", "⬜ 自带重要性"])

    # ---------- Tab 1 ----------
    with t1:
        st.markdown("### 🌐 SHAP 全局特征重要性")
        try:
            import shap
            with st.spinner("计算 SHAP..."):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test[:500])
            imp = pd.DataFrame({"feature": feat_names, "shap_imp": np.abs(sv).mean(axis=0)})
            imp = imp.sort_values("shap_imp", ascending=False)

            c1, c2 = st.columns([1, 2])
            c1.dataframe(imp.head(20).round(4), use_container_width=True, hide_index=True)
            fig = px.bar(imp.head(20).iloc[::-1], x="shap_imp", y="feature",
                        orientation="h", color="shap_imp", color_continuous_scale="Blues")
            fig.update_layout(height=520)
            c2.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"出错：{e}")

    # ---------- Tab 2 ----------
    with t2:
        st.markdown("### 🎯 单样本 SHAP 解释")
        idx = st.number_input("选择样本索引", min_value=0, max_value=len(X_test)-1, value=2, step=1)

        if st.button("🔍 解释此样本", type="primary", use_container_width=True):
            import shap
            explainer = shap.TreeExplainer(model)
            sample = X_test[idx].reshape(1, -1)
            sv = explainer.shap_values(sample)[0]
            base = explainer.expected_value
            proba = model.predict_proba(sample)[0]

            st.divider()
            col1, col2 = st.columns(2)
            label = "⚠️ 违约" if proba[1] >= 0.5 else "✅ 正常"
            color = "#dc3545" if proba[1] >= 0.5 else "#28a745"
            col1.markdown(f"**预测标签**: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            col1.metric("违约概率", f"{proba[1]:.4f}")
            col2.metric("真实标签", f"{int(y_test[idx])}")
            col2.metric("SHAP 基线值", f"{base:.4f}")

            # 瀑布图
            top_n = 12
            order = np.argsort(np.abs(sv))[::-1][:top_n]
            labels = [feat_names[i] for i in order]
            sv_top = sv[order]
            cumul = [base]
            for s in sv_top:
                cumul.append(cumul[-1] + s)

            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(x=[base], y=["基线"], orientation="h", marker_color="#888", showlegend=False))
            for i, (f, s) in enumerate(zip(labels, sv_top)):
                color = "#e74c3c" if s > 0 else "#2ecc71"
                fig_w.add_trace(go.Bar(x=[s], y=[f], orientation="h", marker_color=color,
                                      base=cumul[i], text=f"{s:.4f}", textposition="outside", showlegend=False))
            fig_w.add_trace(go.Bar(x=[cumul[-1]], y=["最终预测"], orientation="h",
                                  marker_color="#4A90E2", showlegend=False))
            fig_w.update_layout(title=f"样本 #{idx} SHAP 贡献（红=增加/绿=降低违约）",
                                height=550, barmode="stack", xaxis_title="log-odds")
            st.plotly_chart(fig_w, use_container_width=True)

            # 贡献表
            contrib = pd.DataFrame({
                "feature": labels,
                "shap_value": sv_top.round(4),
                "direction": ["↑ 增加风险" if s > 0 else "↓ 降低风险" for s in sv_top],
            })
            st.dataframe(contrib, use_container_width=True, hide_index=True)

    # ---------- Tab 3 ----------
    with t3:
        st.markdown("### 📦 XGBoost 自带特征重要性")
        imp = pd.DataFrame({
            "feature": feat_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        c1, c2 = st.columns([1, 2])
        c1.dataframe(imp.head(20).round(4), use_container_width=True, hide_index=True)
        fig = px.bar(imp.head(20).iloc[::-1], x="importance", y="feature",
                    orientation="h", color="importance", color_continuous_scale="Greens")
        fig.update_layout(height=520)
        c2.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    render()