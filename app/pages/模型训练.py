"""
模型训练页面
- Tab 切换：XGBoost / PyTorch MLP / 集成学习
- 超参数配置面板
- 训练进度 + 评估指标
- 历史训练记录表
"""

import os
import sys
import io
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "sample.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "train_history.csv")

# ========== 数据加载与预处理 ==========
@st.cache_data
def prepare_data():
    df = pd.read_csv(DATA_PATH)
    target = df["isDefault"].values

    work = df.drop(columns=["isDefault", "id", "issueDate", "earliesCreditLine", "postCode", "title"], errors="ignore").copy()

    # 缺失填充
    for col in work.select_dtypes(include=[np.number]).columns:
        work[col] = work[col].fillna(work[col].median())
    for col in work.select_dtypes(include=["object"]).columns:
        work[col] = work[col].fillna("Unknown")

    # 类别编码
    from sklearn.preprocessing import LabelEncoder
    cat_cols = work.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        work[col] = LabelEncoder().fit_transform(work[col].astype(str))

    return work.values, target, work.columns.tolist()

# ========== 历史记录管理 ==========
def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["运行ID", "时间", "模型类型", "训练样本", "特征数", "AUC", "F1", "Recall"])

def save_history(entry: dict):
    history = load_history()
    history = pd.concat([history, pd.DataFrame([entry])], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)

# ========== 训练主函数 ==========
def train_xgboost(X_train, y_train, X_test, y_test, params: dict):
    scale_pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
    }
    return model, metrics, y_pred, y_proba

# ========== 页面渲染 ==========
def render():
    st.title("🎯 模型训练")
    st.caption("支持 XGBoost / PyTorch MLP / 集成学习，一键训练 + 完整评估")

    X, y, feature_names = prepare_data()

    # ========== Tab 切换 ==========
    tab_xgb, tab_mlp, tab_ens = st.tabs(["🟢 XGBoost", "🔵 PyTorch MLP", "🟣 集成学习"])

    # ---------- Tab 1: XGBoost ----------
    with tab_xgb:
        st.markdown("### XGBoost 梯度提升树（推荐首选）")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**🔧 超参数配置**")
            n_estimators = st.slider("n_estimators（树的数量）", 50, 500, 200, step=10)
            max_depth = st.slider("max_depth（树深度）", 2, 12, 6)
            learning_rate = st.slider("learning_rate（学习率）", 0.01, 0.3, 0.1, step=0.01)
            subsample = st.slider("subsample（样本采样率）", 0.5, 1.0, 0.8, step=0.05)
            colsample_bytree = st.slider("colsample_bytree（特征采样率）", 0.5, 1.0, 0.8, step=0.05)

        with col_right:
            st.markdown("**📊 数据切分**")
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, step=0.05)
            random_state = st.number_input("随机种子", value=42)

            st.markdown("**📐 自动处理不平衡**")
            st.success(f"scale_pos_weight 自动计算：约 {np.sum(y == 0) / max(np.sum(y == 1), 1):.2f}")

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }

        # 训练按钮
        if st.button("🚀 开始训练 XGBoost", type="primary", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            progress = st.progress(0, text="准备训练...")
            time.sleep(0.3)
            progress.progress(30, text="正在拟合模型...")

            model, metrics, y_pred, y_proba = train_xgboost(X_train, y_train, X_test, y_test, params)

            progress.progress(90, text="保存模型...")

            # 保存模型
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(MODEL_DIR, f"xgb_{run_id}.json")
            model.save_model(model_path)

            # 保存历史
            save_history({
                "运行ID": run_id,
                "时间": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "模型类型": "XGBoost",
                "训练样本": len(X_train),
                "特征数": X.shape[1],
                "AUC": metrics["AUC"],
                "F1": metrics["F1"],
                "Recall": metrics["Recall"],
            })

            progress.progress(100, text="✅ 训练完成！")
            time.sleep(0.5)
            progress.empty()

            # 评估指标展示
            st.divider()
            st.markdown("### 📈 评估指标")

            m1, m2, m3 = st.columns(3)
            m1.metric("AUC", metrics["AUC"], delta_color="inverse")
            m2.metric("F1 Score", metrics["F1"])
            m3.metric("Recall", metrics["Recall"])

            st.divider()

            # ROC 曲线
            fpr_list, tpr_list, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(y_test, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_list, y=tpr_list, mode="lines", name=f"AUC={metrics['AUC']:.4f}"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="随机"))
            fig_roc.update_layout(title="ROC 曲线", xaxis_title="FPR", yaxis_title="TPR", height=400)
            st.plotly_chart(fig_roc, use_container_width=True)

            st.divider()

            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="预测值", y="真实值"),
                x=["正常", "违约"], y=["正常", "违约"],
                title="混淆矩阵",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.divider()

            # 特征重要性 top15
            importance = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(15)

            fig_imp = px.bar(importance, x="importance", y="feature", orientation="h",
                           color="importance", color_continuous_scale="Greens",
                           title="Top15 特征重要性")
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=450)
            st.plotly_chart(fig_imp, use_container_width=True)

            # 保存到 session_state 供其他页面使用
            st.session_state["trained_model"] = model
            st.session_state["trained_features"] = feature_names
            st.session_state["last_metrics"] = metrics

    # ---------- Tab 2: PyTorch MLP（简化占位，调用 sklearn MLPClassifier 代替，快速可跑）----------
    with tab_mlp:
        st.markdown("### 🔵 PyTorch MLP 深度学习模型")
        st.info("💡 当前演示版使用 `sklearn.neural_network.MLPClassifier` 快速训练，真实 PyTorch 版已在 `src/models/mlp_model.py` 预留接口")

        col1, col2 = st.columns([1, 1])
        with col1:
            hidden_layers = st.slider("隐藏层大小", 16, 256, 64, step=16, key="mlp_hidden")
            mlp_lr = st.slider("学习率", 0.001, 0.1, 0.01, step=0.001, key="mlp_lr")
            mlp_epochs = st.slider("最大迭代次数", 50, 500, 200, step=10, key="mlp_epochs")

        with col2:
            test_size_mlp = st.slider("测试集比例", 0.1, 0.4, 0.2, step=0.05, key="mlp_test")
            use_class_weight = st.checkbox("启用类权重（处理不平衡）", value=True, key="mlp_cw")

        if st.button("🚀 开始训练 MLP", type="primary", key="btn_mlp"):
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_mlp, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            progress = st.progress(0, text="正在训练 MLP...")
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_layers, hidden_layers // 2),
                learning_rate_init=mlp_lr,
                max_iter=mlp_epochs,
                random_state=42,
            )
            model.fit(X_train_s, y_train)
            progress.progress(100)

            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]

            metrics = {
                "AUC": round(roc_auc_score(y_test, y_proba), 4),
                "F1": round(f1_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
            }

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_history({
                "运行ID": run_id,
                "时间": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "模型类型": "MLP",
                "训练样本": len(X_train),
                "特征数": X.shape[1],
                "AUC": metrics["AUC"],
                "F1": metrics["F1"],
                "Recall": metrics["Recall"],
            })

            m1, m2, m3 = st.columns(3)
            m1.metric("AUC", metrics["AUC"])
            m2.metric("F1", metrics["F1"])
            m3.metric("Recall", metrics["Recall"])

            st.session_state["trained_model"] = model
            st.session_state["last_metrics"] = metrics

    # ---------- Tab 3: 集成学习 ----------
    with tab_ens:
        st.markdown("### 🟣 集成学习（XGBoost + MLP 投票）")
        st.info("先在上方 XGBoost 或 MLP Tab 中训练好模型，再回到这里做加权投票集成")

        st.metric("演示模式", "🎯 自动用默认参数快速训练两个基模型并集成")

        if st.button("🚀 一键训练集成模型", type="primary", key="btn_ens"):
            progress = st.progress(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # XGB
            xgb_m, _, _, _ = train_xgboost(X_train, y_train, X_test, y_test, {})
            progress.progress(40, text="XGBoost 完成")

            # MLP
            from sklearn.neural_network import MLPClassifier
            mlp_m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
            mlp_m.fit(X_train, y_train)
            progress.progress(70, text="MLP 完成")

            # 加权投票
            proba_xgb = xgb_m.predict_proba(X_test)[:, 1]
            proba_mlp = mlp_m.predict_proba(X_test)[:, 1]
            proba_ens = 0.6 * proba_xgb + 0.4 * proba_mlp
            y_pred_ens = (proba_ens >= 0.5).astype(int)

            metrics = {
                "AUC": round(roc_auc_score(y_test, proba_ens), 4),
                "F1": round(f1_score(y_test, y_pred_ens), 4),
                "Recall": round(recall_score(y_test, y_pred_ens), 4),
            }

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_history({
                "运行ID": run_id,
                "时间": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "模型类型": "Ensemble(XGB+MLP)",
                "训练样本": len(X_train),
                "特征数": X.shape[1],
                "AUC": metrics["AUC"],
                "F1": metrics["F1"],
                "Recall": metrics["Recall"],
            })

            progress.progress(100, text="✅ 集成训练完成")

            m1, m2, m3 = st.columns(3)
            m1.metric("AUC", metrics["AUC"])
            m2.metric("F1", metrics["F1"])
            m3.metric("Recall", metrics["Recall"])

    # ========== 历史训练记录表 ==========
    st.divider()
    st.markdown("### 📄 历史训练记录")

    history = load_history()
    if len(history) > 0:
        st.dataframe(history.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.warning("暂无训练记录，点击上方按钮开始第一次训练吧！")

if __name__ == "__main__":
    render()