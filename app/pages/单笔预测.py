"""
单笔贷款风险评估
三栏表单 + 仪表盘 + SHAP 瀑布图 + 特征贡献表 + LLM 报告
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "sample.csv")

# ========== 数据加载 ==========
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
    cat_maps = {}
    for col in work.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        cat_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    return work.values, target, work.columns.tolist(), cat_maps, work

@st.cache_resource
def get_model():
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    X, y, names, _, _ = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    spw = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, random_state=42,
                          use_label_encoder=False, eval_metric="logloss")
    m.fit(X_train, y_train)
    return m, names

# ========== 表单默认值（参考 sample.csv 均值）==========
DEFAULTS = {
    "loanAmnt": 15000.0, "term": 3, "interestRate": 13.5,
    "employmentLength": "4 years", "homeOwnership": "按揭",
    "annualIncome": 80000.0, "verificationStatus": "源已验证",
    "delinquency_2years": 0, "ficoRangeLow": 690,
    "revolUtil": 50.0, "pubRecBankruptcies": 0,
    "revolBal": 2321.0, "dti": 15.0,
    "grade": "C", "purpose": 0,
}

def render():
    st.title("🎯 单笔贷款风险评估")
    st.caption("录入信息 → 模型预测 → SHAP 解释 → LLM 报告")

    model, feat_names = get_model()

    # ========== 表单：三栏 Tab ==========
    st.markdown("### 📝 申请人信息")
    tab1, tab2, tab3 = st.tabs(["💰 贷款信息", "👤 个人信息", "📊 信用信息"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        d_loan = c1.number_input("贷款金额 (元)", value=DEFAULTS["loanAmnt"], step=1000.0)
        d_term = c2.selectbox("贷款期限 (年)", [3, 5], index=[3,5].index(DEFAULTS["term"]))
        d_grade = c3.selectbox("贷款等级", ["A","B","C","D","E","F","G"], index=["A","B","C","D","E","F","G"].index(DEFAULTS["grade"]))
        d_int = st.slider("利率 (%)", 5.0, 30.0, DEFAULTS["interestRate"], step=0.5)
        d_purpose = st.number_input("贷款用途代码", value=DEFAULTS["purpose"], step=1)

    with tab2:
        c1, c2, c3 = st.columns(3)
        d_income = c1.number_input("年收入 (元)", value=DEFAULTS["annualIncome"], step=5000.0)
        d_emp = c2.selectbox("就业年限", ["< 1 year","1 year","2 years","3 years","4 years","5 years","6 years","7 years","8 years","9 years","10+ years"],
                            index=4)
        d_home = c3.selectbox("房屋所有权", ["按揭","自有","租房","其它"], index=0)
        d_ver = st.selectbox("验证状态", ["未验证","收入已验证","源已验证"], index=2)

    with tab3:
        c1, c2 = st.columns(2)
        d_fico = c1.slider("FICO 分", 300, 850, DEFAULTS["ficoRangeLow"])
        d_delinq = c2.number_input("近 2 年逾期次数", value=DEFAULTS["delinquency_2years"], step=1, min_value=0)
        c1, c2 = st.columns(2)
        d_revol = c1.slider("循环额度利用率 (%)", 0.0, 100.0, DEFAULTS["revolUtil"])
        d_bankrupt = c2.number_input("破产记录", value=DEFAULTS["pubRecBankruptcies"], step=1, min_value=0)
        d_bal = st.number_input("循环余额", value=DEFAULTS["revolBal"], step=500.0)
        d_dti = st.slider("负债率 DTI (%)", 0.0, 50.0, DEFAULTS["dti"])

    # ========== 评估按钮 ==========
    if st.button("🚀 开始风险评估", type="primary", use_container_width=True):
        with st.spinner("模型预测 + SHAP 解释中..."):
            import shap

                        # 组装成与训练数据一致的特征向量
            X, _, feat_names_real, cat_maps, _ = prepare_data()

            # UI 显示值 → 训练数据原始值 的映射表
            UI_TO_TRAIN = {
                "homeOwnership": {"按揭": 1, "自有": 0, "租房": 2, "其它": 3},
                "verificationStatus": {"未验证": 0, "收入已验证": 1, "源已验证": 2},
            }

            def ui2train(field, val):
                if field in UI_TO_TRAIN:
                    return UI_TO_TRAIN[field].get(val, 0)
                return val

            sample_dict = {
                "loanAmnt": d_loan, "term": d_term, "interestRate": d_int,
                "employmentLength": ui2train("employmentLength", d_emp),
                "homeOwnership": ui2train("homeOwnership", d_home),
                "annualIncome": d_income,
                "verificationStatus": ui2train("verificationStatus", d_ver),
                "delinquency_2years": d_delinq, "ficoRangeLow": d_fico,
                "revolUtil": d_revol, "pubRecBankruptcies": d_bankrupt,
                "revolBal": d_bal, "dti": d_dti,
                "grade": d_grade, "purpose": d_purpose,
            }

            # 构造一行特征（按 feat_names_real 顺序）
            row = []
            for f in feat_names_real:
                if f in sample_dict:
                    v = sample_dict[f]
                    if f in cat_maps and isinstance(v, str):
                        v = cat_maps[f].get(v, 0)
                    row.append(v)
                else:
                    row.append(0.0)

            X_sample = np.array(row, dtype=float).reshape(1, -1)

            # 预测
            proba = model.predict_proba(X_sample)[0]
            default_p = proba[1]

            # SHAP
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)[0]
            base = explainer.expected_value

        # ========== 结果展示 ==========
        st.divider()
        st.markdown("### 📊 评估结果")

        # 仪表盘
        col_gauge, col_level = st.columns([2, 1])
        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_p * 100,
                title={"text": "违约概率 (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if default_p > 0.5 else "green"},
                    "steps": [
                        {"range": [0, 30], "color": "#28a745"},
                        {"range": [30, 60], "color": "#ffc107"},
                        {"range": [60, 100], "color": "#dc3545"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.75, "value": 50},
                },
                number={"suffix": "%"},
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_level:
            if default_p < 0.3:
                level, lc, advise = "低风险", "#28a745", "可直接审批"
            elif default_p < 0.6:
                level, lc, advise = "中风险", "#f39c12", "⚠️ 人工复核"
            else:
                level, lc, advise = "高风险", "#dc3545", "❌ 建议拒绝"

            st.markdown(
                f"""
                <div style="text-align:center;padding:1.5rem;border:2px solid {lc};border-radius:12px;">
                    <div style="font-size:1.3rem;color:#666;margin-bottom:0.5rem;">⚠️ 人工复核</div>
                    <div style="font-size:2.2rem;font-weight:bold;color:{lc};">{level}</div>
                    <div style="margin-top:1rem;font-size:0.95rem;">违约概率：<b>{default_p:.2%}</b></div>
                    <div style="margin-top:0.3rem;font-size:0.95rem;">建议月供：<b>¥{d_loan / d_term / 12:,.2f}</b></div>
                    <div style="margin-top:1rem;padding:0.6rem;background:{lc}15;border-radius:8px;color:{lc};">
                        📌 {advise}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # SHAP 瀑布图
        st.markdown("#### 💦 SHAP 特征贡献（瀑布图）")
        top_n = 12
        order = np.argsort(np.abs(sv))[::-1][:top_n]
        labels = [feat_names_real[i] for i in order]
        sv_top = sv[order]
        cumul = [base]
        for s in sv_top:
            cumul.append(cumul[-1] + s)

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=[base], y=["基线"], orientation="h", marker_color="#888", showlegend=False))
        for i, (f, s) in enumerate(zip(labels, sv_top)):
            c = "#e74c3c" if s > 0 else "#2ecc71"
            fig_w.add_trace(go.Bar(x=[s], y=[f], orientation="h", marker_color=c,
                                  base=cumul[i], text=f"{s:.4f}", textposition="outside", showlegend=False))
        fig_w.add_trace(go.Bar(x=[cumul[-1]], y=["最终预测"], orientation="h",
                              marker_color="#4A90E2", showlegend=False))
        fig_w.update_layout(height=550, barmode="stack", xaxis_title="log-odds")
        st.plotly_chart(fig_w, use_container_width=True)

        # 关键特征贡献表
        st.divider()
        st.markdown("#### 📋 关键特征贡献")
        contrib = pd.DataFrame({
            "feature": labels,
            "value": [X_sample[0, feat_names_real.index(l)] if l in feat_names_real else 0 for l in labels],
            "shap_value": sv_top.round(4),
            "direction": ["↑ 增加风险" if s > 0 else "↓ 降低风险" for s in sv_top],
        })
        st.dataframe(contrib, use_container_width=True, hide_index=True)

        # ========== LLM 报告 ==========
        st.divider()
        st.markdown("### 📝 智能风险评估报告")

        if st.button("◎ 调用 LLM 生成...", type="primary", key="btn_llm"):
            llm_report = call_llm_report(sample_dict, default_p, level, labels, sv_top)
            st.markdown(llm_report)

# ========== LLM 报告生成 ==========
def call_llm_report(sample, default_p, level, top_feats, top_sv):
    """尝试调用 DeepSeek API；失败则返回本地模板"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not api_key or api_key == "sk-your-api-key-here":
            raise ValueError("API Key 未配置")

        import requests
        prompt = f"""你是一个资深信贷风控专家。基于以下数据生成一份风险评估报告：

【借款人信息】：贷款 {sample.get('loanAmnt', '?')} 元 / {sample.get('term')}年 / 利率 {sample.get('interestRate')}%
年收入 {sample.get('annualIncome')} 元 / 就业 {sample.get('employmentLength')} / FICO {sample.get('ficoRangeLow')}
DTI {sample.get('dti')}% / 循环利用率 {sample.get('revolUtil')}%

【模型预测】：违约概率 {default_p:.2%}，风险等级 {level}

【关键 SHAP 因素】：{list(zip(top_feats[:5], top_sv[:5]))}

请输出 Markdown 格式的报告，包含：1. 总体结论 2. 主要风险因素 3. 建议（审批/补充材料/拒绝）"""

        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        # 本地模板（占位报告）
        return f"""### 📊 风控评估报告（模板）

> ⚠️ **未调用 LLM**：{e}
> 请在 `.env` 中配置 `DEEPSEEK_API_KEY` 以获取真实 AI 报告。

---

#### 1️⃣ 总体结论
- **风险等级**：{level}
- **违约概率**：{default_p:.2%}

#### 2️⃣ 主要风险因素（SHAP 贡献 Top5）
| 特征 | SHAP 值 | 方向 |
|------|---------|------|
{chr(10).join(f"| {f} | {s:.4f} | {'↑ 增加' if s>0 else '↓ 降低'} |" for f, s in zip(top_feats[:5], top_sv[:5]))}

#### 3️⃣ 建议
- 考虑到违约概率 **{default_p:.2%}**，建议 **人工复核后决策**
- 重点关注 `{top_feats[0] if len(top_feats)>0 else '核心指标'}` 异常情况"""

if __name__ == "__main__":
    render()