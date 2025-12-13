import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

# --- 页面配置 ---
st.set_page_config(page_title="智能统计助手", layout="wide")

st.title("📊 智能统计分析与方法推荐 App")
st.markdown("""
本工具支持 **2-4组数据** 的 **单因素/双因素** 分析。
流程：上传数据 -> 自动进行正态性/方差齐性检验 -> 生成诊断图 (QQ图/残差图) -> **推荐统计方法**。
""")

# --- 侧边栏：数据上传与设置 ---
st.sidebar.header("1. 数据设置")
uploaded_file = st.sidebar.file_uploader("上传 Excel 或 CSV 文件", type=["xlsx", "csv"])

# 示例数据生成（方便用户测试）
if st.sidebar.button("使用示例数据测试"):
    # 生成一个模拟的单因素3组数据
    np.random.seed(42)
    df_demo = pd.DataFrame({
        'Group': ['A']*20 + ['B']*20 + ['C']*20,
        'Value': np.concatenate([np.random.normal(10, 2, 20), np.random.normal(12, 2.5, 20), np.random.normal(11, 2, 20)])
    })
    uploaded_file = df_demo

def load_data(file):
    if isinstance(file, pd.DataFrame):
        return file
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# --- 主逻辑 ---
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### 数据预览", df.head())

    # 模式选择
    analysis_type = st.sidebar.radio("选择分析类型", ["单因素分析 (One-Way)", "双因素分析 (Two-Way)"])
    
    # 变量选择
    cols = df.columns.tolist()
    num_col = st.sidebar.selectbox("选择数值变量 (Dependent Variable)", cols, index=len(cols)-1)
    
    if analysis_type == "单因素分析 (One-Way)":
        cat_col = st.sidebar.selectbox("选择分组变量 (Factor)", [c for c in cols if c != num_col])
        groups = df[cat_col].unique()
        st.write(f"**检测到分组:** {groups} (共 {len(groups)} 组)")
        
        if len(groups) < 2:
            st.error("分组数量必须大于等于 2！")
            st.stop()

        # --- 1. 数据准备 ---
        group_data = [df[df[cat_col] == g][num_col].dropna() for g in groups]
        
        # --- 2. 假设检验 (Assumption Checks) ---
        st.header("2. 假设检验与诊断图")
        
        col1, col2 = st.columns(2)
        
        # A. 正态性检验 (Shapiro-Wilk)
        normality_results = {}
        all_normal = True
        with col1:
            st.subheader("正态性检验 (Shapiro-Wilk)")
            st.info("P > 0.05 表示符合正态分布")
            for i, g_name in enumerate(groups):
                stat, p = stats.shapiro(group_data[i])
                is_norm = p > 0.05
                if not is_norm: all_normal = False
                normality_results[g_name] = is_norm
                st.write(f"- **{g_name}**: P-value={p:.4f} ({'正态' if is_norm else '非正态'})")

        # B. 方差齐性检验 (Levene test)
        with col2:
            st.subheader("方差齐性检验 (Levene)")
            st.info("P > 0.05 表示方差齐")
            stat_lev, p_lev = stats.levene(*group_data)
            is_homoscedastic = p_lev > 0.05
            st.write(f"- **整体**: P-value={p_lev:.4f} ({'方差齐' if is_homoscedastic else '方差不齐'})")

        # C. 可视化诊断
        st.subheader("可视化诊断")
        tab1, tab2, tab3 = st.tabs(["QQ图", "残差图", "箱线图"])
        
        with tab1:
            # QQ Plot
            fig_qq, axes = plt.subplots(1, len(groups), figsize=(15, 5))
            if len(groups) == 1: axes = [axes] # Handle single plot case
            for i, g_name in enumerate(groups):
                sm.qqplot(group_data[i], line='s', ax=axes[i])
                axes[i].set_title(f"QQ Plot: {g_name}")
            st.pyplot(fig_qq)
            
        with tab2:
            # Residual Plot (Value - Mean)
            fig_res, ax = plt.subplots(figsize=(8, 5))
            residuals = []
            fitted = []
            for i, g_name in enumerate(groups):
                mean_val = group_data[i].mean()
                res = group_data[i] - mean_val
                residuals.extend(res)
                fitted.extend([mean_val]*len(res))
            
            sns.scatterplot(x=fitted, y=residuals, ax=ax)
            ax.axhline(0, color='r', linestyle='--')
            ax.set_xlabel("Fitted Values (Group Means)")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            st.pyplot(fig_res)

        with tab3:
            fig_box, ax = plt.subplots()
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
            sns.stripplot(x=cat_col, y=num_col, data=df, color='black', alpha=0.5, ax=ax)
            st.pyplot(fig_box)

        # --- 3. 智能推荐逻辑 ---
        st.header("3. 统计方法推荐与结果")
        
        recommendation = ""
        method_code = ""

        # 逻辑树
        if len(groups) == 2:
            if all_normal and is_homoscedastic:
                recommendation = "✅ 推荐方法：独立样本 T 检验 (Student's t-test)"
                res = stats.ttest_ind(group_data[0], group_data[1])
                method_code = f"T-statistic: {res.statistic:.3f}, P-value: {res.pvalue:.4f}"
            elif all_normal and not is_homoscedastic:
                recommendation = "✅ 推荐方法：Welch's T 检验 (校正方差不齐)"
                res = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                method_code = f"T-statistic: {res.statistic:.3f}, P-value: {res.pvalue:.4f}"
            else:
                recommendation = "✅ 推荐方法：Mann-Whitney U 检验 (非参数检验)"
                res = stats.mannwhitneyu(group_data[0], group_data[1])
                method_code = f"U-statistic: {res.statistic:.3f}, P-value: {res.pvalue:.4f}"
        
        elif len(groups) > 2:
            if all_normal and is_homoscedastic:
                recommendation = "✅ 推荐方法：单因素方差分析 (One-Way ANOVA)"
                res = stats.f_oneway(*group_data)
                method_code = f"F-statistic: {res.statistic:.3f}, P-value: {res.pvalue:.4f}"
            elif all_normal and not is_homoscedastic:
                recommendation = "✅ 推荐方法：Welch's ANOVA (校正方差不齐)"
                # Scipy doesn't have Welch ANOVA easily, suggest pingouin or use statsmodels logic generally
                method_code = "建议使用 Welch ANOVA (本简易版暂仅展示普通ANOVA结果供参考，请注意P值可能不准)"
                res = stats.f_oneway(*group_data) # Fallback for demo
            else:
                recommendation = "✅ 推荐方法：Kruskal-Wallis H 检验 (非参数检验)"
                res = stats.kruskal(*group_data)
                method_code = f"H-statistic: {res.statistic:.3f}, P-value: {res.pvalue:.4f}"

        st.success(recommendation)
        st.code(method_code, language="text")
        
        if "P-value" in method_code:
            p_val = float(method_code.split("P-value: ")[1].split(")")[0] if ")" in method_code else method_code.split("P-value: ")[1])
            if p_val < 0.05:
                st.write("**结论：** 组间存在显著差异 (P < 0.05)，建议进行事后检验 (Post-hoc)。")
            else:
                st.write("**结论：** 组间无显著差异。")

    elif analysis_type == "双因素分析 (Two-Way)":
        factors = [c for c in cols if c != num_col]
        if len(factors) < 2:
            st.error("数据中至少需要两列作为分类变量才能进行双因素分析！")
            st.stop()
            
        f1 = st.sidebar.selectbox("选择因素 1", factors)
        f2 = st.sidebar.selectbox("选择因素 2", [c for c in factors if c != f1])
        
        st.write(f"正在分析：**{num_col}** ~ **{f1}** + **{f2}** + **{f1}:{f2}**")
        
        # 使用 Statsmodels 进行双因素 ANOVA
        # 需要构建公式字符串，处理列名中的特殊字符
        clean_col = "Value"
        clean_f1 = "Factor1"
        clean_f2 = "Factor2"
        
        temp_df = pd.DataFrame({
            clean_col: df[num_col],
            clean_f1: df[f1].astype(str),
            clean_f2: df[f2].astype(str)
        })
        
        model = ols(f'{clean_col} ~ C({clean_f1}) + C({clean_f2}) + C({clean_f1}):C({clean_f2})', data=temp_df).fit()
        
        # 检验正态性（基于残差）
        residuals = model.resid
        stat_shapiro, p_shapiro = stats.shapiro(residuals)
        
        st.header("2. 假设检验 (基于模型残差)")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**残差正态性 (Shapiro)**: P={p_shapiro:.4f}")
            if p_shapiro < 0.05:
                st.warning("警告：残差不符合正态分布，ANOVA结果可能不可靠。")
            else:
                st.success("残差符合正态分布。")
                
        with col2:
            st.write("**残差分布图**")
            fig_res, ax = plt.subplots(figsize=(6, 4))
            sm.qqplot(residuals, line='s', ax=ax)
            st.pyplot(fig_res)
            
        st.header("3. 分析结果 (Two-Way ANOVA)")
        anova_table = sm.stats.anova_lm(model, typ=2)
        st.dataframe(anova_table.style.format("{:.4f}"))
        
        st.info("""
        **解读指南：**
        1. 首先看交互项 (:) 的 P值。如果 P < 0.05，说明两个因素之间有交互作用，单独解释主效应可能不准确。
        2. 如果交互项不显著，则分别看两个主效应 (Factor1, Factor2) 的 P值。
        """)

else:
    st.info("请在左侧上传数据开始分析。")