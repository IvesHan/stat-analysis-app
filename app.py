import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- 0. 页面配置 (低调模式) ---
st.set_page_config(page_title="统计分析工具", layout="wide")
st.title("统计分析工具") # 标题改为通用名称

# --- 工具函数：解析手动输入数据 ---
def parse_manual_input(text_input, sep_char):
    try:
        data = StringIO(text_input)
        if sep_char == '逗号 (CSV)':
            df = pd.read_csv(data)
        elif sep_char == '制表符 (Excel复制)':
            df = pd.read_csv(data, sep='\t')
        elif sep_char == '空格':
            df = pd.read_csv(data, delim_whitespace=True)
        return df
    except Exception as e:
        return None

# --- 侧边栏：数据来源 ---
st.sidebar.header("1. 数据输入")
input_method = st.sidebar.radio("选择数据来源", ["上传文件", "手动输入/粘贴"])

df = None

if input_method == "上传文件":
    uploaded_file = st.sidebar.file_uploader("支持 xlsx, csv", type=["xlsx", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

elif input_method == "手动输入/粘贴":
    st.sidebar.info("请在首行包含列名")
    sep_mode = st.sidebar.selectbox("分隔符格式", ["制表符 (Excel复制)", "逗号 (CSV)", "空格"])
    text_data = st.sidebar.text_area("在此粘贴数据", height=150, 
                                     placeholder="Group\tValue\nA\t10.5\nA\t12.1\nB\t15.3\n...")
    if text_data:
        df = parse_manual_input(text_data, sep_mode)

# --- 主逻辑 ---
if df is not None:
    st.write("### 数据预览", df.head())
    cols = df.columns.tolist()

    # --- 变量设置 ---
    st.sidebar.header("2. 变量设置")
    
    # 智能推断：如果某一列是数值，设为Y；如果某一列不仅数值且重复值多，可能是分组
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols] + [c for c in num_cols if df[c].nunique() < 10] # 允许数值型作为分类变量

    # 用户选择分析目标
    analysis_mode = st.sidebar.selectbox("分析目标", ["数值变量差异比较 (T检验/ANOVA/非参数)", "分类变量关联分析 (卡方/Fisher)"])

    if analysis_mode == "数值变量差异比较 (T检验/ANOVA/非参数)":
        target_col = st.sidebar.selectbox("因变量 (Y, 数值型)", num_cols)
        group_col = st.sidebar.selectbox("分组变量 (X, 分类型)", [c for c in cols if c != target_col])
        
        if st.sidebar.button("开始分析") or True: # 自动运行或按钮触发
            st.divider()
            groups = df[group_col].dropna().unique()
            n_groups = len(groups)
            
            st.write(f"**分析模型**: `{target_col}` by `{group_col}`")
            st.write(f"**分组数量**: {n_groups} 组 ({', '.join(map(str, groups))})")

            if n_groups < 2:
                st.error("错误：分组变量至少需要包含 2 个组别。")
                st.stop()

            # 数据分组提取
            group_data = [df[df[group_col] == g][target_col].dropna() for g in groups]

            # --- 1. 正态性与方差齐性 ---
            col1, col2 = st.columns(2)
            all_normal = True
            with col1:
                st.write("#### 正态性检验 (Shapiro-Wilk)")
                for i, g in enumerate(groups):
                    s, p = stats.shapiro(group_data[i])
                    is_norm = p > 0.05
                    if not is_norm: all_normal = False
                    st.write(f"- 组 {g}: P={p:.4f} {'(正态)' if is_norm else '(非正态)'}")
            
            with col2:
                st.write("#### 方差齐性检验 (Levene)")
                s_lev, p_lev = stats.levene(*group_data)
                is_homo = p_lev > 0.05
                st.write(f"- 整体: P={p_lev:.4f} {'(方差齐)' if is_homo else '(方差不齐)'}")

            # --- 2. 图表诊断 ---
            with st.expander("查看诊断图表 (QQ图/残差图)", expanded=False):
                tabs = st.tabs(["QQ图", "箱线图"])
                with tabs[0]:
                    fig, ax = plt.subplots(1, n_groups, figsize=(4*n_groups, 4))
                    if n_groups == 1: ax = [ax]
                    for i, g in enumerate(groups):
                        stats.probplot(group_data[i], dist="norm", plot=ax[i])
                        ax[i].set_title(f"QQ Plot: {g}")
                    st.pyplot(fig)
                with tabs[1]:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=group_col, y=target_col, data=df, ax=ax)
                    sns.stripplot(x=group_col, y=target_col, data=df, color='black', alpha=0.3, ax=ax)
                    st.pyplot(fig)

            # --- 3. 统计方法推荐与执行 ---
            st.subheader("分析结果")
            
            method_name = ""
            p_value = 1.0
            result_text = ""
            
            # 决策树逻辑
            if n_groups == 2:
                if all_normal and is_homo:
                    method_name = "独立样本 T 检验 (Student's t-test)"
                    res = stats.ttest_ind(group_data[0], group_data[1])
                    p_value = res.pvalue
                elif all_normal and not is_homo:
                    method_name = "Welch's T 检验 (不需方差齐)"
                    res = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                    p_value = res.pvalue
                else:
                    method_name = "Mann-Whitney U 检验 (非参数)"
                    res = stats.mannwhitneyu(group_data[0], group_data[1])
                    p_value = res.pvalue
            else: # > 2 groups
                if all_normal and is_homo:
                    method_name = "单因素方差分析 (One-Way ANOVA)"
                    res = stats.f_oneway(*group_data)
                    p_value = res.pvalue
                elif all_normal and not is_homo:
                    # 简易处理：推荐Welch ANOVA，此处暂用Kruskal或提示
                    method_name = "Kruskal-Wallis 检验 (因方差不齐，推荐非参数)"
                    res = stats.kruskal(*group_data)
                    p_value = res.pvalue
                else:
                    method_name = "Kruskal-Wallis H 检验 (非参数)"
                    res = stats.kruskal(*group_data)
                    p_value = res.pvalue

            st.info(f"💡 推荐并使用的统计方法：**{method_name}**")
            st.metric("P 值 (P-value)", f"{p_value:.4e}" if p_value < 0.001 else f"{p_value:.4f}")

            # --- 4. Post-hoc 分析 (仅当显著时) ---
            if p_value < 0.05:
                st.write("---")
                st.write("#### 事后多重比较 (Post-hoc Analysis)")
                st.caption("检测到组间存在显著差异，正在进行两两比较...")

                if "ANOVA" in method_name:
                    # Tukey HSD
                    tukey = pairwise_tukeyhsd(endog=df[target_col], groups=df[group_col], alpha=0.05)
                    st.text(tukey.summary())
                    # 转换结论为人话
                    sig_pairs = tukey.summary().data[1:]
                    sig_found = [row for row in sig_pairs if row[6] == True] # reject column
                    if sig_found:
                        st.write("**显著差异组对:**")
                        for row in sig_found:
                            st.write(f"- **{row[0]}** vs **{row[1]}** (P={row[3]:.4f})")
                
                elif "Kruskal" in method_name or "Mann-Whitney" in method_name:
                    # 简化版 Post-hoc：Bonferroni校正的Mann-Whitney
                    # scikit-posthocs 库更好，但为了保持单文件运行稳定，这里手写一个简单的校正
                    st.write("**使用 Bonferroni 校正的 Mann-Whitney U 检验:**")
                    import itertools
                    pairs = list(itertools.combinations(groups, 2))
                    adj_alpha = 0.05 / len(pairs)
                    st.write(f"校正后显著性阈值 alpha = {adj_alpha:.5f}")
                    
                    for g1, g2 in pairs:
                        d1 = df[df[group_col] == g1][target_col]
                        d2 = df[df[group_col] == g2][target_col]
                        u_stat, p_u = stats.mannwhitneyu(d1, d2)
                        sig = "**显著**" if p_u < adj_alpha else "不显著"
                        st.write(f"- {g1} vs {g2}: P={p_u:.4f} -> {sig}")

    elif analysis_mode == "分类变量关联分析 (卡方/Fisher)":
        var1 = st.sidebar.selectbox("行变量 (Row)", cols)
        var2 = st.sidebar.selectbox("列变量 (Column)", [c for c in cols if c != var1])
        
        st.divider()
        st.write(f"**列联表分析**: `{var1}` vs `{var2}`")
        
        # 生成列联表
        crosstab = pd.crosstab(df[var1], df[var2])
        st.write("#### 观测频数表 (Observed)")
        st.dataframe(crosstab)
        
        # 计算期望频数
        chi2, p, dof, expected = stats.chi2_contingency(crosstab)
        
        # 判断方法
        total_sample = crosstab.values.sum()
        min_expected = expected.min()
        shape = crosstab.shape
        
        method_name = ""
        
        # 逻辑判定
        if shape == (2, 2):
            if min_expected < 5 or total_sample < 40:
                method_name = "Fisher 精确检验 (Fisher's Exact Test)"
                # Fisher只支持2x2
                oddsratio, p_val = stats.fisher_exact(crosstab)
            else:
                method_name = "卡方检验 (Pearson Chi-Square)"
                p_val = p # 使用上面 chi2_contingency 的结果
        else:
            # R x C 表格
            if min_expected < 5:
                st.warning("警告：超过20%的单元格期望频数小于5，卡方检验结果可能不准。")
            method_name = "卡方检验 (Pearson Chi-Square)"
            p_val = p

        st.info(f"💡 推荐并使用的统计方法：**{method_name}**")
        st.metric("P 值 (P-value)", f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.success("结论：两个变量之间存在显著关联。")
        else:
            st.write("结论：两个变量之间相互独立（无显著关联）。")

else:
    st.info("👈 请在左侧上传文件或粘贴数据以开始。")
