import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 页面配置 ---
st.set_page_config(page_title="Ives统计分析工具", layout="wide")
st.title("Ives统计分析工具")

# --- 1. 定义数据模板 ---
# 这些是展示给用户的“栗子”，用户可以直接在上面改，或者覆盖粘贴
TEMPLATES = {
    "两组比较 (T检验/非参数)": {
        "desc": "宽格式：每一列是一组。适用：实验组 vs 对照组。",
        "data": pd.DataFrame({
            "Control": [10.2, 11.5, 10.8, 12.1, 11.3, 10.9],
            "Treatment": [13.5, 14.2, 15.1, 14.8, 13.9, 15.5]
        })
    },
    "多组比较 (ANOVA/非参数)": {
        "desc": "宽格式：每一列是一组。适用：3组及以上比较。",
        "data": pd.DataFrame({
            "Group_A": [5.1, 5.5, 5.2, 5.8, 5.4],
            "Group_B": [6.2, 6.1, 6.5, 6.3, 6.4],
            "Group_C": [4.5, 4.8, 4.2, 4.6, 4.9],
            "Group_D": [8.1, 8.5, 8.2, 8.6, 8.4]
        })
    },
    "双因素分析 (Two-Way ANOVA)": {
        "desc": "长格式：标准3列。列1=因素A，列2=因素B，列3=数值。",
        "data": pd.DataFrame({
            "Genotype": ["WT"]*4 + ["Mutant"]*4,
            "Drug": ["Vehicle", "Vehicle", "Treated", "Treated"] * 2,
            "Value": [10, 12, 25, 28, 8, 9, 15, 14]
        })
    },
    "列联表 (卡方/Fisher)": {
        "desc": "统计表格式：第一列是行名，后面是数值计数。",
        "data": pd.DataFrame({
            "Outcome": ["Cured", "Not Cured"],
            "Placebo": [15, 35],
            "Drug_A": [30, 20]
        })
    }
}

# --- 2. 侧边栏：选择模板 ---
st.sidebar.header("1. 分析类型选择")
selected_template = st.sidebar.radio(
    "选择你的数据类型", 
    list(TEMPLATES.keys())
)

st.sidebar.info(f"💡 **格式说明**：\n{TEMPLATES[selected_template]['desc']}")

# --- 3. 主界面：可编辑表格 ---
st.subheader("2. 数据录入 (支持从Excel直接复制粘贴)")
st.caption("👇 点击表格左上角可全选删除，然后粘贴你的数据 (Ctrl+V)。")

# 初始化 session state 用于存储数据，防止刷新重置
if 'current_df' not in st.session_state or st.session_state.get('last_template') != selected_template:
    st.session_state.current_df = TEMPLATES[selected_template]['data']
    st.session_state.last_template = selected_template

# 核心组件：可编辑表格
# num_rows="dynamic" 允许用户添加/删除行
edited_df = st.data_editor(
    st.session_state.current_df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"editor_{selected_template}" # 关键：切换模板时强制重绘表格
)

# --- 4. 自动化分析逻辑 ---
if edited_df is not None and not edited_df.empty:
    st.divider()
    st.subheader("3. 分析报告")

    # === 分流处理逻辑 ===
    
    # [场景 A] 宽格式比较 (两组 或 多组)
    if "两组" in selected_template or "多组" in selected_template:
        # 1. 数据清洗：宽格式转长格式 (Melt) 以便处理不同长度的数据
        # 在 data_editor 中，空单元格可能是 None 或 NaN
        cols = edited_df.columns.tolist()
        clean_data = {}
        for c in cols:
            # 提取非空数值
            valid_vals = pd.to_numeric(edited_df[c], errors='coerce').dropna().values
            if len(valid_vals) > 0:
                clean_data[c] = valid_vals
        
        groups = list(clean_data.keys())
        if len(groups) < 2:
            st.warning("⚠️ 请至少输入两列有效数据。")
            st.stop()
            
        group_vals = [clean_data[g] for g in groups]
        
        # 2. 检验正态性与方差
        col1, col2 = st.columns(2)
        all_normal = True
        with col1:
            st.write("**正态性检验 (Shapiro)**")
            for g, vals in clean_data.items():
                if len(vals) >= 3:
                    s, p = stats.shapiro(vals)
                    is_norm = p > 0.05
                    if not is_norm: all_normal = False
                    st.write(f"- {g}: P={p:.3f} {'✅' if is_norm else '❌'}")
                else:
                    st.write(f"- {g}: 样本<3，跳过")
        
        with col2:
            st.write("**方差齐性 (Levene)**")
            s_lev, p_lev = stats.levene(*group_vals)
            is_homo = p_lev > 0.05
            st.write(f"- 整体: P={p_lev:.3f} {'✅' if is_homo else '❌'}")
            
        # 3. 推荐与计算
        method_name = ""
        p_val = 1.0
        
        if len(groups) == 2:
            # T-test 家族
            if all_normal and is_homo:
                method_name = "独立样本 T 检验"
                res = stats.ttest_ind(group_vals[0], group_vals[1])
                p_val = res.pvalue
            elif all_normal and not is_homo:
                method_name = "Welch's T 检验"
                res = stats.ttest_ind(group_vals[0], group_vals[1], equal_var=False)
                p_val = res.pvalue
            else:
                method_name = "Mann-Whitney U 检验"
                res = stats.mannwhitneyu(group_vals[0], group_vals[1])
                p_val = res.pvalue
        else:
            # ANOVA 家族
            if all_normal and is_homo:
                method_name = "单因素方差分析 (ANOVA)"
                res = stats.f_oneway(*group_vals)
                p_val = res.pvalue
            else:
                method_name = "Kruskal-Wallis H 检验"
                res = stats.kruskal(*group_vals)
                p_val = res.pvalue
        
        st.info(f"💡 推荐方法：**{method_name}**")
        st.metric("P-value", f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}")
        
        # 4. 可视化
        with st.expander("📊 查看图表 (箱线图/QQ图)", expanded=True):
            # 为了画图方便，构建一个临时的长格式 DF
            plot_data = []
            for g, vals in clean_data.items():
                for v in vals:
                    plot_data.append({"Group": g, "Value": v})
            df_plot = pd.DataFrame(plot_data)
            
            t1, t2 = st.tabs(["箱线图", "QQ图"])
            with t1:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.boxplot(data=df_plot, x="Group", y="Value", ax=ax, palette="Set2")
                sns.stripplot(data=df_plot, x="Group", y="Value", color='black', alpha=0.5, ax=ax)
                st.pyplot(fig)
            with t2:
                fig, axes = plt.subplots(1, len(groups), figsize=(4*len(groups), 4))
                if len(groups)==1: axes=[axes]
                for i, g in enumerate(groups):
                    stats.probplot(clean_data[g], dist="norm", plot=axes[i])
                    axes[i].set_title(g)
                st.pyplot(fig)

        # 5. Post-hoc
        if p_val < 0.05 and len(groups) > 2:
            st.write("---")
            st.write("**事后多重比较 (Post-hoc)**")
            if "ANOVA" in method_name:
                tukey = pairwise_tukeyhsd(endog=df_plot['Value'], groups=df_plot['Group'])
                st.text(tukey.summary())
            else:
                st.caption("Mann-Whitney U with Bonferroni correction")
                import itertools
                pairs = list(itertools.combinations(groups, 2))
                adj_alpha = 0.05 / len(pairs)
                for g1, g2 in pairs:
                    u, p_pair = stats.mannwhitneyu(clean_data[g1], clean_data[g2])
                    sig = "🔴显著" if p_pair < adj_alpha else "⚪"
                    st.write(f"{g1} vs {g2}: P={p_pair:.4f} {sig}")

    # [场景 B] 双因素分析
    elif "双因素" in selected_template:
        # 检查列数
        if edited_df.shape[1] < 3:
            st.error("双因素分析需要至少3列数据 (因素1, 因素2, 数值)")
            st.stop()
            
        # 默认取前3列
        cols = edited_df.columns
        f1, f2, val = cols[0], cols[1], cols[2]
        st.write(f"**识别变量**: 因素1=`{f1}`, 因素2=`{f2}`, 数值=`{val}`")
        
        # 清洗：确保数值列是数字
        df_clean = edited_df.copy()
        df_clean[val] = pd.to_numeric(df_clean[val], errors='coerce')
        df_clean = df_clean.dropna()
        
        # 必须用 rename 规避列名含空格的问题
        df_model = df_clean.rename(columns={f1: 'F1', f2: 'F2', val: 'Y'})
        
        model = ols('Y ~ C(F1) + C(F2) + C(F1):C(F2)', data=df_model).fit()
        anova_tab = sm.stats.anova_lm(model, typ=2)
        
        # 结果展示
        st.info("💡 方法：**双因素方差分析 (Two-Way ANOVA)**")
        
        # 格式化表格
        display_tab = anova_tab.rename(index={'C(F1)': f1, 'C(F2)': f2, 'C(F1):C(F2)': '交互作用'})
        st.dataframe(display_tab.style.format("{:.4f}"))
        
        # 交互作用图
        with st.expander("📊 交互作用图", expanded=True):
            fig, ax = plt.subplots()
            from statsmodels.graphics.factorplots import interaction_plot
            interaction_plot(x=df_model['F1'], trace=df_model['F2'], response=df_model['Y'], ax=ax)
            st.pyplot(fig)

    # [场景 C] 列联表 (卡方)
    elif "列联表" in selected_template:
        # 假设第一列是 Row Names，后面是数据列
        row_names = edited_df.iloc[:, 0].astype(str).values
        data_cols = edited_df.columns[1:]
        
        try:
            # 提取纯数值矩阵
            observed = edited_df[data_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            st.write("**观测频数表**")
            st.dataframe(pd.DataFrame(observed, index=row_names, columns=data_cols))
            
            chi2, p, dof, expected = stats.chi2_contingency(observed)
            
            method_name = "Pearson Chi-Square"
            if observed.sum() < 40 or expected.min() < 5:
                if observed.shape == (2,2):
                    method_name = "Fisher's Exact Test"
                    odds, p = stats.fisher_exact(observed)
                else:
                    st.warning("⚠️ 样本量较小，卡方结果可能不准。")
            
            st.info(f"💡 推荐方法：**{method_name}**")
            st.metric("P-value", f"{p:.4e}" if p < 0.001 else f"{p:.4f}")
            
        except Exception as e:
            st.error(f"数据格式错误，请确保除第一列外均为数字。错误信息: {e}")
