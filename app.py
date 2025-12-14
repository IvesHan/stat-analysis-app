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
st.set_page_config(page_title="全能统计分析工具", layout="wide")
st.title("📊 全能统计分析工具")
st.markdown("集成 **单因素差异分析**、**双因素方差分析 (矩阵输入)** 与 **列联表分析**。")

# --- session_state 初始化 ---
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "单因素差异分析 (T检验/ANOVA/非参数)"

# --- 1. 侧边栏：功能选择 ---
st.sidebar.header("1. 分析模式选择")
analysis_mode = st.sidebar.radio(
    "请选择分析类型",
    [
        "单因素差异分析 (T检验/ANOVA/非参数)", 
        "双因素方差分析 (Two-Way ANOVA)", 
        "列联表分析 (卡方/Fisher)"
    ]
)
st.session_state.current_mode = analysis_mode

# ==============================================================================
# 模块 A: 单因素差异分析 (T检验, ANOVA, Mann-Whitney, Kruskal-Wallis)
# ==============================================================================
if analysis_mode == "单因素差异分析 (T检验/ANOVA/非参数)":
    st.header("🅰️ 单因素差异分析")
    st.caption("适用：两组或多组数据的均值/分布比较。每一列代表一个组。")

    # --- 数据初始化与加列逻辑 ---
    if 'oneway_df' not in st.session_state:
        st.session_state.oneway_df = pd.DataFrame({
            "Control": [10.2, 11.5, 10.8, 12.1, 11.3, 10.9],
            "Treatment": [13.5, 14.2, 15.1, 14.8, 13.9, 15.5]
        })

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        # ✅ 单因素加列按钮
        if st.button("➕ 增加一组 (列)", key="btn_add_col_oneway"):
            current_cols = len(st.session_state.oneway_df.columns)
            new_col_name = f"Group_{current_cols + 1}"
            st.session_state.oneway_df[new_col_name] = None
            st.rerun()

    with col_info:
        st.info("提示：点击左侧按钮添加新组。每一列是一组数据。")

    # 单因素数据编辑器
    df_input = st.data_editor(
        st.session_state.oneway_df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_oneway"
    )
    # 同步数据状态
    st.session_state.oneway_df = df_input

    if df_input is not None and not df_input.empty:
        # 数据清洗：宽格式转列表
        cols = df_input.columns.tolist()
        clean_data = {}
        for c in cols:
            # 提取非空数值
            valid_vals = pd.to_numeric(df_input[c], errors='coerce').dropna().values
            if len(valid_vals) > 0:
                clean_data[c] = valid_vals
        
        groups = list(clean_data.keys())
        
        if len(groups) < 2:
            st.warning("⚠️ 请至少输入两列有效数据以进行比较。")
        else:
            if st.button("开始分析 (单因素)", type="primary"):
                st.divider()
                group_vals = [clean_data[g] for g in groups]

                # 1. 假设检验
                col1, col2 = st.columns(2)
                all_normal = True
                with col1:
                    st.subheader("1. 正态性检验 (Shapiro)")
                    for g, vals in clean_data.items():
                        if len(vals) >= 3:
                            s, p = stats.shapiro(vals)
                            is_norm = p > 0.05
                            if not is_norm: all_normal = False
                            st.write(f"- **{g}**: P={p:.4f} {'✅' if is_norm else '❌'}")
                        else:
                            st.write(f"- **{g}**: 样本太少，跳过")
                
                with col2:
                    st.subheader("2. 方差齐性检验 (Levene)")
                    if len(group_vals) >= 2:
                        s_lev, p_lev = stats.levene(*group_vals)
                        is_homo = p_lev > 0.05
                        st.write(f"- **整体**: P={p_lev:.4f} {'✅' if is_homo else '❌'}")
                    else:
                        is_homo = False

                # 2. 方法推荐与计算
                st.subheader("3. 统计结果")
                method_name = ""
                p_val = 1.0
                
                if len(groups) == 2:
                    if all_normal and is_homo:
                        method_name = "独立样本 T 检验"
                        res = stats.ttest_ind(group_vals[0], group_vals[1])
                        p_val = res.pvalue
                    elif all_normal and not is_homo:
                        method_name = "Welch's T 检验 (校正方差不齐)"
                        res = stats.ttest_ind(group_vals[0], group_vals[1], equal_var=False)
                        p_val = res.pvalue
                    else:
                        method_name = "Mann-Whitney U 检验 (非参数)"
                        res = stats.mannwhitneyu(group_vals[0], group_vals[1])
                        p_val = res.pvalue
                else: # > 2 groups
                    if all_normal and is_homo:
                        method_name = "单因素方差分析 (One-Way ANOVA)"
                        res = stats.f_oneway(*group_vals)
                        p_val = res.pvalue
                    else:
                        method_name = "Kruskal-Wallis 检验 (非参数)"
                        res = stats.kruskal(*group_vals)
                        p_val = res.pvalue
                
                st.success(f"💡 推荐并执行：**{method_name}**")
                st.metric("P-value", f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}")

                # 3. Post-hoc
                if p_val < 0.05 and len(groups) > 2:
                    st.markdown("---")
                    st.subheader("4. 事后多重比较 (Post-hoc)")
                    # 构造长数据用于Post-hoc
                    ph_data = []
                    for g, vals in clean_data.items():
                        for v in vals: ph_data.append({"Group": g, "Value": v})
                    df_ph = pd.DataFrame(ph_data)

                    if "ANOVA" in method_name:
                        tukey = pairwise_tukeyhsd(endog=df_ph['Value'], groups=df_ph['Group'])
                        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                        sig_tukey = tukey_df[tukey_df['reject'] == True]
                        if not sig_tukey.empty:
                            st.write("**显著差异组对：**")
                            st.dataframe(sig_tukey)
                        else:
                            st.write("ANOVA显著，但Tukey两两比较未发现显著差异。")
                    else:
                        st.write("**Bonferroni 校正的 Mann-Whitney U 检验:**")
                        import itertools
                        pairs = list(itertools.combinations(groups, 2))
                        adj = 0.05 / len(pairs)
                        st.caption(f"校正后 Alpha = {adj:.5f}")
                        found_sig = False
                        for g1, g2 in pairs:
                            u, p_pair = stats.mannwhitneyu(clean_data[g1], clean_data[g2])
                            if p_pair < adj:
                                st.write(f"🔴 **{g1} vs {g2}**: P={p_pair:.4f} (显著)")
                                found_sig = True
                        if not found_sig:
                            st.write("未发现显著差异。")

                # 4. 作图
                with st.expander("查看箱线图", expanded=True):
                    plot_data = []
                    for g, vals in clean_data.items():
                        for v in vals: plot_data.append({"Group": g, "Value": v})
                    df_plot = pd.DataFrame(plot_data)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(data=df_plot, x="Group", y="Value", ax=ax, palette="Set2")
                    sns.stripplot(data=df_plot, x="Group", y="Value", color='black', alpha=0.5, ax=ax)
                    st.pyplot(fig)


# ==============================================================================
# 模块 B: 双因素方差分析 (Two-Way ANOVA) - 矩阵输入版
# ==============================================================================
elif analysis_mode == "双因素方差分析 (Two-Way ANOVA)":
    st.header("🅱️ 双因素方差分析 (矩阵输入模式)")
    st.caption("适用：分析两个因素（如：性别 × 治疗）及其交互作用。")

    # --- 1. 数据准备区 ---
    if 'twoway_df' not in st.session_state:
        st.session_state.twoway_df = pd.DataFrame([
            ["Light smoker", 24.1, 29.2, 24.6, 20.0, 21.9, 17.6],
            ["Heavy smoker", 17.6, 18.8, 23.2, 14.8, 10.3, 11.3]
        ], columns=["Condition", "A1", "A2", "A3", "B1", "B2", "B3"])

    col_tools1, col_tools2 = st.columns([1, 4])
    with col_tools1:
        # ✅ 双因素加列按钮
        if st.button("➕ 增加一列数据", key="btn_add_col_twoway"):
            current_cols = len(st.session_state.twoway_df.columns)
            new_col_name = f"NewCol_{current_cols}"
            st.session_state.twoway_df[new_col_name] = None
            st.rerun()
            
    with col_tools2:
        st.info("提示：第1列输入行因素（如吸烟），后面输入数值列。点击左侧按钮增加列。")

    edited_df = st.data_editor(
        st.session_state.twoway_df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor_twoway"
    )
    st.session_state.twoway_df = edited_df

    # --- 2. 列映射设置 ---
    st.subheader("2. 变量与分组定义")
    all_cols = edited_df.columns.tolist()
    
    if len(all_cols) < 2:
        st.error("数据列太少！")
        st.stop()

    first_col = all_cols[0]
    data_cols = all_cols[1:]

    c1, c2 = st.columns(2)
    with c1:
        factor_a_name = st.text_input("行因素名称 (Factor A)", value=first_col)
    with c2:
        factor_b_name = st.text_input("列因素名称 (Factor B)", value="Gender")

    st.markdown("##### 分配数据列到 Factor B 的不同水平")
    col_grp1, col_grp2 = st.columns(2)
    with col_grp1:
        group1_name = st.text_input("水平 1 名称 (如 Male)", value="Level_1")
        default_g1 = data_cols[:len(data_cols)//2]
        group1_cols = st.multiselect(f"属于 {group1_name} 的列", data_cols, default=default_g1)
    
    with col_grp2:
        group2_name = st.text_input("水平 2 名称 (如 Female)", value="Level_2")
        default_g2 = [c for c in data_cols if c not in default_g1]
        group2_cols = st.multiselect(f"属于 {group2_name} 的列", data_cols, default=default_g2)

    # --- 3. 执行分析 ---
    if st.button("开始分析 (双因素)", type="primary"):
        st.divider()
        if not group1_cols or not group2_cols:
            st.error("请确保每个分组至少分配了一列数据。")
            st.stop()
            
        long_data = []
        try:
            for idx, row in edited_df.iterrows():
                row_label = row[first_col]
                for c in group1_cols:
                    val = pd.to_numeric(row[c], errors='coerce')
                    if not pd.isna(val):
                        long_data.append({factor_a_name: str(row_label), factor_b_name: group1_name, "Value": val})
                for c in group2_cols:
                    val = pd.to_numeric(row[c], errors='coerce')
                    if not pd.isna(val):
                        long_data.append({factor_a_name: str(row_label), factor_b_name: group2_name, "Value": val})
            
            df_long = pd.DataFrame(long_data)
            
            st.subheader("3. 方差分析表 (ANOVA)")
            df_model = df_long.rename(columns={factor_a_name: 'FA', factor_b_name: 'FB', 'Value': 'Y'})
            model = ols('Y ~ C(FA) + C(FB) + C(FA):C(FB)', data=df_model).fit()
            anova_tab = sm.stats.anova_lm(model, typ=2)
            
            display_tab = anova_tab.rename(index={'C(FA)': f'主效应: {factor_a_name}', 'C(FB)': f'主效应: {factor_b_name}', 'C(FA):C(FB)': '交互作用'})
            
            def highlight_sig(s):
                return ['background-color: #d1e7dd' if v < 0.05 else '' for v in s]
            
            st.dataframe(display_tab.style.format("{:.4f}").apply(highlight_sig, subset=['PR(>F)']))

            p_int = anova_tab.loc['C(FA):C(FB)', 'PR(>F)']
            if p_int < 0.05:
                st.warning(f"🔴 检测到显著交互作用 (P={p_int:.4f})")
            else:
                st.success(f"🟢 未检测到交互作用 (P={p_int:.4f})")

            st.subheader("4. 事后多重比较 (Tukey HSD)")
            df_long['Combo'] = df_long[factor_a_name].astype(str) + " + " + df_long[factor_b_name].astype(str)
            tukey = pairwise_tukeyhsd(endog=df_long['Value'], groups=df_long['Combo'], alpha=0.05)
            
            res_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            sig_df = res_df[res_df['reject'] == True]
            
            if not sig_df.empty:
                st.write("**显著差异组对：**")
                st.dataframe(sig_df[['group1', 'group2', 'p-adj', 'meandiff']].style.format({'p-adj': '{:.4f}'}))
            else:
                st.info("未发现显著的两两差异。")

            st.subheader("5. 交互作用图")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.pointplot(data=df_long, x=factor_a_name, y="Value", hue=factor_b_name, markers=['o', 's'], capsize=0.1, ax=ax)
            ax.set_title("Interaction Plot")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"分析出错，请检查输入数据是否包含非数字字符。错误: {e}")


# ==============================================================================
# 模块 C: 列联表分析 (卡方/Fisher)
# ==============================================================================
elif analysis_mode == "列联表分析 (卡方/Fisher)":
    st.header("©️ 列联表分析")
    st.caption("适用：分析两个分类变量的关联性（例如：治愈率 vs 治疗组别）。")

    # 默认示例数据
    default_chi = pd.DataFrame({
        "Outcome": ["Cured", "Not Cured"],
        "Group_A": [30, 20],
        "Group_B": [15, 35]
    })
    
    st.info("👇 请输入频数数据。第一列为结果分类（Row），后续列为各组计数（Column）。")
    
    # 数据编辑器
    df_chi = st.data_editor(default_chi, num_rows="dynamic", use_container_width=True)

    if st.button("开始分析 (卡方)"):
        st.divider()
        try:
            # 1. 提取数值矩阵
            data_cols = df_chi.columns[1:]
            # 强制转为数值，无法转换的变为NaN
            observed = df_chi[data_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            st.write("**观测频数表 (Observed)：**")
            st.dataframe(df_chi)

            # 2. 先做卡方，获取期望频数以判断是否符合条件
            chi2, p, dof, ex = stats.chi2_contingency(observed)
            
            total_n = observed.sum()
            min_ex = ex.min() # 最小期望频数
            
            method = "Pearson卡方检验"
            
            # 3. 智能判断：是否需要 Fisher 精确检验
            # 条件：表格为 2x2 且 (总样本<40 或 有期望频数<5)
            if observed.shape == (2,2) and (total_n < 40 or min_ex < 5):
                method = "Fisher精确检验 (Fisher's Exact Test)"
                odds, p = stats.fisher_exact(observed)
            elif min_ex < 5:
                st.warning("⚠️ 注意：有单元格期望频数小于5，但表格不是2x2，Fisher检验不适用。卡方结果可能不准。")
            
            # 4. 结果输出
            st.success(f"💡 推荐并使用：**{method}**")
            st.metric("P-value", f"{p:.4e}" if p < 0.001 else f"{p:.4f}")
            
            st.markdown("---")
            if p < 0.05:
                st.write("👉 **结论**：拒绝零假设，两个变量之间 **存在显著关联**。")
            else:
                st.write("👉 **结论**：接受零假设，两个变量之间 **相互独立 (无显著关联)**。")

        except Exception as e:
            st.error(f"分析出错：请确保除第一列外，其他列均为纯数字。\n错误信息: {e}")
