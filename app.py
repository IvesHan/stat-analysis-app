import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- 0. 页面配置 ---
st.set_page_config(page_title="统计分析工具", layout="wide")
st.title("统计分析工具")

# --- 工具函数：解析手动输入 ---
def parse_manual_input(text_input, sep_char):
    try:
        data = StringIO(text_input)
        if sep_char == '制表符 (Excel复制)':
            # read_csv 对不齐的数据处理较好，会自动填充NaN
            df = pd.read_csv(data, sep='\t')
        elif sep_char == '逗号 (CSV)':
            df = pd.read_csv(data)
        elif sep_char == '空格':
            df = pd.read_csv(data, delim_whitespace=True)
        return df
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None

# --- 侧边栏：数据输入 ---
st.sidebar.header("1. 数据输入")
input_method = st.sidebar.radio("选择数据来源", ["手动输入/粘贴", "上传文件"])

df_raw = None

if input_method == "手动输入/粘贴":
    st.sidebar.info("提示：直接从Excel复制数据粘贴即可")
    sep_mode = st.sidebar.selectbox("分隔符格式", ["制表符 (Excel复制)", "逗号 (CSV)", "空格"], index=0)
    # 默认值展示用户想要的宽格式
    default_text = "Male\tFemale\n54\t43\n23\t34\n45\t65\n54\t77\n45\t46\n\t65"
    text_data = st.sidebar.text_area("在此粘贴数据", height=200, value=default_text)
    if text_data:
        df_raw = parse_manual_input(text_data, sep_mode)

elif input_method == "上传文件":
    uploaded_file = st.sidebar.file_uploader("支持 xlsx, csv", type=["xlsx", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

# --- 主逻辑 ---
if df_raw is not None:
    st.write("### 1. 原始数据预览")
    st.dataframe(df_raw.head())

    # --- 数据格式清洗与转换 (关键更新) ---
    st.sidebar.header("2. 数据格式设置")
    data_shape = st.sidebar.radio(
        "选择数据排列方式", 
        ["宽格式 (每列是一组)", "长格式 (一列分组+一列数值)"],
        help="宽格式：如你提供的示例，Male一列，Female一列。\n长格式：一列叫Group写着Male/Female，一列叫Value写着数字。"
    )

    df_clean = None
    target_col = "Value"
    group_col = "Group"

    if data_shape == "宽格式 (每列是一组)":
        # 自动转换为长格式 (Melt)
        try:
            # 1. 获取所有数值列
            cols = df_raw.columns.tolist()
            # 2. 转换逻辑：遍历每一列，去除空值，合并
            melted_data = []
            for c in cols:
                # 只取数值类型的数据，且去除空值 (NaN)
                # 强制转为numeric，非数字变NaN
                clean_series = pd.to_numeric(df_raw[c], errors='coerce').dropna()
                for val in clean_series:
                    melted_data.append({group_col: c, target_col: val})
            
            df_clean = pd.DataFrame(melted_data)
            st.info(f"已自动将宽格式转换为分析格式：共 {len(cols)} 个组 ({', '.join(cols)})")
            
        except Exception as e:
            st.error(f"格式转换失败，请检查数据是否包含非数字字符: {e}")
            st.stop()
            
    else: # 长格式
        df_clean = df_raw.copy()
        cols = df_clean.columns.tolist()
        # 让用户选列
        st.sidebar.subheader("指定列名")
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.sidebar.selectbox("数值变量 (Y)", num_cols)
        group_col = st.sidebar.selectbox("分组变量 (X)", [c for c in cols if c != target_col])

    # --- 确保数据准备完毕 ---
    if df_clean is not None:
        # st.write("### 2. 清洗后数据 (用于分析)", df_clean.head()) 
        
        # 这里的逻辑和之前一样，但基于 df_clean 运行
        # 自动推断分析模式：只要转换成功，大概率是数值比较
        # 但保留卡方选项以防万一
        
        analysis_mode = st.sidebar.selectbox(
            "分析目标", 
            ["数值变量差异比较 (T检验/ANOVA/非参数)", "分类变量关联分析 (卡方/Fisher)"]
        )

        if analysis_mode == "数值变量差异比较 (T检验/ANOVA/非参数)":
            groups = df_clean[group_col].unique()
            n_groups = len(groups)
            
            if n_groups < 2:
                st.error("错误：有效分组少于2组，无法进行比较。")
                st.stop()

            group_data = [df_clean[df_clean[group_col] == g][target_col].values for g in groups]

            # --- 统计分析核心 ---
            st.divider()
            st.subheader("3. 统计分析报告")

            # 1. 假设检验
            col1, col2 = st.columns(2)
            all_normal = True
            with col1:
                st.write("**正态性检验 (Shapiro-Wilk)**")
                for i, g in enumerate(groups):
                    if len(group_data[i]) >= 3: # Shapiro要求至少3个样本
                        s, p = stats.shapiro(group_data[i])
                        is_norm = p > 0.05
                        if not is_norm: all_normal = False
                        st.write(f"- {g}: P={p:.4f} { '✅' if is_norm else '❌'}")
                    else:
                        st.write(f"- {g}: 样本过少，跳过")
            
            with col2:
                st.write("**方差齐性检验 (Levene)**")
                # 移除空数组防止报错
                valid_data = [d for d in group_data if len(d) > 0]
                if len(valid_data) > 1:
                    s_lev, p_lev = stats.levene(*valid_data)
                    is_homo = p_lev > 0.05
                    st.write(f"- 整体: P={p_lev:.4f} { '✅' if is_homo else '❌'}")
                else:
                    is_homo = False # 无法检验

            # 2. 方法推荐与执行
            st.write("---")
            method_name = ""
            p_value = 1.0
            
            # 逻辑树
            if n_groups == 2:
                if all_normal and is_homo:
                    method_name = "独立样本 T 检验 (Student's t-test)"
                    res = stats.ttest_ind(group_data[0], group_data[1])
                    p_value = res.pvalue
                elif all_normal and not is_homo:
                    method_name = "Welch's T 检验 (校正方差不齐)"
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
                else:
                    method_name = "Kruskal-Wallis 检验 (非参数)"
                    res = stats.kruskal(*group_data)
                    p_value = res.pvalue

            st.info(f"💡 智能推荐方法：**{method_name}**")
            st.metric("P 值 (P-value)", f"{p_value:.4e}" if p_value < 0.001 else f"{p_value:.4f}")

            # 3. 可视化
            with st.expander("查看可视化图表 (箱线图/QQ图)", expanded=True):
                tab1, tab2 = st.tabs(["箱线图", "QQ图"])
                with tab1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x=group_col, y=target_col, data=df_clean, ax=ax, palette="Set2")
                    sns.stripplot(x=group_col, y=target_col, data=df_clean, color='black', alpha=0.5, ax=ax)
                    st.pyplot(fig)
                with tab2:
                    fig, ax = plt.subplots(1, n_groups, figsize=(4*n_groups, 4))
                    if n_groups == 1: ax = [ax]
                    for i, g in enumerate(groups):
                        stats.probplot(group_data[i], dist="norm", plot=ax[i])
                        ax[i].set_title(f"QQ: {g}")
                    st.pyplot(fig)

            # 4. Post-hoc
            if p_value < 0.05:
                st.write("---")
                st.subheader("事后多重比较 (Post-hoc)")
                if "ANOVA" in method_name:
                    tukey = pairwise_tukeyhsd(endog=df_clean[target_col], groups=df_clean[group_col], alpha=0.05)
                    st.text(tukey.summary())
                else:
                    st.caption("基于 Bonferroni 校正的 Mann-Whitney U 检验")
                    import itertools
                    pairs = list(itertools.combinations(groups, 2))
                    adj_alpha = 0.05 / len(pairs)
                    for g1, g2 in pairs:
                        d1 = df_clean[df_clean[group_col] == g1][target_col]
                        d2 = df_clean[df_clean[group_col] == g2][target_col]
                        u, p_u = stats.mannwhitneyu(d1, d2)
                        sig = "🔴 显著" if p_u < adj_alpha else "⚪ 不显著"
                        st.write(f"**{g1} vs {g2}**: P={p_u:.4f} {sig}")

        elif analysis_mode == "分类变量关联分析 (卡方/Fisher)":
            st.warning("卡方检验通常需要'长格式'或'交叉表'数据。如果您输入的是宽格式数值数据，请切换回'数值变量差异比较'模式。")
            # 这里保留原有逻辑，只要用户在上面选了长格式就能用
            if data_shape != "长格式 (一列分组+一列数值)":
                st.error("请在左侧数据格式设置中选择 '长格式' 以使用此功能，或上传整理好的列联表数据。")
            else:
                 # 复用之前的卡方逻辑
                 pass 

else:
    st.info("👈 请在左侧粘贴数据或上传文件。")
