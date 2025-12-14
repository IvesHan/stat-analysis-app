import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- 0. 页面配置 ---
st.set_page_config(page_title="统计分析工具", layout="wide")
st.title("统计分析工具")

# --- 工具函数 ---
def parse_manual_input(text_input, sep_char):
    try:
        data = StringIO(text_input)
        if sep_char == '制表符 (Excel复制)':
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
    
    # --- 增加：快速示例数据按钮 ---
    col_demo1, col_demo2 = st.sidebar.columns(2)
    if col_demo1.button("单因素示例"):
        st.session_state.demo_text = "GroupA\tGroupB\tGroupC\n12.5\t15.2\t18.1\n13.1\t14.8\t17.5\n11.9\t15.5\t18.6\n12.8\t14.2\t19.0\n13.0\t\t17.8"
        st.session_state.demo_type = "wide"
    
    if col_demo2.button("双因素示例"):
        # 生成标准的三列格式：性别、治疗、数值
        st.session_state.demo_text = "Gender\tDrug\tValue\nMale\tDrugA\t10.5\nMale\tDrugA\t11.2\nMale\tDrugB\t15.4\nMale\tDrugB\t16.1\nFemale\tDrugA\t12.1\nFemale\tDrugA\t13.0\nFemale\tDrugB\t18.2\nFemale\tDrugB\t17.5"
        st.session_state.demo_type = "long"

    # 获取文本框内容
    default_text = st.session_state.get('demo_text', "")
    sep_mode = st.sidebar.selectbox("分隔符格式", ["制表符 (Excel复制)", "逗号 (CSV)", "空格"], index=0)
    
    text_data = st.sidebar.text_area("在此粘贴数据 (建议带表头)", height=200, value=default_text)
    
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
    # 修复问题2：使用 use_container_width 展示完整数据，不使用 .head()
    st.write("### 1. 数据预览", df_raw) 
    st.caption(f"共 {df_raw.shape[0]} 行， {df_raw.shape[1]} 列")

    # --- 数据格式设置 ---
    st.sidebar.divider()
    st.sidebar.header("2. 数据结构")
    
    # 智能判断默认格式：如果列数=3且第一列像是分类，默认切到长格式
    default_fmt_idx = 0
    if st.session_state.get('demo_type') == 'long' or (df_raw.shape[1] == 3 and df_raw.iloc[:,0].dtype == 'O'):
        default_fmt_idx = 1
        
    data_shape = st.sidebar.radio(
        "选择数据排列方式", 
        ["宽格式 (每列是一组，仅限单因素)", "长格式 (标准格式，支持单/双因素)"],
        index=default_fmt_idx,
        help="宽格式：如 GroupA, GroupB 每列一组数据。\n长格式：一列分组(如Gender)，一列数值(Value)。双因素必须用长格式。"
    )

    df_clean = None
    target_col = "Value"
    group_cols = [] # 可能有多个分组变量

    # --- A. 宽格式处理 (自动 Melt) ---
    if "宽格式" in data_shape:
        try:
            cols = df_raw.columns.tolist()
            melted_data = []
            for c in cols:
                clean_series = pd.to_numeric(df_raw[c], errors='coerce').dropna()
                for val in clean_series:
                    melted_data.append({"Group": c, "Value": val})
            df_clean = pd.DataFrame(melted_data)
            group_cols = ["Group"]
            target_col = "Value"
        except Exception as e:
            st.error(f"宽格式转换失败: {e}")
            st.stop()

    # --- B. 长格式处理 (用户指定列) ---
    else:
        df_clean = df_raw.copy()
        all_cols = df_clean.columns.tolist()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # 自动推断
        default_num_idx = 0
        if len(num_cols) > 0:
             # 尝试找名字里带 value, score 的列
            for i, c in enumerate(num_cols):
                if 'val' in c.lower() or 'score' in c.lower():
                    default_num_idx = i
                    break
        
        st.sidebar.subheader("指定列名")
        target_col = st.sidebar.selectbox("数值变量 (Y)", num_cols, index=default_num_idx)
        
        # 剩余的列作为候选分组
        cat_candidates = [c for c in all_cols if c != target_col]
        selected_factors = st.sidebar.multiselect("选择分组变量 (X)", cat_candidates, default=cat_candidates[:2])
        
        if len(selected_factors) == 0:
            st.warning("请至少选择一个分组变量")
            st.stop()
        
        group_cols = selected_factors

    # --- 3. 分析模式选择 ---
    if df_clean is not None:
        st.sidebar.divider()
        st.sidebar.header("3. 分析与检验")
        
        # 自动判断模式
        analysis_mode = "单因素分析"
        if len(group_cols) == 2:
            analysis_mode = "双因素分析 (Two-Way)"
        elif len(group_cols) > 2:
            st.warning("暂不支持3个以上因素的交互分析，将仅进行描述统计。")
            st.stop()
            
        st.subheader(f"分析模式: {analysis_mode}")

        # === 单因素分析流程 ===
        if analysis_mode == "单因素分析":
            g_col = group_cols[0]
            groups = df_clean[g_col].unique()
            group_data = [df_clean[df_clean[g_col] == g][target_col].values for g in groups]
            
            # 1. 假设检验
            col1, col2 = st.columns(2)
            all_normal = True
            with col1:
                st.write("**正态性 (Shapiro)**")
                for i, g in enumerate(groups):
                    if len(group_data[i]) >= 3:
                        s, p = stats.shapiro(group_data[i])
                        is_norm = p > 0.05
                        if not is_norm: all_normal = False
                        st.write(f"- {g}: P={p:.4f} {'✅' if is_norm else '❌'}")
            
            with col2:
                st.write("**方差齐性 (Levene)**")
                valid_data = [d for d in group_data if len(d) > 0]
                if len(valid_data) > 1:
                    s_lev, p_lev = stats.levene(*valid_data)
                    is_homo = p_lev > 0.05
                    st.write(f"- 整体: P={p_lev:.4f} {'✅' if is_homo else '❌'}")
                else: is_homo = False

            # 2. 推荐逻辑
            method_name = ""
            p_value = 1.0
            
            if len(groups) == 2:
                if all_normal and is_homo:
                    method_name = "独立样本 T 检验"
                    res = stats.ttest_ind(group_data[0], group_data[1])
                    p_value = res.pvalue
                elif all_normal and not is_homo:
                    method_name = "Welch's T 检验"
                    res = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                    p_value = res.pvalue
                else:
                    method_name = "Mann-Whitney U 检验"
                    res = stats.mannwhitneyu(group_data[0], group_data[1])
                    p_value = res.pvalue
            else:
                if all_normal and is_homo:
                    method_name = "单因素方差分析 (One-Way ANOVA)"
                    res = stats.f_oneway(*group_data)
                    p_value = res.pvalue
                else:
                    method_name = "Kruskal-Wallis 检验"
                    res = stats.kruskal(*group_data)
                    p_value = res.pvalue
            
            st.info(f"💡 推荐方法：**{method_name}**")
            st.write(f"**P-value**: {p_value:.4f}")

            # 3. Post-hoc
            if p_value < 0.05:
                st.write("---")
                st.write("**事后多重比较**")
                if "ANOVA" in method_name:
                    tukey = pairwise_tukeyhsd(df_clean[target_col], df_clean[g_col])
                    st.text(tukey.summary())
                else:
                    st.write("非参数两两比较 (Bonferroni校正):")
                    import itertools
                    pairs = list(itertools.combinations(groups, 2))
                    adj = 0.05 / len(pairs)
                    for g1, g2 in pairs:
                        d1 = df_clean[df_clean[g_col]==g1][target_col]
                        d2 = df_clean[df_clean[g_col]==g2][target_col]
                        u, p_u = stats.mannwhitneyu(d1, d2)
                        sig = "🔴显著" if p_u < adj else "⚪"
                        st.write(f"{g1} vs {g2}: P={p_u:.4f} {sig}")
            
            # 4. 可视化
            with st.expander("图表"):
                fig, ax = plt.subplots()
                sns.boxplot(x=g_col, y=target_col, data=df_clean, ax=ax)
                st.pyplot(fig)

        # === 双因素分析流程 ===
        elif analysis_mode == "双因素分析 (Two-Way)":
            f1, f2 = group_cols[0], group_cols[1]
            st.write(f"**模型**: `{target_col} ~ {f1} + {f2} + {f1}:{f2}`")
            
            # 必须用 statsmodels 的 ols 字符串公式
            # 需要对列名进行清洗，防止空格报错
            df_temp = df_clean.rename(columns={target_col: 'Y', f1: 'F1', f2: 'F2'})
            
            model = ols('Y ~ C(F1) + C(F2) + C(F1):C(F2)', data=df_temp).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # 1. 假设检验 (残差正态性)
            resid = model.resid
            s_shapiro, p_shapiro = stats.shapiro(resid)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**正态性检验 (残差)**")
                st.write(f"P-value = {p_shapiro:.4f} {'✅' if p_shapiro > 0.05 else '❌ (建议慎重)'}")
            with col2:
                st.write("**方差齐性**")
                st.write("Levene检验在双因素下较复杂，建议观察残差图。")

            st.info("💡 推荐方法：**双因素方差分析 (Two-Way ANOVA)**")
            
            # 2. ANOVA 表
            st.write("**ANOVA 结果表**")
            # 翻译索引名以便于阅读
            anova_display = anova_table.rename(index={'C(F1)': f'主效应: {f1}', 'C(F2)': f'主效应: {f2}', 'C(F1):C(F2)': '交互作用'})
            st.dataframe(anova_display.style.format("{:.4f}"))

            # 3. 结果解读
            p_inter = anova_table.loc['C(F1):C(F2)', 'PR(>F)']
            if p_inter < 0.05:
                st.warning(f"🔴 检测到显著的交互作用 (P={p_inter:.4f})！这表明 {f1} 的效果依赖于 {f2}。单独解释主效应可能不准确，建议进行简单效应分析。")
            else:
                st.success(f"🟢 未检测到交互作用 (P={p_inter:.4f})。可以分别解释两个主效应。")

            # 4. 可视化
            with st.expander("交互作用图 (Interaction Plot)", expanded=True):
                fig, ax = plt.subplots()
                from statsmodels.graphics.factorplots import interaction_plot
                # Interaction plot 需要 numpy array
                interaction_plot(x=df_temp['F1'], trace=df_temp['F2'], response=df_temp['Y'], ax=ax)
                ax.set_xlabel(f1)
                ax.set_ylabel(f"Mean of {target_col}")
                ax.legend(title=f2)
                st.pyplot(fig)
