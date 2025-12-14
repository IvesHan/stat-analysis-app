import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 页面配置 ---
st.set_page_config(page_title="高级统计工具", layout="wide")

# 自定义一些CSS让表格更紧凑
st.markdown("""
<style>
    .stDataFrame { border: 1px solid #f0f2f6; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("📊 高级双因素方差分析工具 (Two-Way ANOVA)")
st.caption("支持矩阵格式直接粘贴 | 自动交互作用分析 | Post-hoc 事后检验")

# --- 1. 数据录入模块 ---
st.header("1. 数据录入")

with st.expander("📖 查看操作指南 (点击展开)", expanded=True):
    st.markdown("""
    1. **直接粘贴**：从 Excel 复制你的数据（包括表头），点击下方表格左上角，按 `Ctrl+V` 粘贴。
    2. **格式要求**：
       - **第 1 列**：必须是**行因素**（例如：吸烟状态、基因型）。
       - **第 2~N 列**：全是**数值列**。你需要告诉程序哪些列属于 Group A，哪些属于 Group B。
    3. **增加行/列**：点击表格右侧或下方的 `+` 号。
    """)

# 初始化默认示例数据 (仿照你的截图)
if 'init_df' not in st.session_state:
    st.session_state.init_df = pd.DataFrame([
        ["Light smoker", 24.1, 29.2, 24.6, 20.0, 21.9, 17.6],
        ["Heavy smoker", 17.6, 18.8, 23.2, 14.8, 10.3, 11.3]
    ], columns=["Condition", "Male_1", "Male_2", "Male_3", "Female_1", "Female_2", "Female_3"])

# 数据编辑器 (核心组件)
edited_df = st.data_editor(
    st.session_state.init_df,
    num_rows="dynamic",  # 允许添加行
    use_container_width=True,
    key="matrix_editor"
)

# --- 2. 变量映射模块 ---
st.header("2. 变量定义与列分组")

# 自动提取列名
all_cols = edited_df.columns.tolist()
first_col = all_cols[0]
data_cols = all_cols[1:]

col_cfg1, col_cfg2 = st.columns(2)

with col_cfg1:
    st.subheader("🅰️ 因素 A (行变量)")
    factor_a_name = st.text_input("给第一列起个名字 (如 Smoking)", value=first_col)
    
with col_cfg2:
    st.subheader("🅱️ 因素 B (列分组)")
    factor_b_name = st.text_input("给列分组起个名字 (如 Gender)", value="Gender")

st.markdown("---")
st.markdown("#### 👇 请分配数据列到 因素 B 的水平 (Subgroups)")

# 动态列分配
# 默认认为前一半是组1，后一半是组2 (智能预判)
half = len(data_cols) // 2
c1, c2 = st.columns(2)

with c1:
    group1_label = st.text_input("分组 1 名称 (如 Male)", value="Group_1")
    group1_cols = st.multiselect(f"选择属于 {group1_label} 的列", data_cols, default=data_cols[:half])

with c2:
    group2_label = st.text_input("分组 2 名称 (如 Female)", value="Group_2")
    # 自动排除已被组1选走的列
    remaining_cols = [c for c in data_cols if c not in group1_cols]
    group2_cols = st.multiselect(f"选择属于 {group2_label} 的列", remaining_cols, default=remaining_cols)

# 校验
if not group1_cols or not group2_cols:
    st.warning("⚠️ 请确保两个分组都至少分配了一列数据。")
    st.stop()

# --- 3. 分析执行模块 ---
st.markdown("---")
if st.button("🚀 开始分析", type="primary", use_container_width=True):
    
    # === A. 数据清洗与重构 (Melt) ===
    long_data = []
    
    try:
        # 遍历每一行
        for index, row in edited_df.iterrows():
            level_a = row[first_col]  # 获取行标签 (如 Light smoker)
            
            # 提取 Group 1 数据
            for col in group1_cols:
                val = pd.to_numeric(row[col], errors='coerce') # 转数字，非数字变NaN
                if not pd.isna(val):
                    long_data.append({
                        factor_a_name: str(level_a),
                        factor_b_name: group1_label,
                        "Value": val
                    })
            
            # 提取 Group 2 数据
            for col in group2_cols:
                val = pd.to_numeric(row[col], errors='coerce')
                if not pd.isna(val):
                    long_data.append({
                        factor_a_name: str(level_a),
                        factor_b_name: group2_label,
                        "Value": val
                    })
        
        df_long = pd.DataFrame(long_data)
        
        if df_long.empty:
            st.error("没有提取到有效数据，请检查输入表格是否包含数字。")
            st.stop()

    except Exception as e:
        st.error(f"数据处理出错: {e}")
        st.stop()

    # === B. 统计建模 (Two-Way ANOVA) ===
    st.header("3. 分析报告")
    
    # 准备数据列名 (重命名为标准变量名以免公式报错)
    df_model = df_long.rename(columns={factor_a_name: 'FA', factor_b_name: 'FB', 'Value': 'Y'})
    
    # 拟合 OLS 模型
    model = ols('Y ~ C(FA) + C(FB) + C(FA):C(FB)', data=df_model).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # 格式化输出表
    anova_display = anova_table.rename(index={
        'C(FA)': f'主效应: {factor_a_name}', 
        'C(FB)': f'主效应: {factor_b_name}', 
        'C(FA):C(FB)': '交互作用 (Interaction)'
    })

    # 显示 ANOVA 表
    st.subheader("📋 方差分析表 (ANOVA Table)")
    
    # 高亮显著的 P 值
    def highlight_p(s):
        is_sig = s < 0.05
        return ['background-color: #d4edda; color: green; font-weight: bold' if is_sig else '' for v in s]

    st.dataframe(
        anova_display.style.format("{:.4f}")
        .apply(lambda x: highlight_p(x), subset=['PR(>F)'])
    )

    # 获取交互作用的 P 值
    p_interaction = anova_table.loc['C(FA):C(FB)', 'PR(>F)']

    # === C. 结果解读 ===
    st.subheader("💡 结果解读")
    if p_interaction < 0.05:
        st.warning(f"🔴 **检测到显著的交互作用 (P={p_interaction:.4f})**")
        st.write(f"这意味着 **{factor_b_name}** 对结果的影响取决于 **{factor_a_name}**。建议重点关注下方的“两两比较”和“交互作用图”。")
    else:
        st.success(f"🟢 **未检测到交互作用 (P={p_interaction:.4f})**")
        st.write("各因素的影响是独立的。你可以分别解释主效应的 P 值。")

    # === D. 事后多重比较 (Post-hoc) ===
    st.markdown("---")
    st.subheader("🔍 事后两两比较 (Tukey HSD Post-hoc)")
    st.caption("比较所有【行因素 + 列因素】组合之间的差异")
    
    # 构建组合列 (例如: Light smoker + Male)
    df_long['Combination'] = df_long[factor_a_name].astype(str) + " + " + df_long[factor_b_name].astype(str)
    
    # 执行 Tukey 检验
    tukey = pairwise_tukeyhsd(endog=df_long['Value'], groups=df_long['Combination'], alpha=0.05)
    
    # 整理结果
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    # 筛选显著结果
    sig_df = tukey_df[tukey_df['reject'] == True].copy()
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.write("**🔴 存在显著差异的组对 (Significant Pairs):**")
        if not sig_df.empty:
            st.dataframe(sig_df[['group1', 'group2', 'p-adj', 'meandiff']].style.format({'p-adj': '{:.4f}', 'meandiff': '{:.2f}'}))
        else:
            st.info("未发现任何组对之间存在显著差异。")
            
    with col_res2:
        with st.expander("查看所有比较结果 (All Pairs)"):
            st.dataframe(tukey_df)

    # === E. 可视化 (交互作用图) ===
    st.markdown("---")
    st.subheader("📈 交互作用图 (Interaction Plot)")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 使用 Seaborn 绘制点线图
    sns.pointplot(
        data=df_long, 
        x=factor_a_name, 
        y="Value", 
        hue=factor_b_name, 
        markers=["o", "s"], 
        capsize=.1, 
        err_kws={'linewidth': 1.5},
        linestyle='--',
        ax=ax
    )
    
    ax.set_title(f"Interaction: {factor_a_name} × {factor_b_name}")
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

    # === F. 描述统计 ===
    with st.expander("查看详细描述统计 (均值/标准差)"):
        desc_stats = df_long.groupby([factor_a_name, factor_b_name])['Value'].agg(['count', 'mean', 'std']).reset_index()
        st.dataframe(desc_stats.style.format("{:.2f}"))
