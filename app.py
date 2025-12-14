import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 页面配置 ---
st.set_page_config(page_title="高级双因素统计工具", layout="wide")
st.title("📊 高级双因素方差分析 (带矩阵输入 & Post-hoc)")

# --- 1. 数据录入区 ---
st.header("1. 数据录入")
st.markdown("""
**操作指南：**
1. 在下方表格中，**第一列**输入行因素（如：Light smoker, Heavy smoker）。
2. 后面的列输入数值。如果一组有3个重复，就占用3列。
3. 点击表格右侧/下方的 `+` 号可以自由添加行和列。
""")

# 初始化默认数据 (仿照你的图片结构)
# 行头 + Group A (3 cols) + Group B (3 cols)
default_data = pd.DataFrame([
    ["Light smoker", 24.1, 29.2, 24.6, 20.0, 21.9, 17.6],
    ["Heavy smoker", 17.6, 18.8, 23.2, 14.8, 10.3, 11.3]
], columns=["Row_Factor", "A1", "A2", "A3", "B1", "B2", "B3"])

# 数据编辑器
edited_df = st.data_editor(
    default_data,
    num_rows="dynamic", # 允许加行
    use_container_width=True,
    key="matrix_editor"
)

# --- 2. 列映射设置 (关键步骤) ---
st.header("2. 定义列与分组")
st.info("👆 请告诉程序，上面表格中的哪些列属于哪个分组（因素 B）。")

# 自动获取除第一列外的所有列名
data_cols = edited_df.columns.tolist()[1:] 
row_factor_name = st.text_input("行因素名称 (Factor A)", value="Smoking_Status")
col_factor_name = st.text_input("列因素名称 (Factor B)", value="Gender_Group")

# 动态分组配置
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### 分组 1 设置")
    group1_name = st.text_input("分组 1 名称", value="Male (Group A)")
    group1_cols = st.multiselect("选择属于分组 1 的数据列", data_cols, default=data_cols[:3])

with col2:
    st.markdown("##### 分组 2 设置")
    group2_name = st.text_input("分组 2 名称", value="Female (Group B)")
    group2_cols = st.multiselect("选择属于分组 2 的数据列", data_cols, default=data_cols[3:])

# 还可以增加分组3、4 (如果有需要可以扩展)
# 这里演示支持两组列因素，如果需要更多组，可以依葫芦画瓢添加

# --- 3. 数据重构 (Reshape) ---
if st.button("开始分析", type="primary"):
    st.divider()
    
    # 验证输入
    if not group1_cols or not group2_cols:
        st.error("请至少为两个分组分配数据列！")
        st.stop()
        
    # --- 核心逻辑：将宽矩阵转换为长格式 (Melt) ---
    long_data = []
    
    # 遍历每一行
    for index, row in edited_df.iterrows():
        r_label = row[edited_df.columns[0]] # 获取行标签 (Light/Heavy)
        
        # 提取 Group 1 数据
        for col in group1_cols:
            val = pd.to_numeric(row[col], errors='coerce') # 强制转数字
            if not pd.isna(val): # 去除空值
                long_data.append({
                    row_factor_name: r_label,
                    col_factor_name: group1_name,
                    "Value": val
                })
                
        # 提取 Group 2 数据
        for col in group2_cols:
            val = pd.to_numeric(row[col], errors='coerce')
            if not pd.isna(val):
                long_data.append({
                    row_factor_name: r_label,
                    col_factor_name: group2_name,
                    "Value": val
                })
    
    df_long = pd.DataFrame(long_data)
    
    # 展示转换后的数据（调试用）
    with st.expander("查看重构后的长格式数据 (用于统计后台)"):
        st.dataframe(df_long)
        
    # --- 4. 统计分析 ---
    st.header("3. 分析报告")
    
    # (A) 描述统计
    st.subheader("📊 描述统计 (均值 ± 标准差)")
    summary = df_long.groupby([row_factor_name, col_factor_name])['Value'].agg(['mean', 'std', 'count']).reset_index()
    st.dataframe(summary.style.format("{:.2f}"))

    # (B) 双因素方差分析 (Two-Way ANOVA)
    st.subheader("Expected Result: ANOVA Table")
    
    # 重命名列以适应公式（去除特殊字符）
    df_model = df_long.rename(columns={row_factor_name: 'FactorA', col_factor_name: 'FactorB', 'Value': 'Y'})
    
    # OLS 模型
    model = ols('Y ~ C(FactorA) + C(FactorB) + C(FactorA):C(FactorB)', data=df_model).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # 结果美化
    anova_display = anova_table.rename(index={
        'C(FactorA)': f'主效应: {row_factor_name}', 
        'C(FactorB)': f'主效应: {col_factor_name}', 
        'C(FactorA):C(FactorB)': '交互作用 (Interaction)'
    })
    
    def highlight_sig(s):
        is_sig = s < 0.05
        return ['background-color: #d1e7dd' if is_sig else '' for v in s]

    st.dataframe(anova_display.style.format("{:.4f}").apply(lambda x: ['background-color: #ffffcc' if v < 0.05 else '' for v in x], subset=['PR(>F)']))

    # 交互作用判断
    p_inter = anova_table.loc['C(FactorA):C(FactorB)', 'PR(>F)']
    
    # (C) Post-hoc Analysis (这是你重点要求的)
    st.subheader("🔍 事后多重比较 (Post-hoc)")
    
    # 策略：生成一个新的组合变量 "FactorA - FactorB"
    # 这样可以直接比较所有组合 (例如 Light-Male vs Heavy-Female)
    df_long['Combination'] = df_long[row_factor_name].astype(str) + " + " + df_long[col_factor_name].astype(str)
    
    tukey = pairwise_tukeyhsd(endog=df_long['Value'], groups=df_long['Combination'], alpha=0.05)
    
    # 将 Tukey 结果转为 DataFrame
    tukey_data = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    # 筛选显著的结果
    sig_results = tukey_data[tukey_data['reject'] == True]
    
    if not sig_results.empty:
        st.write("🔴 **发现显著差异的组对 (P < 0.05):**")
        st.dataframe(sig_results.style.format({'p-adj': '{:.4f}'}))
    else:
        st.write("⚪ 未发现显著的两两差异。")
        
    with st.expander("查看所有两两比较结果"):
        st.dataframe(tukey_data)

    # (D) 可视化
    st.subheader("📈 交互作用图")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 传统的交互作用图
    from statsmodels.graphics.factorplots import interaction_plot
    # interaction_plot 需要数值型的 x 轴有时会方便点，这里直接用分类
    sns.pointplot(data=df_long, x=row_factor_name, y="Value", hue=col_factor_name, 
                  markers=["o", "s"], capsize=.1, errorbar="se", ax=ax)
    
    ax.set_title("Interaction Plot (Mean ± SE)")
    st.pyplot(fig)
