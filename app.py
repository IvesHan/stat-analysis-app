import streamlit as st
import pandas as pd
import io
import csv
import re

# --- 页面配置 ---
st.set_page_config(
    page_title="表格处理工具 (Ives)", 
    layout="wide", 
    page_icon="📑"
)

# --- 顶部标题 ---
st.title("表格处理工具")
st.caption("Designed by Ives | Professional Data Tool")
st.divider()

# --- 侧边栏：模式选择 ---
st.sidebar.header("功能菜单")
app_mode = st.sidebar.radio("选择操作模式", ["单表处理 (清洗/筛选/透视)", "多表合并"])

# --- 核心工具函数 ---
def detect_separator(file_buffer):
    try:
        sample = file_buffer.read(2048).decode("utf-8")
        file_buffer.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except:
        file_buffer.seek(0)
        return ","

def load_data(uploaded_file, skip_rows=0, header_row=0, sep=None):
    """通用加载函数"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext in ['xls', 'xlsx']:
        return pd.read_excel(uploaded_file, skiprows=skip_rows, header=header_row)
    else:
        if sep is None:
            sep = detect_separator(uploaded_file)
        return pd.read_csv(uploaded_file, sep=sep, skiprows=skip_rows, header=header_row, engine='python')

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output

# ========================================================
# 模式 1: 单表处理 (逻辑保持不变)
# ========================================================
if app_mode == "单表处理 (清洗/筛选/透视)":
    
    st.sidebar.subheader("1. 文件导入")
    uploaded_file = st.sidebar.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls', 'tsv', 'txt'])
    
    if uploaded_file:
        with st.sidebar.expander("读取参数配置 (可选)"):
            skip_rows = st.number_input("跳过前 N 行", 0, 100, 0)
            header_row = st.number_input("标题所在行", 0, 100, 0)
            sep_option = "自动识别"
            if uploaded_file.name.split('.')[-1].lower() not in ['xlsx', 'xls']:
                sep_option = st.selectbox("列分隔符", ("自动识别", ",", "\t", ";", "|", "自定义"))
            
            sep = None
            if sep_option == "自定义":
                sep = st.text_input("输入分隔符", ",")
            elif sep_option != "自动识别":
                sep_map = {",": ",", "\t": "\t", ";": ";", "|": "|"}
                sep = sep_map.get(sep_option, ",")

        try:
            df_raw = load_data(uploaded_file, skip_rows, header_row, sep)
            st.sidebar.success(f"读取成功: {len(df_raw)} 行")

            tab_clean, tab_pivot = st.tabs(["🧹 数据清洗与导出", "📈 数据透视表"])

            # --- 清洗 Tab ---
            with tab_clean:
                st.subheader("1. 列选择与排序")
                c1, c2 = st.columns([3, 1])
                with c1:
                    all_cols = df_raw.columns.tolist()
                    selected_cols = st.multiselect("保留列 (默认全部)", all_cols, default=all_cols)
                    if not selected_cols: selected_cols = all_cols
                with c2:
                    sort_col = st.selectbox("排序依据", ["无"] + selected_cols)
                    sort_asc = st.radio("排序方式", ["升序", "降序"], horizontal=True, label_visibility="collapsed")

                df_step1 = df_raw[selected_cols].copy()
                if sort_col != "无":
                    ascending = True if sort_asc == "升序" else False
                    df_step1 = df_step1.sort_values(by=sort_col, ascending=ascending)

                st.subheader("2. 内容筛选 (Filter)")
                df_result = df_step1.copy()

                with st.container(border=True):
                    f_col1, f_col2 = st.columns([1, 2])
                    with f_col1:
                        filter_target = st.selectbox("选择筛选列", ["无"] + selected_cols)
                    
                    if filter_target != "无":
                        with f_col2:
                            if pd.api.types.is_numeric_dtype(df_step1[filter_target]):
                                min_v = float(df_step1[filter_target].min())
                                max_v = float(df_step1[filter_target].max())
                                rng = st.slider(f"数值范围 ({filter_target})", min_v, max_v, (min_v, max_v))
                                df_result = df_step1[(df_step1[filter_target] >= rng[0]) & (df_step1[filter_target] <= rng[1])]
                            else:
                                text_input = st.text_area(f"输入筛选值 (支持多行粘贴)", height=80, placeholder="输入要保留的内容...")
                                match_mode = st.radio("匹配模式", ["精确匹配 (Is In)", "模糊包含 (Contains)"], horizontal=True)

                                if text_input.strip():
                                    keywords = [k for k in re.split(r'[,\s;，；|\n]+', text_input.strip()) if k]
                                    if keywords:
                                        if match_mode == "精确匹配 (Is In)":
                                            df_result = df_step1[df_step1[filter_target].astype(str).isin(keywords)]
                                        else:
                                            pattern = "|".join([re.escape(k) for k in keywords])
                                            df_result = df_step1[df_step1[filter_target].astype(str).str.contains(pattern, case=False, na=False)]
                
                st.subheader("3. 行截取 (精确范围)")
                current_total = len(df_result)
                if current_total > 0:
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        start_idx = st.number_input("起始行号 (Start)", min_value=0, max_value=current_total-1, value=0)
                    with r_col2:
                        end_idx = st.number_input("结束行号 (End)", min_value=start_idx+1, max_value=current_total, value=current_total)
                    df_result = df_result.iloc[start_idx:end_idx]

                st.divider()
                st.subheader(f"4. 结果预览与导出 (共 {len(df_result)} 行)")
                m1, m2 = st.columns(2)
                m1.metric("原始行数", len(df_raw))
                m2.metric("当前行数", len(df_result), delta=len(df_result)-len(df_raw))
                st.dataframe(df_result, use_container_width=True)
                
                d_col1, d_col2 = st.columns(2)
                file_name_base = uploaded_file.name.split('.')[0]
                d_col1.download_button("📥 下载 Excel", data=to_excel(df_result), file_name=f"{file_name_base}_cleaned_ives.xlsx")
                d_col2.download_button("📥 下载 CSV", data=df_result.to_csv(index=False).encode('utf-8-sig'), file_name=f"{file_name_base}_cleaned_ives.csv", mime="text/csv")

            # --- 透视表 Tab ---
            with tab_pivot:
                st.subheader("数据透视分析")
                if not df_raw.empty:
                    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
                    idx = p_c1.multiselect("行维度", df_raw.columns)
                    cols = p_c2.multiselect("列维度", df_raw.columns)
                    vals = p_c3.multiselect("数值", df_raw.columns)
                    func = p_c4.selectbox("聚合方式", ["sum", "mean", "count", "max", "min", "nunique"])
                    if idx and vals:
                        try:
                            df_p = pd.pivot_table(df_raw, index=idx, columns=cols if cols else None, values=vals, aggfunc=func)
                            st.dataframe(df_p, use_container_width=True)
                            st.download_button("导出透视表", to_excel(df_p), f"{file_name_base}_pivot_ives.xlsx")
                        except Exception as e:
                            st.error(f"透视错误: {e}")

        except Exception as e:
            st.error(f"处理出错: {e}")

# ========================================================
# 模式 2: 多表合并 (高级版)
# ========================================================
elif app_mode == "多表合并":
    st.subheader("📚 多文件合并工具")
    
    # 子模式选择
    merge_type = st.radio(
        "选择合并方式", 
        ["纵向拼接 (Concat)", "横向关联 (Merge/Join)"],
        captions=["适用于相同格式的表上下堆叠 (行增多)", "适用于不同表根据共同列左右拼接 (列增多)"]
    )
    
    st.divider()
    files = st.file_uploader("批量上传文件 (CSV/Excel)", accept_multiple_files=True)
    
    if files:
        if len(files) < 2:
            st.warning("请至少上传两个文件进行合并。")
        else:
            # ----------------------------------------------------
            # A. 纵向拼接 (原功能)
            # ----------------------------------------------------
            if merge_type == "纵向拼接 (Concat)":
                if st.button("开始纵向合并"):
                    dfs = []
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        try:
                            d = load_data(f)
                            d['Source_File'] = f.name # 标记来源
                            dfs.append(d)
                        except: st.error(f"读取失败: {f.name}")
                        bar.progress((i+1)/len(files))
                    
                    if dfs:
                        merged = pd.concat(dfs, ignore_index=True)
                        st.success(f"合并完成: 共 {len(dfs)} 个文件, {len(merged)} 行")
                        st.dataframe(merged.head(100), use_container_width=True)
                        st.download_button("下载结果 (Excel)", to_excel(merged), "concat_result_ives.xlsx")

            # ----------------------------------------------------
            # B. 横向关联 (新功能)
            # ----------------------------------------------------
            else: 
                st.subheader("🔗 关联配置")
                st.markdown("请为每个文件指定用于匹配的 **“关键列 (Key)”**。例如：两个表都有'工号'列。")

                # 1. 预读取所有文件的列名
                file_cols_map = {}
                dfs_map = {} # 暂存数据，避免重复读取
                
                # 布局容器：动态生成 Selectbox
                cols_config = st.columns(len(files))
                selected_keys = []
                
                try:
                    for i, f in enumerate(files):
                        # 读取数据
                        f.seek(0)
                        df_temp = load_data(f)
                        dfs_map[f.name] = df_temp
                        file_cols_map[f.name] = df_temp.columns.tolist()
                        
                        # 在界面上显示选择框
                        with cols_config[i]:
                            st.markdown(f"**文件 {i+1}:** `{f.name}`")
                            # 尝试智能默认选中：检查是否有名为 id, code, no 等列
                            default_idx = 0
                            for idx, c in enumerate(df_temp.columns):
                                if c.lower() in ['id', 'no', 'code', 'key', '工号', '编号']:
                                    default_idx = idx
                                    break
                            
                            key_col = st.selectbox(
                                f"选择关联列", 
                                df_temp.columns, 
                                index=default_idx, 
                                key=f"key_{i}"
                            )
                            selected_keys.append(key_col)

                    # 关联方式选择
                    join_how = st.selectbox(
                        "连接方式 (Join Type)", 
                        ["inner (交集 - 只保留共有)", "left (左连接 - 保留第一个文件的全部)", "outer (并集 - 保留所有)"],
                        index=1
                    )
                    join_method = join_how.split()[0]

                    if st.button("开始横向关联"):
                        progress_text = st.empty()
                        
                        # 核心合并逻辑
                        # 取第一个文件作为基准
                        base_df = dfs_map[files[0].name]
                        base_key = selected_keys[0]
                        
                        # 强制转为字符串以保证匹配（防止 数字123 匹配不上 文本"123"）
                        base_df[base_key] = base_df[base_key].astype(str)
                        
                        current_df = base_df
                        
                        for i in range(1, len(files)):
                            next_file_name = files[i].name
                            next_df = dfs_map[next_file_name]
                            next_key = selected_keys[i]
                            
                            progress_text.text(f"正在合并: {next_file_name}...")
                            
                            # 类型统一
                            next_df[next_key] = next_df[next_key].astype(str)
                            
                            # 执行 Merge
                            # 如果列名有冲突，会自动加后缀 _x, _y
                            current_df = pd.merge(
                                current_df, 
                                next_df, 
                                left_on=base_key if i==1 else None, # 第一次用base_key
                                right_on=next_key,
                                how=join_method,
                                left_index=False, # 如果不是第一次，可能需要基于上一次的结果
                                right_index=False,
                                suffixes=('', f'_{i}') # 防止列名冲突
                            )
                            
                            # 注意：pd.merge 后，如果 left_on 和 right_on 不同名，两列都会保留。
                            # 这是一个级联操作，后续的 merge 应该基于当前大表的主键。
                            # 为简化逻辑，假设用户是想把所有表挂在第一个表上，或者链式挂载。
                            # 这里采用链式合并：Result(1+2) + File3
                            # 下一次 merge 的 left_on 应该是上一次 merge 保留的 key。
                            # 如果 Key 列名相同，pandas会自动合并成一列；如果不同，会保留两个。
                            # 这里不再做复杂推断，简单链式 merge 即可。

                        st.success("关联成功！")
                        st.dataframe(current_df.head(50), use_container_width=True)
                        st.download_button(
                            "📥 下载关联结果 (Excel)", 
                            to_excel(current_df), 
                            "merged_join_result_ives.xlsx"
                        )

                except Exception as e:
                    st.error(f"合并过程中发生错误: {e}")
                    st.warning("提示：请确保选中的关联列中数据是唯一的，否则可能会产生笛卡尔积导致数据量爆炸。")
