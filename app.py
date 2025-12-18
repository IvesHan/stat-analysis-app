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
    """检测文本文件分隔符"""
    try:
        sample = file_buffer.read(2048).decode("utf-8")
        file_buffer.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter
    except:
        file_buffer.seek(0)
        return ","

def load_data(uploaded_file, skip_rows=0, header_row=0, sep=None, sheet_name=0):
    """
    通用加载函数
    新增参数: sheet_name (可以是索引数字，也可以是名称字符串)
    """
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext in ['xls', 'xlsx']:
        # Excel 读取指定 Sheet
        return pd.read_excel(uploaded_file, skiprows=skip_rows, header=header_row, sheet_name=sheet_name)
    else:
        # 文本文件忽略 sheet_name
        if sep is None:
            sep = detect_separator(uploaded_file)
        return pd.read_csv(uploaded_file, sep=sep, skiprows=skip_rows, header=header_row, engine='python')

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output

# ========================================================
# 模式 1: 单表处理 (新增 Sheet 选择功能)
# ========================================================
if app_mode == "单表处理 (清洗/筛选/透视)":
    
    st.sidebar.subheader("1. 文件导入")
    uploaded_file = st.sidebar.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls', 'tsv', 'txt'])
    
    if uploaded_file:
        # 获取文件扩展名
        file_ext = uploaded_file.name.split('.')[-1].lower()

        # --- 动态参数配置区 ---
        with st.sidebar.expander("读取参数配置", expanded=True):
            
            # [新增功能] 如果是 Excel，先解析 Sheet 列表供用户选择
            selected_sheet = 0 # 默认读取第一个
            if file_ext in ['xlsx', 'xls']:
                try:
                    # 使用 ExcelFile 读取元数据，不加载内容，速度快
                    xl = pd.ExcelFile(uploaded_file)
                    sheet_names = xl.sheet_names
                    st.markdown("#### Excel 工作表")
                    selected_sheet = st.selectbox("选择要读取的 Sheet", sheet_names)
                    # 重置指针，防止后续读取报错
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Excel 解析失败: {e}")

            # 通用参数
            st.markdown("#### 行设置")
            skip_rows = st.number_input("跳过前 N 行 (Skip Rows)", 0, 100, 0)
            header_row = st.number_input("标题所在行 (Header)", 0, 100, 0)
            
            # CSV 专用参数
            sep = None
            if file_ext not in ['xlsx', 'xls']:
                st.markdown("#### 分隔符")
                sep_option = st.selectbox("列分隔符", ("自动识别", ",", "\t", ";", "|", "自定义"))
                if sep_option == "自定义":
                    sep = st.text_input("输入分隔符", ",")
                elif sep_option != "自动识别":
                    sep_map = {",": ",", "\t": "\t", ";": ";", "|": "|"}
                    sep = sep_map.get(sep_option, ",")

        try:
            # 加载数据 (传入 sheet_name)
            df_raw = load_data(uploaded_file, skip_rows, header_row, sep, sheet_name=selected_sheet)
            st.sidebar.success(f"读取成功: {len(df_raw)} 行")

            # --- 下方逻辑保持不变 ---
            tab_clean, tab_pivot = st.tabs(["🧹 数据清洗与导出", "📈 数据透视表"])

            # [Tab 1: 清洗]
            with tab_clean:
                # 1. 列选择
                st.subheader("1. 列选择与排序")
                c1, c2 = st.columns([3, 1])
                with c1:
                    all_cols = df_raw.columns.tolist()
                    selected_cols = st.multiselect("保留列", all_cols, default=all_cols)
                    if not selected_cols: selected_cols = all_cols
                with c2:
                    sort_col = st.selectbox("排序依据", ["无"] + selected_cols)
                    sort_asc = st.radio("排序", ["升序", "降序"], horizontal=True, label_visibility="collapsed")

                df_step1 = df_raw[selected_cols].copy()
                if sort_col != "无":
                    ascending = True if sort_asc == "升序" else False
                    df_step1 = df_step1.sort_values(by=sort_col, ascending=ascending)

                # 2. 内容筛选
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
                                text_input = st.text_area("输入筛选值 (支持多行/逗号分隔)", height=80)
                                match_mode = st.radio("模式", ["精确匹配 (Is In)", "模糊包含 (Contains)"], horizontal=True)

                                if text_input.strip():
                                    keywords = [k for k in re.split(r'[,\s;，；|\n]+', text_input.strip()) if k]
                                    if keywords:
                                        if match_mode == "精确匹配 (Is In)":
                                            df_result = df_step1[df_step1[filter_target].astype(str).isin(keywords)]
                                        else:
                                            pattern = "|".join([re.escape(k) for k in keywords])
                                            df_result = df_step1[df_step1[filter_target].astype(str).str.contains(pattern, case=False, na=False)]
                
                # 3. 行截取 (手动输入)
                st.subheader("3. 行截取 (精确范围)")
                current_total = len(df_result)
                if current_total > 0:
                    r_col1, r_col2 = st.columns(2)
                    with r_col1:
                        start_idx = st.number_input("起始行 (Start)", 0, current_total-1, 0)
                    with r_col2:
                        end_idx = st.number_input("结束行 (End)", start_idx+1, current_total, current_total)
                    df_result = df_result.iloc[start_idx:end_idx]

                # 4. 导出
                st.divider()
                st.subheader(f"4. 结果预览与导出 (共 {len(df_result)} 行)")
                m1, m2 = st.columns(2)
                m1.metric("原始行数", len(df_raw))
                m2.metric("当前行数", len(df_result), delta=len(df_result)-len(df_raw))
                
                st.dataframe(df_result, use_container_width=True)
                
                d_col1, d_col2 = st.columns(2)
                file_name_base = uploaded_file.name.split('.')[0]
                d_col1.download_button("📥 下载 Excel", to_excel(df_result), f"{file_name_base}_cleaned_ives.xlsx")
                d_col2.download_button("📥 下载 CSV", df_result.to_csv(index=False).encode('utf-8-sig'), f"{file_name_base}_cleaned_ives.csv", "text/csv")

            # [Tab 2: 透视表]
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
# 模式 2: 多表合并
# ========================================================
elif app_mode == "多表合并":
    st.subheader("📚 多文件合并工具")
    merge_type = st.radio("合并方式", ["纵向拼接 (Concat)", "横向关联 (Merge/Join)"], captions=["行增多 (结构相同)", "列增多 (按Key关联)"])
    st.divider()
    files = st.file_uploader("批量上传文件", accept_multiple_files=True)
    
    if files:
        if len(files) < 2:
            st.warning("请至少上传两个文件。")
        else:
            # 纵向拼接
            if merge_type == "纵向拼接 (Concat)":
                if st.button("开始纵向合并"):
                    dfs = []
                    bar = st.progress(0)
                    for i, f in enumerate(files):
                        try:
                            # 默认读取第一个 Sheet
                            d = load_data(f, sheet_name=0)
                            d['Source_File'] = f.name 
                            dfs.append(d)
                        except: st.error(f"读取失败: {f.name}")
                        bar.progress((i+1)/len(files))
                    
                    if dfs:
                        merged = pd.concat(dfs, ignore_index=True)
                        st.success(f"合并完成: {len(merged)} 行")
                        st.dataframe(merged.head(100), use_container_width=True)
                        st.download_button("下载结果", to_excel(merged), "concat_result_ives.xlsx")

            # 横向关联
            else: 
                st.subheader("🔗 关联配置")
                file_cols_map = {}
                dfs_map = {}
                cols_config = st.columns(len(files))
                selected_keys = []
                
                try:
                    for i, f in enumerate(files):
                        f.seek(0)
                        # 默认读取第一个 Sheet
                        df_temp = load_data(f, sheet_name=0) 
                        dfs_map[f.name] = df_temp
                        with cols_config[i]:
                            st.markdown(f"**{f.name}**")
                            default_idx = 0
                            for idx, c in enumerate(df_temp.columns):
                                if c.lower() in ['id', 'no', 'code', 'key', '工号']: default_idx = idx
                                break
                            key_col = st.selectbox(f"关联键", df_temp.columns, index=default_idx, key=f"key_{i}")
                            selected_keys.append(key_col)

                    join_how = st.selectbox("连接方式", ["inner", "left", "outer"], index=1).split()[0]

                    if st.button("开始横向关联"):
                        base_df = dfs_map[files[0].name]
                        base_key = selected_keys[0]
                        base_df[base_key] = base_df[base_key].astype(str)
                        current_df = base_df
                        
                        for i in range(1, len(files)):
                            next_name = files[i].name
                            next_df = dfs_map[next_name]
                            next_key = selected_keys[i]
                            next_df[next_key] = next_df[next_key].astype(str)
                            
                            current_df = pd.merge(current_df, next_df, left_on=base_key if i==1 else None, right_on=next_key, how=join_how, suffixes=('', f'_{i}'))

                        st.success("关联成功")
                        st.dataframe(current_df.head(50), use_container_width=True)
                        st.download_button("下载关联结果", to_excel(current_df), "merged_join_result_ives.xlsx")

                except Exception as e:
                    st.error(f"错误: {e}")
