import streamlit as st
import requests
import random
import datetime
import time
from lunar_python import Lunar
from openai import OpenAI

# ==========================================
# 1. 页面配置与 CSS 魔法 (仪式感的核心)
# ==========================================
st.set_page_config(page_title="AI 灵性运势 | 星际指引", page_icon="🔮", layout="centered")

def set_style(bg_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            transition: background-image 0.5s ease-in-out;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.7); /* 背景加深，让卡牌更明显 */
            padding: 3rem;
            border-radius: 20px;
            color: #E0E0E0;
            backdrop-filter: blur(10px);
        }}
        h1, h2, h3, p {{ color: #FFF !important; }}
        
        /* --- ⚡️ 强力仪式感动画：震动+闪烁 --- */
        @keyframes shake-flash {{
            0% {{ transform: translate(1px, 1px) rotate(0deg); filter: brightness(1); }}
            10% {{ transform: translate(-1px, -2px) rotate(-1deg); filter: brightness(1.2); }}
            20% {{ transform: translate(-3px, 0px) rotate(1deg); filter: brightness(1.5) drop-shadow(0 0 10px gold); }}
            30% {{ transform: translate(3px, 2px) rotate(0deg); filter: brightness(1.2); }}
            40% {{ transform: translate(1px, -1px) rotate(1deg); filter: brightness(1); }}
            50% {{ transform: translate(-1px, 2px) rotate(-1deg); filter: brightness(1.2); }}
            60% {{ transform: translate(-3px, 1px) rotate(0deg); filter: brightness(1.5) drop-shadow(0 0 15px cyan); }}
            70% {{ transform: translate(3px, 1px) rotate(-1deg); filter: brightness(1.2); }}
            80% {{ transform: translate(-1px, -1px) rotate(1deg); filter: brightness(1); }}
            90% {{ transform: translate(1px, 2px) rotate(0deg); filter: brightness(1.2); }}
            100% {{ transform: translate(1px, -2px) rotate(-1deg); filter: brightness(1); }}
        }}
        
        .tarot-card-back {{
            width: 220px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            transition: all 0.5s ease;
            cursor: pointer;
            display: block;
            margin: 0 auto;
        }}
        
        /* 激活状态：应用震动动画 */
        .tarot-active {{
            animation: shake-flash 0.5s infinite; /* 0.5秒循环一次，非常快 */
            border: 2px solid #FFF; /* 加个白边框确保能看见变化 */
        }}
        
        /* 按钮样式 */
        .stButton>button {{
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            height: 3.5rem;
            font-size: 1.2rem;
            border-radius: 10px;
        }}
        
        #MainMenu, footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 2. 数据获取层 (真实数据 + 免费API)
# ==========================================

def get_real_weather():
    """获取 IP 定位和真实天气"""
    try:
        # 1. IP 定位
        loc = requests.get('http://ip-api.com/json/?lang=zh-CN', timeout=2).json()
        if loc['status'] != 'success': raise Exception("IP Fail")
        
        # 2. 天气获取
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={loc['lat']}&longitude={loc['lon']}&current=temperature_2m,weather_code&timezone=auto"
        w_data = requests.get(w_url, timeout=2).json()['current']
        
        code = w_data['weather_code']
        # WMO Code 简化映射
        desc = "未知"
        if code == 0: desc = "晴空万里"
        elif code in [1,2,3]: desc = "云层流转"
        elif code in [45,48]: desc = "迷雾笼罩"
        elif code >= 51 and code <= 67: desc = "细雨绵绵"
        elif code >= 80: desc = "雷雨交加"
        else: desc = "风云变幻"
        
        return f"{loc['city']} · {desc} {w_data['temperature_2m']}°C"
    except:
        return "神秘维度 · 能量场稳定 22°C"

def get_huangli():
    """获取黄历"""
    lunar = Lunar.fromDate(datetime.datetime.now())
    return {
        "date": f"{lunar.getMonthInChinese()}月{lunar.getDayInChinese()}",
        "yi": " ".join(lunar.getDayYi()[:4]),
        "ji": " ".join(lunar.getDayJi()[:4])
    }

def draw_tarot_card():
    """塔罗牌库"""
    deck = [
        {"name": "愚人", "en": "The Fool", "key": "无限潜力，新的旅程"},
        {"name": "女祭司", "en": "The High Priestess", "key": "直觉，潜意识的智慧"},
        {"name": "皇后", "en": "The Empress", "key": "丰盛，自然的滋养"},
        {"name": "皇帝", "en": "The Emperor", "key": "秩序，稳固的基础"},
        {"name": "教皇", "en": "The Hierophant", "key": "传统，精神指引"},
        {"name": "恋人", "en": "The Lovers", "key": "和谐，重要的选择"},
        {"name": "战车", "en": "The Chariot", "key": "意志力，克服障碍"},
        {"name": "隐士", "en": "The Hermit", "key": "内省，寻找真理"},
        {"name": "命运之轮", "en": "Wheel of Fortune", "key": "改变，命运的转折"},
        {"name": "正义", "en": "Justice", "key": "因果，真相显现"},
        {"name": "倒吊人", "en": "The Hanged Man", "key": "牺牲，换个角度"},
        {"name": "死神", "en": "Death", "key": "结束，彻底的转化"},
        {"name": "节制", "en": "Temperance", "key": "平衡，疗愈"},
        {"name": "魔鬼", "en": "The Devil", "key": "束缚，物质诱惑"},
        {"name": "塔", "en": "The Tower", "key": "剧变，觉醒"},
        {"name": "星星", "en": "The Star", "key": "希望，灵感"},
        {"name": "月亮", "en": "The Moon", "key": "幻觉，不安"},
        {"name": "太阳", "en": "The Sun", "key": "成功，喜悦"},
        {"name": "审判", "en": "Judgement", "key": "重生，召唤"},
        {"name": "世界", "en": "The World", "key": "圆满，达成"},
    ]
    return random.choice(deck)

# ==========================================
# 3. AI 大脑 (SiliconFlow)
# ==========================================

def consult_oracle(api_key, zodiac, mbti, weather, huangli, card):
    """调用 LLM 生成文案"""
    client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
    
    # 这里填你想要用的模型，比如 Qwen/Qwen2.5-7B-Instruct (免费)
    # 或者 Qwen/Qwen3-8B-Instruct (如果有)
    MODEL_NAME = "Qwen/Qwen3-8B-Instruct"
    
    system_prompt = "你是一位精通荣格心理学、星象学与塔罗奥义的神秘占卜师。你的语言风格是：唯美、治愈、富有哲理且带有一丝神秘感。"
    
    user_prompt = f"""
    请根据以下时空能量进行解读：
    
    【求问者】
    - 星座：{zodiac}
    - MBTI：{mbti}
    - 此时此地：{weather}
    - 历法能量：{huangli['date']} (宜：{huangli['yi']})
    - 命运卡牌：{card['name']} ({card['en']}) - 核心：{card['key']}
    
    【解读要求】
    请用 Markdown 格式输出，包含以下三个章节（不需要标题太长）：
    1. 🌌 **能量共振**：结合天气与黄历，描述当下的整体氛围。
    2. 🎴 **星际指引**：结合塔罗牌与星座，深入剖析今日运势。
    3. 💡 **灵魂建议**：给 {mbti} 人格的 2 条具体行动指南。
    
    最后附上一句简短的箴言。
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ 宇宙信号被干扰：{str(e)} (请检查 API Key)"

# ==========================================
# 4. 界面逻辑 (控制流)
# ==========================================

# 初始化 Session State
if 'bg_url' not in st.session_state:
    st.session_state.bg_url = "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=1920"

# 应用样式
set_style(st.session_state.bg_url)

# --- 侧边栏 ---
with st.sidebar:
    st.title("🧙‍♂️ 档案设定")
    
    # 优先从 Secrets 读取 Key，如果没有则显示输入框
    try:
        api_key = st.secrets["SILICON_KEY"]
        st.success("✅ 密钥已安全加载")
    except FileNotFoundError:
        st.warning("本地模式：请配置 secrets.toml")
        api_key = st.text_input("SiliconFlow Key", type="password")
        
    st.divider()
    zodiac = st.selectbox("星座", ["白羊座","金牛座","双子座","巨蟹座","狮子座","处女座","天秤座","天蝎座","射手座","摩羯座","水瓶座","双鱼座"])
    mbti = st.selectbox("MBTI", ["INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP","ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"])
    st.caption("Designed with AI & Streamlit")

# --- 主界面 ---
st.markdown("<h1 style='text-align: center; letter-spacing: 4px;'>🌌 AI Soul · 命运回响</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8; margin-bottom: 30px;'>连接宇宙意识，聆听内心声音</p>", unsafe_allow_html=True)

# 占位符：用于控制卡牌显示
card_spot = st.empty()
# 占位符：用于显示进度文字
msg_spot = st.empty()

# 牌背图片链接
CARD_BACK_URL = "https://images.unsplash.com/photo-1620052581237-5d36667be337?q=80&w=400&auto=format&fit=crop"

# [状态 A]：还没结果，显示静态牌背 + 按钮
if 'result' not in st.session_state:
    # 1. 显示静态牌背
    card_spot.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <img src="{CARD_BACK_URL}" class="tarot-card-back">
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_btn = st.button("🔮 轻触牌面，开启仪式")

    if start_btn:
        if not api_key:
            st.error("⚠️ 请先在侧边栏配置 API Key")
        else:
            # === 仪式开始 ===
            
            # 1. ⚡️ 立即切换为“震动”牌背
            # 注意：我加了一个 unique_id 防止缓存，确保浏览器重绘
            card_spot.markdown(f"""
                <div style="display: flex; justify-content: center;">
                    <img src="{CARD_BACK_URL}" class="tarot-card-back tarot-active">
                </div>
            """, unsafe_allow_html=True)
            
            # 💡 关键点：给浏览器 0.1 秒去渲染 CSS 动画，否则 Python 会直接卡死 UI
            time.sleep(0.1) 
            
            # 2. 模拟连接过程 (进度提示)
            steps = [
                "⚡️ 能量注入中...",
                "☁️ 正在读取星象...",
                "🌀 阿卡西记录开启...",
                "🧠 AI 先知通灵中..."
            ]
            
            # 3. 进度条与数据并行
            msg_spot = st.empty()
            
            # 每一段 sleep 都会让用户盯着震动的卡牌看
            msg_spot.info(steps[0])
            time.sleep(1.5) 
            
            msg_spot.info(steps[1])
            weather_data = get_real_weather()
            time.sleep(1.0)
            
            msg_spot.info(steps[2])
            huangli_data = get_huangli()
            card_data = draw_tarot_card()
            time.sleep(1.0)
            
            msg_spot.info(steps[3])
            # 调用 AI
            ai_text = consult_oracle(api_key, zodiac, mbti, weather_data, huangli_data, card_data)
            
            # 4. 结果计算完毕，准备展示
            
            # 根据结果决定背景
            if "雨" in weather_data or "死神" in card_data['name'] or "塔" in card_data['name']:
                new_bg = "https://images.unsplash.com/photo-1514477917009-389c76a86b68?q=80&w=1920"
            elif "晴" in weather_data or "太阳" in card_data['name']:
                new_bg = "https://images.unsplash.com/photo-1470252649378-9c29740c9fa8?q=80&w=1920"
            else:
                new_bg = "https://images.unsplash.com/photo-1419242902214-272b3f66ee7a?q=80&w=1920"
            
            # 更新 Session
            st.session_state.bg_url = new_bg
            st.session_state.result = ai_text
            st.session_state.card = card_data
            st.session_state.weather = weather_data
            
            st.rerun()

# [状态 C]：结果展示页
else:
    # 清空之前的卡牌占位（不需要显示牌背了，因为要显示结果）
    card_spot.empty()
    
    st.markdown("---")
    
    # 顶部信息栏
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info(f"📍 {st.session_state.weather}")
    with c2:
        huangli = get_huangli()
        st.success(f"📅 {huangli['date']} · 宜 {huangli['yi']}")

    # 核心展示区
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <h2 style="color: #FFD700; text-shadow: 0 0 10px #FFD700;">🎴 {st.session_state.card['name']}</h2>
        <p style="font-style: italic; opacity: 0.8;">{st.session_state.card['en']} · {st.session_state.card['key']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI 文本渲染
    st.markdown(st.session_state.result)
    
    st.markdown("---")
    
    # 重置按钮
    if st.button("🔄 重新建立连接 (Restart)"):
        # 清除状态
        del st.session_state.result
        del st.session_state.card
        del st.session_state.weather
        st.rerun()

