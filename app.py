import streamlit as st
import importlib.util

# 设置页面配置
st.set_page_config(
    page_title="NeuroPredict Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_page(page_path):
    """动态加载页面"""
    try:
        spec = importlib.util.spec_from_file_location("page", page_path)
        page_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(page_module)

        if hasattr(page_module, 'main'):
            page_module.main()
    except Exception as e:
        st.error(f"加载页面时出错: {str(e)}")
        st.error(f"页面路径: {page_path}")


# 自定义CSS样式
st.markdown("""
<style>
    /* 侧边栏背景色 */
    [data-testid=stSidebar] {
        background-color: #1D5746;
    }

    /* 侧边栏一般文字颜色（不包括按钮） */
    [data-testid=stSidebar] .css-1d391kg {
        color: white;
    }

    [data-testid=stSidebar] h1,
    [data-testid=stSidebar] h2,
    [data-testid=stSidebar] h3 {
        color: white;
    }

    [data-testid=stSidebar] p:not(.stButton p) {
        color: white;
    }

    /* 非当前页面按钮样式 - 绿底白字 */
    [data-testid=stSidebar] .stButton > button[kind="secondary"] {
        width: 100%;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border: 2px solid #2D7556;
        border-radius: 10px;
        background-color: #2D7556;
        color: white;
        text-align: center;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    [data-testid=stSidebar] .stButton > button[kind="secondary"]:hover {
        background-color: #3D8566;
        border-color: #4D9576;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(45, 117, 86, 0.3);
    }

    /* 当前页面按钮样式 - 白底绿字 */
    [data-testid=stSidebar] .stButton > button[kind="primary"] {
        width: 100%;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border: 2px solid #ffffff;
        border-radius: 10px;
        background-color: #ffffff;
        color: #1D5746;
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    [data-testid=stSidebar] .stButton > button[kind="primary"]:hover {
        background-color: #f0f0f0;
        border-color: #e0e0e0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
    }

    .nav-section-title {
        color: #B8E6C8;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 1.5rem 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    # Logo在最顶部
    st.image("Neu.png", width=250)

    # 导航标题
    st.markdown('<p class="nav-section-title">Navigation</p>', unsafe_allow_html=True)

    # 初始化session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"

    # 创建导航按钮
    pages = ["Dashboard", "Descriptive", "Diagnostic"]

    for page in pages:
        # 当前页面使用primary类型（白底绿字），其他页面使用secondary类型（绿底白字）
        button_type = "primary" if st.session_state.current_page == page else "secondary"

        if st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True,
                type=button_type
        ):
            st.session_state.current_page = page
            st.rerun()

    st.markdown("---")

# 页面映射
page_files = {
    "Dashboard": "app_pages/Dashboard.py",
    "Descriptive": "app_pages/Descriptive.py",
    "Diagnostic": "app_pages/Diagnostic.py"
}

# 加载选中的页面
current_file = page_files[st.session_state.current_page]
load_page(current_file)