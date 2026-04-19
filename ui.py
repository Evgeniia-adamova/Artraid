import streamlit as st

# глобальные стили
def apply_global_styles():
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            padding-left: 1.8rem;
            padding-right: 1.8rem;
        }

        [data-testid="stSidebar"] {
            background: #0B2545;
        }

        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: #F0F2F5;
            padding: 0.35rem;
            border-radius: 0.9rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0.7rem;
            padding: 0.55rem 1rem;
            font-weight: 600;
            color: #3D3D5C;
        }

        .stTabs [aria-selected="true"] {
            background: #FFFFFF;
            color: #0B2545;
            box-shadow: 0 1px 4px rgba(27, 27, 47, 0.08);
        }

        h1, h2, h3 {
            color: #0B2545 !important;
        }

        .app-banner {
            background: linear-gradient(135deg, #0B2545 0%, #134074 60%, #1B9AAA 100%);
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            margin-bottom: 1.2rem;
        }

        .app-banner h1 {
            color: #FFFFFF !important;
            margin: 0;
            font-size: 2rem;
        }

        .app-banner p {
            color: #AED9E0;
            margin-top: 0.4rem;
        }

        div[data-testid="metric-container"] {
            background: #F0F2F5;
            border-radius: 0.9rem;
            padding: 0.9rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# единая точка для стилей
def setup_page(title: str):
    st.set_page_config(
        page_title=title,
        layout="wide"
    )
    apply_global_styles()

# header

def render_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="app-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# сайдбар
def render_sidebar():
    with st.sidebar:
        st.markdown(" ")
        st.caption(" ")
        st.caption(" ")
        st.caption(" ")

        st.divider()

        st.markdown("### Ключевые метрики")
        st.metric("Заказов", "16 881")
        st.metric("Выкуп", "82.5%")
        st.metric("Невыкуп", "2 954")

        st.divider()

        st.caption("v1.0")