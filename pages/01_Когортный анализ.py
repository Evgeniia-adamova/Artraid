import streamlit as st
import pandas as pd

from ui import setup_page, render_header, render_sidebar

setup_page("Когортный анализ")
render_sidebar()
render_header(
    "Artraid Analytics",
    "Решение команды Сtrl+Alt+Win"
)

@st.cache_data
def load_data():
    return pd.read_excel("data_preparation/data/clean/clean_data.xlsx")

df = load_data()

st.markdown("## Когортный анализ выкупа")

CHARTS = "Cohort/outputs/charts"

tab1, tab2, tab3 = st.tabs([
    " Динамика выкупа и продаж",
    " Выкуп по месяцам и источникам",
    " Связь скорости доставки и выкупа по неделям",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Размеры недельных когорт")
        st.image(f"{CHARTS}/cohort_sizes_weekly_without_title.png", use_container_width=True)
    with col2:
        st.markdown("#### Недельные когорты — финальный buyout rate")
        st.image(f"{CHARTS}/final_buyout_rate_weekly_without_title.png", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Динамика накопительного выкупа")
        st.image(f"{CHARTS}/cohort_curves_weekly_selected_without_title.png", use_container_width=True)
    with col4:
        st.markdown("#### Месячные когорты — финальный buyout rate")
        st.image(f"{CHARTS}/final_buyout_rate_monthly_without_title.png", use_container_width=True)

with tab2:
    st.markdown("#### Выкуп по месяцам и источникам")
    st.image(f"{CHARTS}/source_month_heatmap_without_title.png", use_container_width=True)

with tab3:
    st.markdown("#### Недельные когорты — финальный buyout rate")
    st.image(f"{CHARTS}/final_buyout_rate_weekly_without_title.png", use_container_width=True)

    st.markdown("#### Дни от продажи до отправления по неделям")
    st.image(f"{CHARTS}/03_delivery_timing_without_title.png", use_container_width=True)