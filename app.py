import streamlit as st

st.set_page_config(
    page_title="Artraid — Анализ выкупа",
    page_icon="📈 ",
    layout="wide"
)

st.title("Разработка решений для Артрейд. Аналитика выкупа (Data Analytics)")
st.caption("Период анализа: март 2025 – март 2026")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Заказов", "16 881")
col2.metric("Выкуп", "82.5%")
col3.metric("Невыкуп", "2 954", "≈ X млн ₽ потерь")
col4.metric("Медиана доставки", "3.2 дня")
col5.metric("Средний чек", "8 450 ₽")
""
