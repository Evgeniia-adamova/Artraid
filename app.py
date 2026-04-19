import streamlit as st
from ui import setup_page, render_header, render_sidebar
from src.data import load_clean_data

setup_page("Artraid")

render_header(
    "Artraid Analytics",
    "Добро пожаловать в систему аналитики команды CAW!"
)

render_sidebar()

df = load_clean_data()  # кэш сделает своё дело

st.write("Выберите страницу слева 👈")