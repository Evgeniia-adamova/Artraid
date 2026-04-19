import streamlit as st
from ui import setup_page, render_header, render_sidebar

setup_page("Artraid")

render_header(
    "Artraid Analytics",
    "Добро пожаловать в систему аналитики команды CAW!"
)

render_sidebar()

st.write("Выберите страницу слева 👈")