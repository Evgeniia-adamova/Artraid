from pathlib import Path
import streamlit as st
import pandas as pd

_DATA_FILE = Path("data_preparation/data/clean/clean_data.xlsx")

@st.cache_data
def load_clean_data():
    if not _DATA_FILE.exists():
        st.error(f"Файл не найден: {_DATA_FILE}")
        st.stop()
    return pd.read_excel(_DATA_FILE)