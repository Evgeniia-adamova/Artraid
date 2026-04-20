import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ui import setup_page, render_header, render_sidebar
from src.data import load_clean_data

setup_page("Финансовые потери и риски")
render_sidebar()
render_header(
    "Artraid Analytics",
    "Решение команды Ctrl + Alt + Win (CAW!)"
)
# 0. Загрузка данных
df = load_clean_data()

st.markdown("## Потери от невыкупа и риски")

LOSS = "fin/loss_analysis_charts"
COHORT = "Cohort/outputs/charts"

(sub_logistics, sub_price, sub_payment, sub_sources) = st.tabs([
    " Логистика и регионы",
    " Стоимость заказа",
    " Способ оплаты",
    " Источники лидов",
])

# Sub-tab 1: Логистика и регионы
with sub_logistics:

    st.markdown("### 1.Потери по регионам")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Топ-10 регионов по общим потерям и % выкупа")
        st.image(f"{LOSS}/01_region_loss_01_total_without_title.png", use_container_width=True)
    with col2:
        st.markdown("#### Топ-10 регионов по проценту невыкупа")
        st.image(f"{LOSS}/01_region_loss_02_nonbuyout_without_title.png", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Невыкупы и средняя потеря по регионам")
        st.image(f"{LOSS}/01_region_loss_03_orders_loss_without_title.png", use_container_width=True)
    with col4:
        st.markdown("#### Распределение выкупа/невыкупа по регионам")
        st.image(f"{LOSS}/01_region_loss_04_share_without_title.png", use_container_width=True)

    st.markdown("### 2. Потери по длительности доставки")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Распределение заказов по длительности доставки")
        st.image(f"{LOSS}/02_delivery_group_orders_without_title.png", use_container_width=True)
    with col6:
        st.markdown("#### Потери от невыкупа по длительности доставки")
        st.image(f"{LOSS}/02_loss_analysis_by_delivery_group_without_title.png", use_container_width=True)

    st.markdown("### 3. Потери по службам доставки")

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("#### Потери от невыкупа по службам доставки")
        st.image(f"{LOSS}/06_loss_service_01_total_buyout.png", use_container_width=True)
    with col8:
        st.markdown("#### Количество невыкупленных заказов")
        st.image(f"{LOSS}/06_loss_service_02_nonbuyouts.png", use_container_width=True)

    col9, col10 = st.columns(2)
    with col9:
        st.markdown("#### Средняя потеря на невыкупленный заказ")
        st.image(f"{LOSS}/06_loss_service_03_avg_loss.png", use_container_width=True)
    with col10:
        st.markdown("#### Распределение выкупа/невыкупа по службам")
        st.image(f"{LOSS}/06_loss_service_04_share.png", use_container_width=True)

    st.markdown("### 4. Накопительный выкуп: сравнение региональных когорт")
    st.image(f"{LOSS}/01_regional_loss_5_curves_comparison_without_title.png", use_container_width=True)

# Sub-tab 2: Стоимость заказа
with sub_price:

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Потери от невыкупа по ценовым группам")
        st.image(f"{LOSS}/05_loss_price_01_total_buyout.png", use_container_width=True)
    with col2:
        st.markdown("#### Распределение заказов по ценовым группам")
        st.image(f"{LOSS}/05_loss_price_02_nonbuyouts.png", use_container_width=True)

# Sub-tab 3: Способ оплаты
with sub_payment:

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Потери от невыкупа по виду оплаты")
        st.image(f"{LOSS}/07_loss_payment_01_total_buyout.png", use_container_width=True)
    with col2:
        st.markdown("#### Количество невыкупленных заказов")
        st.image(f"{LOSS}/07_loss_payment_02_nonbuyouts.png", use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Средняя потеря на невыкупленный заказ")
        st.image(f"{LOSS}/07_loss_payment_03_avg_loss.png", use_container_width=True)
    with col4:
        st.markdown("#### Распределение выкупа/невыкупа по виду оплаты")
        st.image(f"{LOSS}/07_loss_payment_04_share.png", use_container_width=True)

# Sub-tab 4: Источники лидов

with sub_sources:

    st.markdown("#### Потери от невыкупа по источникам лидов")
    st.image(f"{LOSS}/04_lead_source_loss_01_total_without_title.png", use_container_width=True)
