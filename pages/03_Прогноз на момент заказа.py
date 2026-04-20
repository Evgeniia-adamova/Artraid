import streamlit as st
import pandas as pd
from datetime import datetime

from ui import setup_page, render_header, render_sidebar
from ml_utils import predict_mk1, render_results
from src.data import load_clean_data

setup_page("Прогноз на момент заказа")
render_sidebar()

render_header(
    "Artraid Analytics",
    "Решение команды Ctrl + Alt + Win (CAW!)"
)

df = load_clean_data()

tab_predict, tab_features = st.tabs([" Прогноз выкупа", " Важность признаков"])

with tab_predict:
    st.markdown("### Параметры заказа")

    c1, c2, c3 = st.columns(3)
    with c1:
        price = st.slider("Стоимость заказа",
            int(df["lead_price"].min()),
            int(df["lead_price"].max()),
            int(df["lead_price"].median()), step=500)
        lead_source = st.selectbox("Источник клиента",
            sorted(df["lead_source_category"].dropna().unique()))
        region = st.selectbox("Регион",
            sorted(df["lead_region"].dropna().unique()))

    with c2:
        payment_type = st.selectbox("Вид оплаты",
            sorted(df["lead_Вид оплаты"].dropna().unique()))
        problem = st.selectbox("Проблема клиента",
            sorted(df["lead_Проблема"].dropna().unique()))
        delivery_service = st.selectbox("Служба доставки",
            sorted(df["lead_Служба доставки"].dropna().unique()))

    with c3:
        lead_quality = st.selectbox("Квалификация лида",
            sorted(df["lead_Квалификация лида"].dropna().unique()))
        category = st.selectbox("Оценка клиента по методологии DISC",
            sorted(df["lead_Категория и варианты выбора"].dropna().unique()))
        delivery_tariff = st.selectbox("Тариф доставки",
            sorted(df["lead_Тариф Доставки"].dropna().unique()))

    products = st.multiselect("Товары в заказе",
        ["Маска", "Наколенник", "Бандаж шейный", "Повязка",
         "Напульсник", "Обувь", "Подушка", "Матрас",
         "Постельное белье", "Пояс", "Аксессуары", "Крем", "Бады"])

    cb1, cb2, cb3, cb4 = st.columns(4)
    with cb1: is_repeat = st.checkbox("Повторный клиент")
    with cb2: is_company = st.checkbox("Юр. лицо")
    with cb3: has_promo = st.checkbox("Промокод")
    with cb4: has_discount = st.checkbox("Скидка")

    st.divider()
    predict_clicked = st.button("Прогнозировать",
        type="primary", use_container_width=True)

    def build_input():
        now = datetime.now()
        return {
            "lead_price": price,
            "lead_region": region,
            "lead_Служба доставки": delivery_service,
            "lead_Тариф Доставки": delivery_tariff,
            "lead_Вид оплаты": payment_type,
            "lead_Квалификация лида": lead_quality,
            "lead_Категория и варианты выбора": category,
            "lead_Проблема": problem,
            "lead_source_category": lead_source,
            "is_repeat_client": int(is_repeat),
            "is_yur": int(is_company),
            "has_promo": int(has_promo),
            "has_discount": int(has_discount),
            "has_yclid": 0,
            "has_маска": int("маска" in products),
            "has_наколенник": int("наколенник" in products),
            "has_бандаж_шейный": int("бандаж_шейный" in products),
            "has_повязка": int("повязка" in products),
            "has_напульсник": int("напульсник" in products),
            "has_обувь": int("обувь" in products),
            "has_подушка": int("подушка" in products),
            "has_матрас": int("матрас" in products),
            "has_постельное": int("постельное" in products),
            "has_пояс": int("пояс" in products),
            "has_аксессуары": int("аксессуары" in products),
            "has_крем": int("крем" in products),
            "has_бады": int("бады" in products),
            "n_product_categories": len(products),
            "sale_hour": now.hour,
            "sale_day_of_week": now.weekday(),
            "sale_month": now.month,
            "lead_created_hour": now.hour,
            "lead_created_day_of_week": now.weekday(),
            "lead_created_month": now.month,
            "days_creation_to_sale": 0,
        }

    if predict_clicked:
        input_data = build_input()
        with st.spinner("Считаем..."):
            probs = predict_mk1(input_data)
        render_results(probs)

with tab_features:
    st.markdown("### Важность признаков (MK1)")

    ftab1, ftab2, ftab3, ftab4 = st.tabs(["По группам", "CatBoost", "XGBoost (SHAP)", "Random Forest"])

    with ftab1:
        c1, c2 = st.columns(2)
        c1.image("ML/Images/CatBoost-MK1_grouped.png", caption="CatBoost", use_container_width=True)
        c2.image("ML/Images/XGBoost-MK1_grouped.png", caption="XGBoost", use_container_width=True)
        c3, c4 = st.columns(2)
        c3.image("ML/Images/LogReg-MK1_grouped.png", caption="LogReg", use_container_width=True)
        c4.image("ML/Images/RF-MK1_grouped.png", caption="Random Forest", use_container_width=True)

    with ftab2:
        c1, c2 = st.columns(2)
        c1.image("ML/Images/CatBoost-MK1_features.png", caption="Feature importance", use_container_width=True)
        c2.image("ML/Images/Permutation_importance_CatB-MK1_5000.png", caption="Permutation importance", use_container_width=True)

    with ftab3:
        c1, c2 = st.columns(2)
        c1.image("ML/Images/XGBoost-MK1_features.png", caption="Feature importance", use_container_width=True)
        c2.image("ML/Images/Summary_XGB-MK1_5000.png", caption="SHAP summary", use_container_width=True)
        st.image("ML/Images/Permutation_importance_XGB-MK1_5000.png", caption="Permutation importance", use_container_width=True)

    with ftab4:
        c1, c2 = st.columns(2)
        c1.image("ML/Images/RF-MK1_perimp.png", caption="Permutation importance", use_container_width=True)
        c2.image("ML/Images/LogReg-MK1_coef.png", caption="LogReg коэффициенты", use_container_width=True)
