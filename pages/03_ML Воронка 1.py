import streamlit as st
import pandas as pd
from datetime import datetime

from ui import setup_page, render_header, render_sidebar

@st.cache_data
def load_data():
    return pd.read_excel("data_preparation/data/clean/clean_data.xlsx")

df = load_data()

setup_page("ML Воронка 1")
render_sidebar()

render_header(
    "Artraid Analytics",
    "ML-прогнозирование вероятности выкупа на момент заказа"
)

df = load_data()

# форма
st.markdown("### Параметры заказа")

# первый блок: клиент
st.markdown("#### Клиент")

col1, col2 = st.columns(2)

with col1:
    lead_source = st.selectbox(
        "Источник клиента",
        sorted(df["lead_source_category"].dropna().unique())
    )

    lead_quality = st.selectbox(
        "Квалификация лида",
        sorted(df["lead_Квалификация лида"].dropna().unique())
    )

    region = st.selectbox(
        "Регион",
        sorted(df["lead_region"].dropna().unique())
    )

with col2:
    is_repeat = st.checkbox("Повторный клиент")
    is_company = st.checkbox("Юридическое лицо")


# заказ
st.markdown("#### Заказ")

col3, col4 = st.columns(2)

with col3:
    price = st.slider(
        "Стоимость заказа",
        int(df["lead_price"].min()),
        int(df["lead_price"].max()),
        int(df["lead_price"].median()),
        step=500
    )

    category = st.selectbox(
        "Категория заказа",
        sorted(df["lead_Категория и варианты выбора"].dropna().unique())
    )

    problem = st.selectbox(
        "Проблема клиента",
        sorted(df["lead_Проблема"].dropna().unique())
    )

with col4:
    products = st.multiselect(
        "Товары в заказе",
        [
            "маска", "наколенник", "бандаж_шейный", "повязка",
            "напульсник", "обувь", "подушка", "матрас",
            "постельное", "пояс", "аксессуары", "крем", "бады"
        ]
    )

    has_promo = st.checkbox("Есть промокод")
    has_discount = st.checkbox("Есть скидка")


# --- Блок 3: Логистика
st.markdown("#### Логистика")

col5, col6 = st.columns(2)

with col5:
    delivery_service = st.selectbox(
        "Служба доставки",
        sorted(df["lead_Служба доставки"].dropna().unique())
    )

with col6:
    delivery_tariff = st.selectbox(
        "Тариф доставки",
        sorted(df["lead_Тариф Доставки"].dropna().unique())
    )

payment_type = st.selectbox(
    "Вид оплаты",
    sorted(df["lead_Вид оплаты"].dropna().unique())
)


st.divider()

# кнопка
predict_clicked = st.button("Прогнозировать", use_container_width=True)

# PREPARE INPUT (КОНТРАКТ)
def build_input():
    now = datetime.now()

    data = {
        # базовые
        "lead_price": price,
        "lead_region": region,
        "lead_Служба доставки": delivery_service,
        "lead_Тариф Доставки": delivery_tariff,
        "lead_Вид оплаты": payment_type,
        "lead_Квалификация лида": lead_quality,
        "lead_Категория и варианты выбора": category,
        "lead_Проблема": problem,
        "lead_source_category": lead_source,

        # флаги
        "is_repeat_client": int(is_repeat),
        "is_yur": int(is_company),
        "has_promo": int(has_promo),
        "has_discount": int(has_discount),
        "has_yclid": 0,

        # товары → бинарные
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

        # агрегаты
        "n_product_categories": len(products),

        # время
        "sale_hour": now.hour,
        "sale_day_of_week": now.weekday(),
        "sale_month": now.month,
        "lead_created_hour": now.hour,
        "lead_created_day_of_week": now.weekday(),
        "lead_created_month": now.month,
        "days_creation_to_sale": 0,
    }

    return data

# результат (заглушка)
if predict_clicked:

    input_data = build_input()

    # result = predict(input_data)

    st.markdown("### Результаты моделей")

    mock = pd.DataFrame({
        "Модель": ["LogReg", "RF", "XGB", "CatBoost"],
        "Вероятность": ["72%", "69%", "71%", "75%"],
        "Вердикт": ["Средний риск"] * 4
    })

    st.table(mock)

    st.markdown("### Итог")

    st.metric("Средняя вероятность", "71.8%")
    st.progress(0.718)
    st.warning("Есть риск невыкупа")