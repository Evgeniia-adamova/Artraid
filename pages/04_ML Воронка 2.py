import streamlit as st
import pandas as pd
from datetime import datetime

from ui import setup_page, render_header, render_sidebar
from ml_utils import predict_mk2, render_results

@st.cache_data
def load_data():
    return pd.read_excel("data_preparation/data/clean/clean_data.xlsx")

setup_page("ML Воронка 2")
render_sidebar()

render_header(
    "Artraid Analytics",
    "ML-прогнозирование вероятности выкупа на момент передачи в доставку"
)

df = load_data()

st.info("Воронка 2 использует модели MK2, обученные с учётом данных логистики, "
        "которые становятся известны в момент передачи заказа в доставку.")

st.markdown("### Параметры заказа")

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
        ["маска", "наколенник", "бандаж_шейный", "повязка",
         "напульсник", "обувь", "подушка", "матрас",
         "постельное", "пояс", "аксессуары", "крем", "бады"]
    )
    has_promo = st.checkbox("Есть промокод")
    has_discount = st.checkbox("Есть скидка")


st.markdown("#### Логистика")
col5, col6 = st.columns(2)

with col5:
    delivery_service = st.selectbox(
        "Служба доставки",
        sorted(df["lead_Служба доставки"].dropna().unique())
    )
    delivery_tariff = st.selectbox(
        "Тариф доставки",
        sorted(df["lead_Тариф Доставки"].dropna().unique())
    )

with col6:
    delivery_group = st.selectbox(
        "Скорость доставки (группа)",
        sorted(df["delivery_group"].dropna().unique()) if "delivery_group" in df.columns else ["быстрая", "средняя", "долгая", "очень долгая"]
    )
    days_sale_to_handed = st.slider(
        "Дней на сборку заказа",
        min_value=0, max_value=30, value=3
    )
    days_handed_to_issued_pvz = st.slider(
        "Дней доставки до выдачи",
        min_value=0, max_value=30, value=5
    )

payment_type = st.selectbox(
    "Вид оплаты",
    sorted(df["lead_Вид оплаты"].dropna().unique())
)

st.divider()

predict_clicked = st.button("Прогнозировать", use_container_width=True)


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
        "delivery_group": delivery_group,
        "days_sale_to_handed": days_sale_to_handed,
        "days_handed_to_issued_pvz": days_handed_to_issued_pvz,
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
        probs = predict_mk2(input_data)
    render_results(probs)
