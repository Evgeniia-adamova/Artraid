#!/usr/bin/env python
# coding: utf-8


import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import os
from pathlib import Path
from PIL import Image
import io

warnings.filterwarnings('ignore')

# Палитра Артрейд
# ──────────────────────────────────────────────
DARK       = "#1B1B2F"
TEXT       = "#3D3D5C"
GRAY       = "#8E8EA0"
LIGHT      = "#F0F2F5"
BG         = "#FFFFFF"
RED        = "#C0392B"
ORANGE     = "#E67E22"

GRADIENT_POOL = ["#0B2545", "#134074", "#13678A", "#1B9AAA", "#45B7D1", "#73C2D4", "#AED9E0"]

ARTRAID_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "artraid", ["#FFFFFF", "#AED9E0", "#45B7D1", "#1B9AAA", "#13678A", "#134074", "#0B2545"]
)

LOSS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "loss",
    ["#F5B7B1", "#F1948A", "#E74C3C", "#C0392B", "#922B21"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
})

# # Анализ финансовых потерь от невыкупа заказов
# 
# Этот блокнот анализирует потери от невыкупленных заказов по различным группирующим признакам:
# - lead_region (регион лида)
# - price_group (ценовая группа)
# - delivery_group (группа доставки)
# 
# Цель: определить основные источники потерь для финансового анализа

# Создаем подпапку для результатов анализа
results_folder = './loss_analysis_charts'

Path(results_folder).mkdir(exist_ok=True)

print(f" Подпапка для результатов создана: {results_folder}")
print(f"✓ Все результаты будут сохранены в этой папке")

USE_COLS = [
    "buyout_flag",
    "lead_id",
    "days_handed_to_issued_pvz",
    "lead_price",
    "lead_region",
    "price_group",
    "delivery_group",
    "lead_Тариф Доставки",
    "lead_Вид оплаты",
    "lead_Служба доставки",
    "has_маска",
    "has_наколенник",
    "has_бандаж_шейный",
    "has_повязка",
    "has_напульсник",
    "has_обувь",
    "has_подушка",
    "has_матрас",
    "has_постельное",
    "has_пояс",
    "has_аксессуары",
    "has_крем",
    "has_бады"
]
# Фильтры на статистически бесполезные данные
MIN_SOURCE_COUNT = 500
MIN_PAYMENT_METHOD_COUNT = 100
MIN_DELIVERY_SERVICE_COUNT = 100

# 1. Загрузим данные
file_path = '../data_preparation/data/clean/clean_data.xlsx'
df = pd.read_excel(file_path, usecols=USE_COLS)

# Проверим наличие boolean колонок с товарами
product_columns = ['has_маска', 'has_наколенник', 'has_бандаж_шейный',
                   'has_повязка', 'has_напульсник', 'has_обувь',
                   'has_подушка', 'has_матрас', 'has_постельное',
                   'has_пояс', 'has_аксессуары', 'has_крем', 'has_бады']

# Найдём какие колонки существуют в данных
existing_cols = [col for col in product_columns if col in df.columns]
missing_cols = [col for col in product_columns if col not in df.columns]

# 2. Подготовка данных и расчет потерь

# Создаем функцию для фильтрации групп, в которых недостаточно данных для статистического анализа
def clean_categorical(df):
    df = df.copy()

    # 1. Фильтруем по службе доставки
    if 'lead_Служба доставки' in df.columns:
        counts = df['lead_Служба доставки'].value_counts()
        valid = counts[counts >= MIN_DELIVERY_SERVICE_COUNT].index
        df = df[df['lead_Служба доставки'].isin(valid)]

    # 2. Фильтруем по виду оплаты
    if 'lead_Вид оплаты' in df.columns:
        counts = df['lead_Вид оплаты'].value_counts()
        valid = counts[counts >= MIN_PAYMENT_METHOD_COUNT].index
        df = df[df['lead_Вид оплаты'].isin(valid)]

    # 3. Источники — порог по частоте
    if 'lead_source_category' in df.columns:
        counts = df['lead_source_category'].value_counts()
        valid = counts[counts >= MIN_SOURCE_COUNT].index
        df = df[df['lead_source_category'].isin(valid)]

    return df

df = clean_categorical(df)

# Определим потери: для невыкупленных заказов (buyout_flag = 0 или False) потеря = lead_price
# Создадим новую колонку для потерь
df['loss'] = 0.0
df.loc[df['buyout_flag'] == 0, 'loss'] = df.loc[df['buyout_flag'] == 0, 'lead_price']

# Альтернативный способ (на случай если используются булевы значения)
if df['buyout_flag'].dtype == bool:
    df.loc[~df['buyout_flag'], 'loss'] = df.loc[~df['buyout_flag'], 'lead_price']

# 3. Создаем функцию для подсчета потерь по различным факторам

def calculate_loss_metrics(df, group_col):
    # Основная агрегация
    result = df.groupby(group_col).agg({
        'loss': ['sum', 'mean', 'count'],
        'lead_price': 'sum'
    }).round(2)

    result.columns = [
        'Total Loss',
        'Avg Loss per Order',
        'Non-buyout Orders',
        'Total Order Value'
    ]

    # Считаем количество выкупов
    buyout_counts = df.groupby(group_col)['buyout_flag'].apply(lambda x: (x == 1).sum())

    # Общее количество заказов
    total_counts = df.groupby(group_col).size()

    # Добавляем процент выкупа
    result['Buyout Rate %'] = (buyout_counts / total_counts * 100).round(2)

    # Сортировка
    result = result.sort_values('Total Loss', ascending=False)

    return result

# 4. Анализ потерь по регионам (lead_region)
loss_by_region = calculate_loss_metrics(df, 'lead_region')
# 5. Анализ потерь по ценовым группам (price_group)
loss_by_price_group = calculate_loss_metrics(df, 'price_group')
# 6. Анализ потерь по группам доставки (delivery_group)
loss_by_delivery_group = calculate_loss_metrics(df, 'delivery_group')

# 7. Создаем функцию для многомерного анализа потерь по нескольким факторам.
def multi_dimensional_analysis(df, group_cols):
    result = df.groupby(group_cols).agg({
        'loss': ['sum', lambda x: (x > 0).sum()]
    }).round(2)

    result.columns = ['Total Loss', 'Non-buyout Orders']
    result = result.sort_values('Total Loss', ascending=False)

    pivot = pd.pivot_table(
        df,
        values='loss',
        index=group_cols[0],
        columns=group_cols[1],
        aggfunc='sum'
    ).fillna(0)

    return result, pivot

# 8. Многомерный анализ: Регион + Ценовая группа
loss_region_price, pivot_region_price = multi_dimensional_analysis(df, ['lead_region', 'price_group'])
# 9. Многомерный анализ: Регион + Группа доставки
loss_region_delivery, pivot_region_delivery = multi_dimensional_analysis(df, ['lead_region', 'delivery_group'])
# 10. Многомерный анализ: Ценовая группа + Группа доставки
loss_price_delivery, pivot_price_delivery = multi_dimensional_analysis(df, ['lead_region', 'price_group'])
# 11. Многомерный анализ: Регион + служба доставки,
# агрегация по скорости доставки (медиана), количеству заказов (количество) и потерям (сумма)
pivot_delivery = pd.pivot_table(
    df,
    values=['days_handed_to_issued_pvz', 'lead_id', 'loss'],
    index='lead_region',
    columns='lead_Служба доставки',
    aggfunc={
        'days_handed_to_issued_pvz': 'median',
        'lead_id': 'count',
        'loss': 'sum'
    },
).round(1)

# 12. Анализ потерь по товарам (boolean категории)
# Список товаров
product_columns = ['маска', 'наколенник', 'бандаж_шейный', 
                   'повязка', 'напульсник', 'обувь', 
                   'подушка', 'матрас', 'постельное', 
                   'пояс', 'аксессуары', 'крем', 'бады']

# Распунстируем данные: для каждого товара создаём отдельные строки
product_rows = []

for idx, row in df.iterrows():
    order_loss = row['loss']
    order_buyout = row['buyout_flag']
    lead_price = row['lead_price']

    # Для каждого товара проверяем его наличие в заказе
    for product in product_columns:
        col_name = f'has_{product}'
        if row[col_name]:  # Если товар есть в заказе
            product_rows.append({
                'product': product,
                'loss': order_loss,
                'buyout_flag': order_buyout,
                'lead_price': lead_price
            })

df_products = pd.DataFrame(product_rows)

# Анализируем потери по товарам
loss_by_product = df_products.groupby('product').agg({
    'loss': ['sum', 'mean'],
    'buyout_flag': lambda x: (x == 1).sum(),
    'lead_price': 'count'
}).round(2)

loss_by_product.columns = ['Total Loss', 'Avg Loss per Item', 'Buyout Count', 'Total Items']
loss_by_product['Non-buyout Count'] = loss_by_product['Total Items'] - loss_by_product['Buyout Count']
loss_by_product['Buyout Rate %'] = (loss_by_product['Buyout Count'] / loss_by_product['Total Items'] * 100).round(2)
loss_by_product['Loss per Non-buyout Item'] = (loss_by_product['Total Loss'] / loss_by_product['Non-buyout Count']).round(2)

loss_by_product = loss_by_product.sort_values('Total Loss', ascending=False)

# 13. Анализ потерь по виду оплаты (lead_Вид оплаты)
loss_by_payment_type = calculate_loss_metrics(df, 'lead_Вид оплаты')

# 14. Анализ потерь по службе доставки
loss_by_delivery_service = calculate_loss_metrics(df, 'lead_Служба доставки')

#==============================================================================
#                                ВИЗУАЛИЗАЦИИ
#==============================================================================
# 1. Потери по регионам

top_regions = loss_by_region.head(10).copy()
top_regions["Non-buyout %"] = 100 - top_regions["Buyout Rate %"]

# 1) Потери + % выкупа
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1_twin = ax1.twiny()

ax1.barh(
    range(len(top_regions)),
    top_regions["Total Loss"],
    color=GRADIENT_POOL[1],
    alpha=0.85
)
ax1_twin.plot(
    top_regions["Buyout Rate %"],
    range(len(top_regions)),
    color=RED,
    marker="o",
    linewidth=2.5,
    markersize=7
)

ax1.set_yticks(range(len(top_regions)))
ax1.set_yticklabels(top_regions.index)
ax1.set_xlabel("Общие потери", fontsize=11)
ax1.set_ylabel("Регион", fontsize=11)
ax1_twin.set_xlabel("% Выкупа", fontsize=11, color=RED)
ax1_twin.tick_params(axis="x", colors=RED)
ax1.set_title(" ") # убрала title для ручного контроля в streamlit
ax1.invert_yaxis()
ax1.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{results_folder}/01_region_loss_01_total_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2) % невыкупа
fig, ax2 = plt.subplots(figsize=(12, 7))
ax2.barh(
    range(len(top_regions)),
    top_regions["Non-buyout %"],
    color=ORANGE,
    alpha=0.85
)
ax2.set_yticks(range(len(top_regions)))
ax2.set_yticklabels(top_regions.index)
ax2.set_xlabel("% Невыкупа", fontsize=11)
ax2.set_title(" ") # убрала title для ручного контроля в streamlit
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{results_folder}/01_region_loss_02_nonbuyout_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 3) Невыкупы + средняя потеря
fig, ax3 = plt.subplots(figsize=(12, 7))
ax3_twin = ax3.twinx()

ax3.bar(
    range(len(top_regions)),
    top_regions["Non-buyout Orders"],
    color=GRADIENT_POOL[4],
    alpha=0.85
)
ax3_twin.plot(
    range(len(top_regions)),
    top_regions["Avg Loss per Order"],
    color=RED,
    marker="^",
    linestyle="--",
    linewidth=2.5,
    markersize=7
)

ax3.set_xticks(range(len(top_regions)))
ax3.set_xticklabels(top_regions.index, rotation=45, ha="right")
ax3.set_ylabel("Количество невыкупленных заказов", fontsize=11)
ax3_twin.set_ylabel("Средняя потеря на заказ", fontsize=11, color=RED)
ax3_twin.tick_params(axis="y", colors=RED)
ax3.set_title(" ") # убрала title для ручного контроля в streamlit
ax3.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{results_folder}/01_region_loss_03_orders_loss_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 4) Доля выкупа/невыкупа
fig, ax4 = plt.subplots(figsize=(12, 7))
x = np.arange(len(top_regions))

ax4.bar(x, top_regions["Buyout Rate %"], label="% Выкупа", color=ORANGE, alpha=0.85)
ax4.bar(
    x,
    top_regions["Non-buyout %"],
    bottom=top_regions["Buyout Rate %"],
    label="% Невыкупа",
    color=RED,
    alpha=0.85
)

ax4.set_xticks(x)
ax4.set_xticklabels(top_regions.index, rotation=45, ha="right")
ax4.set_ylabel("Процент (%)", fontsize=11)
ax4.set_title(" ") # убрала title для ручного контроля в streamlit
ax4.legend(fontsize=10)
ax4.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{results_folder}/01_region_loss_04_share_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

#  2. Потери по скорости доставки
loss_by_delivery_group_sorted = loss_by_delivery_group.sort_values("Total Loss", ascending=False).copy()
delivery_order_counts = df.groupby("delivery_group").size().sort_values(ascending=False)

label_map = {
    "быстрая": "Быстрая (0-3 дня)",
    "средняя": "Средняя (4-5 дней)",
    "долгая": "Долгая (6-10 дней)",
    "очень долгая": "Очень долгая (10+ дней)",
}
labels = [label_map.get(x, x) for x in loss_by_delivery_group_sorted.index]

bar_colors = ARTRAID_CMAP(np.linspace(0.95, 0.35, len(loss_by_delivery_group_sorted)))

fig, ax2 = plt.subplots(figsize=(12, 7))
ax2_twin = ax2.twinx()

ax2.bar(
    range(len(loss_by_delivery_group_sorted)),
    loss_by_delivery_group_sorted["Total Loss"],
    color=bar_colors,
    alpha=0.98
)
ax2_twin.plot(
    range(len(loss_by_delivery_group_sorted)),
    loss_by_delivery_group_sorted["Buyout Rate %"],
    color=RED,
    marker="o",
    linewidth=2.4,
    markersize=7
)

ax2.set_xticks(range(len(loss_by_delivery_group_sorted)))
ax2.set_xticklabels(labels, rotation=45, ha="right")
ax2.set_ylabel("Общие потери", fontsize=11)
ax2_twin.set_ylabel("% Выкупа", fontsize=11, color=RED)
ax2_twin.tick_params(axis="y", colors=RED)
ax2.set_title("")
ax2.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(f"{results_folder}/02_loss_analysis_by_delivery_group_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# График 2: Кол-во заказов по группам доставки
fig, ax4 = plt.subplots(figsize=(12, 7))
ax4.bar(
    range(len(delivery_order_counts)),
    delivery_order_counts,
    color=ORANGE,
    alpha=0.88
)
ax4.set_xticks(range(len(delivery_order_counts)))
ax4.set_xticklabels(labels, rotation=45, ha="right")
ax4.set_ylabel("Количество заказов", fontsize=11)
ax4.set_title(" ")
ax4.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(f"{results_folder}/02_delivery_group_orders_without_title.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("✓ Сохранены 2 отдельных файла:")
print(f"- {results_folder}/02_loss_analysis_by_delivery_group.png")
print(f"- {results_folder}/02_delivery_group_orders.png")

# 3. Потери по службам доставки

try:
    loss_by_delivery_service_sorted = loss_by_delivery_service.sort_values('Total Loss', ascending=False).copy()

    loss_colors = ARTRAID_CMAP(np.linspace(0.95, 0.35, len(loss_by_delivery_service_sorted)))
    avg_loss_sorted = loss_by_delivery_service.sort_values('Avg Loss per Order', ascending=False).copy()
    avg_colors = LOSS_CMAP(np.linspace(0.95, 0.35, len(avg_loss_sorted)))

    # 1) Потери по Службе доставки + % выкупа
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1_twin = ax1.twinx()

    ax1.barh(
        range(len(loss_by_delivery_service_sorted)),
        loss_by_delivery_service_sorted['Total Loss'],
        color=loss_colors,
        alpha=0.98
    )
    ax1_twin.plot(
        loss_by_delivery_service_sorted['Buyout Rate %'],
        range(len(loss_by_delivery_service_sorted)),
        color=RED,
        marker='o',
        linewidth=2.4,
        markersize=7
    )

    ax1.set_yticks(range(len(loss_by_delivery_service_sorted)))
    ax1.set_yticklabels(loss_by_delivery_service_sorted.index)
    ax1.set_xlabel('Общие потери', fontsize=11)
    ax1.set_ylabel('Служба доставки', fontsize=11)
    ax1_twin.set_xlabel('% Выкупа', fontsize=11, color=RED)
    ax1_twin.set_ylim(70, 100)
    ax1_twin.tick_params(axis='x', colors=RED)
    ax1.set_title('')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/06_loss_service_01_total_buyout.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2) Количество невыкупов по Службе доставки
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.barh(
        range(len(loss_by_delivery_service_sorted)),
        loss_by_delivery_service_sorted['Non-buyout Orders'],
        color=ORANGE,
        alpha=0.98
    )
    ax2.set_yticks(range(len(loss_by_delivery_service_sorted)))
    ax2.set_yticklabels(loss_by_delivery_service_sorted.index)
    ax2.set_xlabel('Количество невыкупов', fontsize=11)
    ax2.set_ylabel('Служба доставки', fontsize=11)
    ax2.set_title('')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/06_loss_service_02_nonbuyouts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3) Средняя потеря
    fig, ax3 = plt.subplots(figsize=(12, 7))
    ax3.barh(
        range(len(avg_loss_sorted)),
        avg_loss_sorted['Avg Loss per Order'],
        color=avg_colors,
        alpha=0.98
    )
    ax3.set_yticks(range(len(avg_loss_sorted)))
    ax3.set_yticklabels(avg_loss_sorted.index)
    ax3.set_xlabel('Средняя потеря на заказ', fontsize=11)
    ax3.set_ylabel('Служба доставки', fontsize=11)
    ax3.set_title('')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/06_loss_service_03_avg_loss.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4) Распределение выкупа/невыкупа
    fig, ax4 = plt.subplots(figsize=(12, 7))
    non_buyout_pct = 100 - loss_by_delivery_service_sorted['Buyout Rate %']

    ax4.barh(
        range(len(loss_by_delivery_service_sorted)),
        loss_by_delivery_service_sorted['Buyout Rate %'],
        label='% Выкупа',
        color=GRADIENT_POOL[4],
        alpha=0.92
    )
    ax4.barh(
        range(len(loss_by_delivery_service_sorted)),
        non_buyout_pct,
        left=loss_by_delivery_service_sorted['Buyout Rate %'],
        label='% Невыкупа',
        color=RED,
        alpha=0.85
    )

    ax4.set_yticks(range(len(loss_by_delivery_service_sorted)))
    ax4.set_yticklabels(loss_by_delivery_service_sorted.index)
    ax4.set_xlabel('Процент (%)', fontsize=11)
    ax4.set_title('')
    ax4.legend(fontsize=10, loc='lower right')
    ax4.grid(axis='x', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/06_loss_service_04_share.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('✓ Графики сохранены:')
    print(f'- {results_folder}/06_loss_service_01_total_buyout.png')
    print(f'- {results_folder}/06_loss_service_02_nonbuyouts.png')
    print(f'- {results_folder}/06_loss_service_03_avg_loss.png')
    print(f'- {results_folder}/06_loss_service_04_share.png')

except Exception as e:
    print(f"⚠️ Ошибка при создании графиков для Службы доставки: {e}")


# 4. Потери по видам оплаты
try:
    loss_by_payment_type_sorted = loss_by_payment_type.sort_values('Total Loss', ascending=False).copy()

    norm_loss = plt.Normalize(
        loss_by_payment_type_sorted['Total Loss'].min(),
        loss_by_payment_type_sorted['Total Loss'].max()
    )
    norm_nonbuy = plt.Normalize(
        loss_by_payment_type_sorted['Non-buyout Orders'].min(),
        loss_by_payment_type_sorted['Non-buyout Orders'].max()
    )
    avg_sorted = loss_by_payment_type.sort_values('Avg Loss per Order', ascending=False).copy()
    norm_avg = plt.Normalize(
        avg_sorted['Avg Loss per Order'].min(),
        avg_sorted['Avg Loss per Order'].max()
    )

    loss_colors = ARTRAID_CMAP(norm_loss(loss_by_payment_type_sorted['Total Loss'].values))
    nonbuy_colors = ARTRAID_CMAP(norm_nonbuy(loss_by_payment_type_sorted['Non-buyout Orders'].values))
    avg_colors = ARTRAID_CMAP(norm_avg(avg_sorted['Avg Loss per Order'].values))

    # 1) Потери по Виду оплаты + % выкупа
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1_twin = ax1.twinx()

    ax1.bar(
        range(len(loss_by_payment_type_sorted)),
        loss_by_payment_type_sorted['Total Loss'],
        color=GRADIENT_POOL[1],
        alpha=0.98,
        edgecolor=DARK,
        linewidth=0.5
    )
    ax1_twin.plot(
        range(len(loss_by_payment_type_sorted)),
        loss_by_payment_type_sorted['Buyout Rate %'],
        color=RED,
        marker='o',
        linewidth=2.4,
        markersize=7
    )

    ax1_twin.set_ylim(75, 100)
    ax1.set_xticks(range(len(loss_by_payment_type_sorted)))
    ax1.set_xticklabels(loss_by_payment_type_sorted.index, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('Общие потери', fontsize=12)
    ax1_twin.set_ylabel('% Выкупа', fontsize=12, color=RED)
    ax1_twin.tick_params(axis='x', colors=RED)
    ax1.set_title(" ")
    ax1.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/07_loss_payment_01_total_buyout.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2) Количество невыкупов
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.bar(
        range(len(loss_by_payment_type_sorted)),
        loss_by_payment_type_sorted['Non-buyout Orders'],
        color=ORANGE,
        alpha=0.98,
        edgecolor='none',
        linewidth=0
    )
    ax2.set_xticks(range(len(loss_by_payment_type_sorted)))
    ax2.set_xticklabels(loss_by_payment_type_sorted.index, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Количество невыкупов', fontsize=12)
    ax2.set_title(" ")
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/07_loss_payment_02_nonbuyouts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3) Средняя потеря
    bar_colors = ARTRAID_CMAP(np.linspace(0.95, 0.35, len(avg_sorted)))

    fig, ax3 = plt.subplots(figsize=(12, 7))
    ax3.bar(
        range(len(avg_sorted)),
        avg_sorted['Avg Loss per Order'],
        color=bar_colors,
        alpha=0.98,
        edgecolor='none',
        linewidth=0
    )
    ax3.set_xticks(range(len(avg_sorted)))
    ax3.set_xticklabels(avg_sorted.index, rotation=45, ha='right', fontsize=11)
    ax3.set_ylabel('Средняя потеря на заказ', fontsize=12)
    ax3.set_title(" ")
    ax3.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/07_loss_payment_03_avg_loss.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4) Распределение выкупа/невыкупа
    fig, ax4 = plt.subplots(figsize=(12, 7))
    non_buyout_pct = 100 - loss_by_payment_type_sorted['Buyout Rate %']

    ax4.bar(
        range(len(loss_by_payment_type_sorted)),
        loss_by_payment_type_sorted['Buyout Rate %'],
        label='% Выкупа',
        color=ORANGE,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )
    ax4.bar(
        range(len(loss_by_payment_type_sorted)),
        non_buyout_pct,
        bottom=loss_by_payment_type_sorted['Buyout Rate %'],
        label='% Невыкупа',
        color=RED,
        alpha=0.85,
        edgecolor='none',
        linewidth=0
    )

    ax4.set_xticks(range(len(loss_by_payment_type_sorted)))
    ax4.set_xticklabels(loss_by_payment_type_sorted.index, rotation=45, ha='right', fontsize=11)
    ax4.set_ylabel('Процент (%)', fontsize=12)
    ax4.set_title(" ")
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/07_loss_payment_04_share.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('✓ Графики сохранены:')
    print(f'- {results_folder}/07_loss_payment_01_total_buyout.png')
    print(f'- {results_folder}/07_loss_payment_02_nonbuyouts.png')
    print(f'- {results_folder}/07_loss_payment_03_avg_loss.png')
    print(f'- {results_folder}/07_loss_payment_04_share.png')

except Exception as e:
    print(f"⚠️ Ошибка при создании графиков для Вида оплаты: {e}")

# 5. Потери по ценовым группам
# потери по ценовым группам+% выкупа
# 5. Потери по ценовым группам

try:
    loss_by_price_group_sorted = loss_by_price_group.sort_values('Total Loss', ascending=False).copy()

    norm_loss = plt.Normalize(
        loss_by_price_group_sorted['Total Loss'].min(),
        loss_by_price_group_sorted['Total Loss'].max()
    )
    norm_nonbuy = plt.Normalize(
        loss_by_price_group_sorted['Non-buyout Orders'].min(),
        loss_by_price_group_sorted['Non-buyout Orders'].max()
    )

    loss_colors = ARTRAID_CMAP(norm_loss(loss_by_price_group_sorted['Total Loss'].values))
    nonbuy_colors = ARTRAID_CMAP(norm_nonbuy(loss_by_price_group_sorted['Non-buyout Orders'].values))

    # 1) Потери по ценовым группам + % выкупа
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1_twin = ax1.twinx()
    bar_colors = ARTRAID_CMAP(np.linspace(0.70, 0.25, len(loss_by_price_group_sorted)))

    ax1.bar(
        range(len(loss_by_price_group_sorted)),
        loss_by_price_group_sorted['Total Loss'],
        color=bar_colors,
        alpha=0.98,
        edgecolor='none',
        linewidth=0
    )
    ax1_twin.plot(
        range(len(loss_by_price_group_sorted)),
        loss_by_price_group_sorted['Buyout Rate %'],
        color=RED,
        marker='o',
        linewidth=2.4,
        markersize=7
    )

    ax1_twin.set_ylim(75, 100)
    ax1.set_xticks(range(len(loss_by_price_group_sorted)))
    ax1.set_xticklabels(loss_by_price_group_sorted.index, rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('Общие потери', fontsize=12)
    ax1_twin.set_ylabel('% Выкупа', fontsize=12, color=RED)
    ax1_twin.tick_params(axis='y', colors=RED)
    ax1.set_title('')
    ax1.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/05_loss_price_01_total_buyout.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2) Количество невыкупов
    fig, ax2 = plt.subplots(figsize=(12, 7))
    ax2.bar(
        range(len(loss_by_price_group_sorted)),
        loss_by_price_group_sorted['Non-buyout Orders'],
        color=GRADIENT_POOL[4],
        alpha=0.98,
        edgecolor='none',
        linewidth=0
    )
    ax2.set_xticks(range(len(loss_by_price_group_sorted)))
    ax2.set_xticklabels(loss_by_price_group_sorted.index, rotation=45, ha='right', fontsize=11)
    ax2.set_ylabel('Количество невыкупов', fontsize=12)
    ax2.set_title('')
    ax2.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{results_folder}/05_loss_price_02_nonbuyouts.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('✓ Графики сохранены:')
    print(f'- {results_folder}/05_loss_price_01_total_buyout.png')
    print(f'- {results_folder}/05_loss_price_02_nonbuyouts.png')

except Exception as e:
    print(f"⚠️ Ошибка при создании графиков для ценовых групп: {e}")