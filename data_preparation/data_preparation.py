# чистый скрипт для подготовки данных
import pandas as pd

#запуск из корневой папки
df = pd.read_csv('data/raw/MIPT_hackathon_dataset.csv')

# приводим типы данных к целевым (указаны в data_dictionary.xlsx)
# в файле data_preparation_research.ipynb я указала обоснование выбора целевых типов

# 1) str → category (для экономия памяти приводим строки в столбцах с малым количеством уникальных значений
# к типу category)
str_to_cat_cols = [
    'lead_Метод доставки', 'lead_Статус заказа на сайте',
    'lead_Ответственный за доставку', 'lead_source', 'lead_type',
    'lead_group', 'lead_Условный отказ', 'lead_LEADQUALIFYCATION',
    'lead_Источник', 'lead_FORMNAME', 'lead_utm_source',
    'lead_utm_medium', 'lead_utm_group', 'lead_будущие покупки',
    'lead_Модель телефона', 'lead_Квалификация лида',
    'lead_Категория и варианты выбора', 'lead_Проблема',
    'lead_Вид оплаты', 'lead_Компания Отправитель',
    'lead_Служба доставки', 'lead_Тариф Доставки',
]
for col in str_to_cat_cols:
    df[col] = df[col].astype('category')

# 2) float64 → datetime64[ns]
float_to_dt_cols = [
    'returned_ts', 'rejected_ts',
    'lead_Дата получения денег на Р/С',
    'lead_Дата перехода Передан в доставку',
    'received_ts', 'lead_closed_at', 'lead_Дата создания сделки',
    'issued_or_pvz_ts', 'handed_to_delivery_ts', 'closed_ts',
    'lead_Дата перехода в Сборку',
]
for col in float_to_dt_cols:
    df[col] = pd.to_datetime(df[col], unit='s')

# 3) int64 → datetime64[ns]
int_to_dt_cols = [
    'sale_ts', 'lead_created_at', 'lead_updated_at',
    'contact_created_at', 'contact_updated_at',
]
for col in int_to_dt_cols:
    df[col] = pd.to_datetime(df[col], unit='s')

# 4) str → datetime64[ns] (в столбце sale_date даты были строкового типа, приводим к datetime)
str_to_dt_cols = ['sale_date']
for col in str_to_dt_cols:
    df[col] = pd.to_datetime(df[col])

# 5) str → float64 (в исходных данных стоимость доставки хранится как '647,234'; приводим к float)
str_to_float_cols = ['lead_Стоимость доставки']
for col in str_to_float_cols:
    df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False), errors='coerce')

# 6) object → Int64 (True/False + NaN → 1/0 + NaN, nullable integer)
obj_to_int_cols = ['buyout_flag']
for col in obj_to_int_cols:
    df[col] = df[col].astype('Int64')

# выгрузка — версия с конвертацией типов
df.to_excel('data/clean/clean_data.xlsx', index=False)
print(f'clean_data.xlsx: {df.shape[0]} строк, {df.shape[1]} столбцов')

# удаление столбцов
# источник: docs/data_dictionary.xlsx (столбцы помечены серым, даны комментарии)

# 1) практически пустые столбцы (99-100% пропусков)
empty_cols = [
    'lead_loss_reason_id',            # 100% пропусков
    'lead_Нумерация сделки',          # 99.98%, 1 значение
    'lead_Поиск товаров GoSklad',     # 99.98%, 1 значение
    'lead_ACTUAL-FORMAT',             # 99.98%, 1 значение
    'lead_BANNER-SIZES',              # 99.98%, 1 значение
    'lead_Список товаров GoSklad',    # 99.98%, 1 значение
    'lead_Счет оплачен',              # 99.96%, 1 значение
    'lead_Дата приобретения изделия',  # 99.96%
    'lead_ПВЗ СДЭК',                 # 99.80%
    'lead_Оплачено клиентом',         # 99.78%
    'lead_Тип отправления',           # 99.72%
]

# 2) дубли столбцов (подтверждено в data_preparation_research.ipynb)
duplicate_cols = [
    'lead_WIDTH',                     # дубль, 99.98% пропусков
    'lead_HEIGHT',                    # дубль, 99.98% пропусков
    'lead_LEADQUALIFYCATION',         # дубль lead_Квалификация лида
    'lead_Линейная ширина (см)',      # дубль lead_Ширина
    'lead_Линейная высота (см)',      # дубль lead_Высота
    'lead_Линейная длина (см)',       # дубль lead_Длина
    'lead_Масса (гр)',                # дубль lead_Вес (грамм)* на 84.6%
    'current_status_id',              # полный дубль lead_status_id
    'lead_Сумма наложенного платежа (руб)',  # дубль lead_Объявленная ценность на 99.9%
]

# 3) неинформативные столбцы (1 уникальное значение во всём столбце)
zero_info_cols = [
    'lifecycle_incomplete',           # все значения одинаковые
    'lead_is_deleted',                # все значения одинаковые
]

# 4) неинформативные столбцы (>80% пропусков)
high_missing_cols = [
    'lead_Почтовый индекс',          # 95.97% пропусков
    'lead_Метод доставки',            # 95.97%
    'lead_Сумма заказа',              # 95.97%, не полносттью совпадает с lead_price
    'lead_Статус заказа на сайте',    # 95.82%
    'lead_LTV',                       # 95.78%
    'lead_Ответственный за доставку', # 92.33%
    'lead_source',                    # 90.01%, дубль lead_Источник
    'lead_Дата возврата посылки на склад',  # 87.24%
    'lead_type',                      # 90.01%
]

cols_to_drop = empty_cols + duplicate_cols + zero_info_cols + high_missing_cols
df_trimmed = df.drop(columns=cols_to_drop)
print(f'Удалено столбцов: {len(cols_to_drop)}')


# удаление пустых строк и дублирующейся информации
rows_before = len(df_trimmed)

df_trimmed = df_trimmed.dropna(how='all')
after_empty = len(df_trimmed)
print(f'Удалено пустых строк: {rows_before - after_empty}')

df_trimmed = df_trimmed.drop_duplicates()
after_dupes = len(df_trimmed)
print(f'Удалено дубликатов: {after_empty - after_dupes}')

# выгрузка версии без мусора
df_trimmed.to_excel('data/clean/clean_data_trimmed.xlsx', index=False)
print(f'clean_data_trimmed.xlsx: {df_trimmed.shape[0]} строк, {df_trimmed.shape[1]} столбцов')