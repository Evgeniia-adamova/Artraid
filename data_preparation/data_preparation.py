# чистый скрипт для подготовки данных
#запуск из папки data_preparation

import pandas as pd

new_df = pd.read_csv('data/raw/dataset_2025-03-01_2026-03-29_external.csv',
                      low_memory=False)
print(f'Загружено: {new_df.shape[0]} строк, {new_df.shape[1]} столбцов')


# приведение типов (ручная логика из первого скрипта + автоматическое удаление мусорных столбцов

# 1) str → category
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
    # один столбец из нового датасета (ок. 70 уникальных значений из 8200 строк -> удобнее хранить в category)
    'lead_REFERER',
]
for col in str_to_cat_cols:
    if col in new_df.columns:
        new_df[col] = new_df[col].astype('category')

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
    if col in new_df.columns:
        new_df[col] = pd.to_datetime(new_df[col], unit='s')

# 3) int64 → datetime64[ns]
int_to_dt_cols = [
    'sale_ts', 'lead_created_at', 'lead_updated_at',
    'contact_created_at', 'contact_updated_at',
]
for col in int_to_dt_cols:
    if col in new_df.columns:
        new_df[col] = pd.to_datetime(new_df[col], unit='s')

# 4) str → datetime64[ns]
str_to_dt_cols = ['sale_date']
for col in str_to_dt_cols:
    if col in new_df.columns:
        new_df[col] = pd.to_datetime(new_df[col])

# 5) str → float64
str_to_float_cols = ['lead_Стоимость доставки']
for col in str_to_float_cols:
    if col in new_df.columns:
        new_df[col] = pd.to_numeric(
            new_df[col].str.replace(',', '.', regex=False),
            errors='coerce'
        )

# 6) object → Int64
obj_to_int_cols = ['buyout_flag']
for col in obj_to_int_cols:
    if col in new_df.columns:
        new_df[col] = new_df[col].astype('Int64')

#____________________________________________________
# Удаляем мусор
# ЧАСТЬ 1: ручные списки (старая логика, проверено в research.ipynb; позже сделаю логику более гибкой)

# 1) практически пустые столбцы (99-100% пропусков)
empty_cols = [
    'lead_loss_reason_id',
    'lead_Нумерация сделки',
    'lead_Поиск товаров GoSklad',
    'lead_ACTUAL-FORMAT',
    'lead_BANNER-SIZES',
    'lead_Список товаров GoSklad',
    'lead_Счет оплачен',
    'lead_Дата приобретения изделия',
    'lead_ПВЗ СДЭК',
    'lead_Оплачено клиентом',
    'lead_Тип отправления',
]

# 2) дубли столбцов (подтверждено в research.ipynb)
duplicate_cols = [
    'lead_WIDTH',
    'lead_HEIGHT',
    'lead_LEADQUALIFYCATION',
    'lead_Линейная ширина (см)',
    'lead_Линейная высота (см)',
    'lead_Линейная длина (см)',
    'lead_Масса (гр)',
    'current_status_id',
    'lead_Сумма наложенного платежа (руб)',
]

# 3) неинформативные (1 уникальное значение)
zero_info_cols = [
    'lifecycle_incomplete',
    'lead_is_deleted',
]

# 4) >80% пропусков
high_missing_cols = [
    'lead_Почтовый индекс',
    'lead_Метод доставки',
    'lead_Сумма заказа',
    'lead_Статус заказа на сайте',
    'lead_LTV',
    'lead_Ответственный за доставку',
    'lead_source',
    'lead_Дата возврата посылки на склад',
    'lead_type',
]

manual_cols_to_drop = empty_cols + duplicate_cols + zero_info_cols + high_missing_cols

# ЧАСТЬ 2: автоматическое удаление новых мусорных столбцов

# старые столбцы, которые мы пока не удаляем автоматически
protected_cols = set(
    str_to_cat_cols + float_to_dt_cols + int_to_dt_cols +
    str_to_dt_cols + str_to_float_cols + obj_to_int_cols + [
        'lead_id', 'contact_id', 'lead_pipeline_id',
        'lead_status_id', 'lead_group_id',
        'lead_responsible_user_id', 'contact_responsible_user_id',
        'lead_price', 'lead_Объявленная ценность (руб)',
        'contact_Город', 'lead_Состав заказа',
        'lead_Ширина', 'lead_Высота', 'lead_Длина',
        'lead_Вес (грамм)*',
    ]
)

null_limit = 0.9
unique_ratio_limit = 0.8

def get_auto_trash(df, protected):

    trash = set()
    total = len(df)

    for col in df.columns:
        if col in protected:
            continue

        null_share = df[col].isna().mean()
        n_unique = df[col].nunique(dropna=True)
        name = col.lower()

        # >90% пропусков
        if null_share > null_limit:
            trash.add(col)
            continue

        # 0-1 уникальное значение
        if n_unique <= 1:
            trash.add(col)
            continue

        # уникальных > 80% строк (ID-подобные)
        if n_unique > total * unique_ratio_limit and col not in ['lead_tags']:
            trash.add(col)
            continue

        # технические CRM/рекламные поля
        marketing_patterns = [
            'click', 'roistat', 'yclid', 'banner', 'asset',
            'pcode', 'cookie', 'tranid', 'ym_uid', 'clientid',
            'etext', 'ysclid', 'ybaip', 'yprqee', 'rendered',
            'constructor', 'tildaspec', 'checkbox', 'stat-id',
            'policy_marketing', 'order-banners', 'test-tag',
            'ctime', 'rs_stat', 'cb',
        ]
        if any(p in name for p in marketing_patterns):
            trash.add(col)
            continue

        # персональные данные
        personal_patterns = [
            'телефон', 'email', 'phone', 'адрес клиента',
            'фио', 'first_name', 'last_name', 'день рождения',
            'telegramid', 'telegramusername', 'whatsgroup',
        ]
        if any(p in name for p in personal_patterns):
            trash.add(col)
            continue

    return trash


auto_trash = get_auto_trash(new_df, protected_cols)

# объединяем ручные и автоматические списки
all_cols_to_drop = set(manual_cols_to_drop) | auto_trash

# удаляем только те столбцы, которые есть в датасете
actual_to_drop = [c for c in all_cols_to_drop if c in new_df.columns]
missing = [c for c in all_cols_to_drop if c not in new_df.columns]


print(f'\nРучное удаление: {len(manual_cols_to_drop)} столбцов')
print(f'Автоматическое удаление: {len(auto_trash)} столбцов')
print(f'Итого к удалению: {len(actual_to_drop)} столбцов')
if missing:
    print(f'Warning: не найдены в датасете: {missing}')

new_df_trimmed = new_df.drop(columns=actual_to_drop)

#удаленяем пустые строки и дубли
rows_before = len(new_df_trimmed)

new_df_trimmed = new_df_trimmed.dropna(how='all')
after_empty = len(new_df_trimmed)
print(f'Удалено пустых строк: {rows_before - after_empty}')

if 'lead_id' in new_df_trimmed.columns:
    new_df_trimmed = new_df_trimmed.drop_duplicates(subset=['lead_id'], keep='first')
else:
    new_df_trimmed = new_df_trimmed.drop_duplicates()
after_dupes = len(new_df_trimmed)
print(f'Удалено дубликатов: {after_empty - after_dupes}')

# удаление строк с outcome_unknown == True (по запросу команды)
if 'outcome_unknown' in new_df_trimmed.columns:
    rows_before = len(new_df_trimmed)
    new_df_trimmed = new_df_trimmed[new_df_trimmed['outcome_unknown'] != True]
    print(f'Удалено строк с outcome_unknown=True: {rows_before - len(new_df_trimmed)}')

#делаем выгрузку
new_df_trimmed.to_excel('data/clean/clean_data_trimmed.xlsx', index=False)
print(f'clean_data_trimmed.xlsx: {new_df_trimmed.shape[0]} строк, {new_df_trimmed.shape[1]} столбцов')
