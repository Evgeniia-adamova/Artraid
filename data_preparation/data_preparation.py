import pandas as pd
import yaml

# Загрузка конфига
with open('config/target_dtypes.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Загрузка только нужных столбцов
use_cols = config['use_cols']
df = pd.read_csv(
    'data/raw/dataset_2025-03-01_2026-03-29_external.csv',
    usecols=use_cols,
    low_memory=False
)
print(f'Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов')

# Приведение типов

# str → category
for col in config.get('str_to_category', []):
    if col in df.columns:
        df[col] = df[col].astype('category')

#str →float
for col in config.get('str_to_float', []):
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(',', '.', regex=False),
            errors='coerce'
        )

# float → datetime (unix seconds)
for col in config.get('float_to_datetime', []):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')

# int → datetime (unix seconds)
for col in config.get('int_to_datetime', []):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')

# str → datetime
for col in config.get('str_to_datetime', []):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# object → Int64
for col in config.get('object_to_int', []):
    if col in df.columns:
        df[col] = df[col].astype('Int64')


# Дедупликация
rows_before = len(df)
df = df.dropna(how='all')
after_empty = len(df)
print(f'Удалено пустых строк: {rows_before - after_empty}')

if 'lead_id' in df.columns:
    df = df.drop_duplicates(subset=['lead_id'], keep='first')
else:
    df = df.drop_duplicates()
after_dupes = len(df)
print(f'Удалено дубликатов: {after_empty - after_dupes}')

# удаление строк с outcome_unknown == True (по запросу команды)
if 'outcome_unknown' in df.columns:
    rows_before = len(df)
    df = df[df['outcome_unknown'] != True]
    print(f'Удалено строк с outcome_unknown=True: {rows_before - len(df)}')
    df = df.drop(columns=['outcome_unknown'])

# Сохранение
df.to_excel('data/clean/clean_data.xlsx', index=False, engine='openpyxl')
print(f'Excel сохранён: clean_data.xlsx')
