# feature engineering: создание флагов и новых полей
# запуск из папки data_preparation

import re
import pandas as pd

df = pd.read_excel('data/clean/clean_data.xlsx')
df.drop(columns=['product_category'], errors='ignore', inplace=True)  # убираем старый столбец если остался
print(f'Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов')


#простые флаги

df['has_yclid'] = df['lead_yclid'].notna()
df['is_paid_mop'] = df['lead_Оплата МОП'].fillna('') == 'Оплачен'
df['is_repeat_client'] = df['contact_Число сделок'].fillna(1) > 1
df['has_discount'] = df['lead_Скидка'].notna()

# lead_tags

PROMO_PATTERNS = [
    'промокод', 'промо', 'artraid15', 'артрейд15', 'артрейд 15',
    '1+1', '2+1', '1+1=3', 'акция 1+1', 'акция',
    'маска в подарок', 'напульсник в подарок', 'наколенник в подарок',
    'пояс в подарок', 'кодовое слово', 'колесо призов',
    'для своих', 'дляcвоих',
]

# порядок - от более специфичного к общему
SOURCE_MAP = {
    'npotpz.ru': 'npotpz',
    'landings.npotpz.ru': 'npotpz',
    'poyasnica.npotpz.ru': 'npotpz',
    'tilda': 'tilda',
    'landing-artraid': 'artraid',
    'artraid.ru': 'artraid',
    'artraid': 'artraid',
    'callibri': 'callibri',
    'входящий': 'inbound',
    'jivo': 'jivo',
    'baza artraid': 'base',
    'база artraid': 'base',
    'база artraid исходящий': 'base',
    'baza': 'base',
    'база': 'base',
    'lояльная база': 'base',
    'лояльная база': 'base',
    'тпз': 'tpz',
    'авито тпз': 'tpz',
    'органик': 'organic',
    'посетители без рекламной кампании': 'organic',
    'тг канал': 'telegram',
    'zdorov.com': 'zdorov',
    'здоров': 'zdorov',
    'смс рассылка': 'sms',
}

def parse_tags(raw):
    if pd.isna(raw):
        return False, None, False

    tags_lower = [t.strip().lower() for t in str(raw).split(',')]

    has_promo = any(
        any(p in t for p in PROMO_PATTERNS)
        for t in tags_lower
    )

    source = None
    for t in tags_lower:
        if t in SOURCE_MAP:
            source = SOURCE_MAP[t]
            break
    if source is None:
        for t in tags_lower:
            for key, val in SOURCE_MAP.items():
                if key in t:
                    source = val
                    break
            if source:
                break

    is_yur = 'yur' in tags_lower

    return has_promo, source, is_yur


parsed = df['lead_tags'].apply(parse_tags)
df['has_promo'] = parsed.apply(lambda x: x[0])
df['lead_source_category'] = parsed.apply(lambda x: x[1]).astype('category')
df['is_yur'] = parsed.apply(lambda x: x[2])


# lead_Состав заказа: бинарные флаги т.к. потом для машинного обучения проще будет

CATEGORY_KEYWORDS = {
    'маска': 'маска',
    'наколенник': 'наколенник',
    'налокотник': 'наколенник', # налокотник = та же категория ортезов
    'шейный': 'бандаж_шейный',
    'бандаж': 'бандаж_шейный',
    'рогалик': 'бандаж_шейный', # рогалик на шею
    'повязка': 'повязка',
    'держатель': 'повязка', # держатель к повязке
    'напульсник': 'напульсник',
    'муфта': 'напульсник', # муфта -> запястье
    'тапки': 'обувь',
    'сапог': 'обувь',
    'боты': 'обувь',
    'стельки': 'обувь',
    'подушка': 'подушка',
    'матрас': 'матрас',
    'одеяло': 'постельное',
    'капсула': 'постельное',
    'пояс': 'пояс',
    'чехол': 'аксессуары',
    'варежка': 'аксессуары',
    'шапка': 'аксессуары',
    'шорты': 'аксессуары',
    'жилет': 'аксессуары',
    'накладка': 'аксессуары',
    'бандана': 'аксессуары',
    'накидка': 'аксессуары',
    'сумка': 'аксессуары',
    'пушап': 'аксессуары',
    'крем': 'крем',
    'воск': 'крем',
    'масляный': 'крем',
    # пчелиные продукты/БАДы
    'прополис': 'бады',
    'перга': 'бады',
    'огнёвка': 'бады',
    'огневка': 'бады',
    'гомогенат': 'бады',
    'трутневый': 'бады',
    'экстракт': 'бады',
}

ALL_CATEGORIES = list(dict.fromkeys(CATEGORY_KEYWORDS.values()))

_item_re = re.compile(r'\d+\)\s*(.+?)(?:\n|$)') # формат 1) берем слово после скобки
_alt_item_re = re.compile(r'([^,]+?)\s+x\d+')  # альтернативный формат: "Название x1 цена, ..." 


def get_order_categories(raw) -> set:
    if pd.isna(raw):
        return set()

    text = str(raw)

    items = _item_re.findall(text)
    items = [
        i.strip() for i in items
        if 'Доставка' not in i
        and 'Артикул' not in i
        and 'Кол-во' not in i
        and 'цена' not in i.lower()
    ]

    if not items:
        items = [
            m.group(1).strip() for m in _alt_item_re.finditer(text)
            if 'доставка' not in m.group(1).lower()
        ]

    # формат 3: "Название 2940р\nДоставка 300р"
    if not items:
        items = [
            line.strip() for line in text.split('\n')
            if line.strip()
            and 'доставка' not in line.lower()
            and not line.strip().isdigit()
        ]

    cats = set()
    for item in items:
        item_lower = item.lower()
        for kw, cat in CATEGORY_KEYWORDS.items():
            if kw in item_lower:
                cats.add(cat)
                break
    return cats


order_cats = df['lead_Состав заказа'].apply(get_order_categories)

for cat in ALL_CATEGORIES:
    df[f'has_{cat}'] = order_cats.apply(lambda cats: cat in cats)

df['n_product_categories'] = order_cats.apply(len).astype('Int64')


# приведение таймстемпов

df['sale_ts'] = pd.to_datetime(df['sale_ts'], errors='coerce')
df['lead_Дата создания сделки'] = pd.to_datetime(df['lead_Дата создания сделки'], errors='coerce')

TS_COLS = ['received_ts', 'issued_or_pvz_ts', 'handed_to_delivery_ts', 'closed_ts', 'returned_ts', 'rejected_ts']
for col in TS_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')


# таймстемп-фичи: час, день недели, месяц

for ts_col, prefix in [('sale_ts', 'sale'), ('lead_Дата создания сделки', 'lead_created')]:
    if ts_col in df.columns:
        df[f'{prefix}_hour']       = df[ts_col].dt.hour.astype('Int64')
        df[f'{prefix}_day_of_week'] = df[ts_col].dt.dayofweek.astype('Int64')  # 0=пн, 6=вс
        df[f'{prefix}_month']      = df[ts_col].dt.month.astype('Int64')


# интервалы между событиями (в днях)

def days_between(ts_end, ts_start):
    delta = (ts_end - ts_start).dt.total_seconds() / 86400
    delta[delta < 0] = None  # отрицательные интервалы - артефакты
    return delta

INTERVALS = [
    ('days_creation_to_sale', 'sale_ts', 'lead_Дата создания сделки'),
    ('days_sale_to_received', 'received_ts', 'sale_ts'),
    ('days_received_to_issued', 'issued_or_pvz_ts', 'received_ts'),
    ('days_sale_to_closed', 'closed_ts', 'sale_ts'),
    ('days_sale_to_returned', 'returned_ts', 'sale_ts'),
    ('days_sale_to_rejected', 'rejected_ts', 'sale_ts'),
]

for col_name, end_col, start_col in INTERVALS:
    if end_col in df.columns and start_col in df.columns:
        df[col_name] = days_between(df[end_col], df[start_col])

# проверка аномальных лагов sale_ts − lead_Дата создания сделки

lag = df['days_creation_to_sale'] if 'days_creation_to_sale' in df.columns else (df['sale_ts'] - df['lead_Дата создания сделки']).dt.total_seconds() / 86400
print(f'\nЛаг sale_ts − lead_Дата создания сделки:')
print(f' среднее {lag.mean():.2f} дней, медиана {lag.median():.2f} дней')
print(f' >100 дней: {(lag > 100).sum()} строк')
print(f' <0 дней:   {(lag < 0).sum()} строк')

# как обсудили на косультационной встрече - можно дропнуть 4 строки с аномальными лагами
anomaly_mask = lag > 100
if anomaly_mask.sum() > 0:
    df = df[~anomaly_mask].reset_index(drop=True)
    print(f' Удалено аномальных строк (лаг >100 дней): {anomaly_mask.sum()}')

# удаление технических заказов без тарифа доставки
if 'lead_Тариф Доставки' in df.columns:
    n_before = len(df)
    df = df[df['lead_Тариф Доставки'].notna()].reset_index(drop=True)
    print(f' Удалено строк без тарифа доставки: {n_before - len(df)}')


# статистика

print('\nФлаги:')
for col in ['has_yclid', 'has_promo', 'is_paid_mop', 'is_repeat_client', 'is_yur']:
    n = df[col].sum()
    print(f'  {col}: {n} ({df[col].mean()*100:.1f}%)')

print('\nlead_source_category:')
print(df['lead_source_category'].value_counts(dropna=False).to_string())


print('\nКатегории товаров:')
for cat in ALL_CATEGORIES:
    col = f'has_{cat}'
    n = df[col].sum()
    print(f'  {col}: {n} ({n/len(df)*100:.1f}%)')
print(f'  n_product_categories: среднее={df["n_product_categories"].mean():.2f}, медиана={df["n_product_categories"].median()}')

print('\nТаймстемп-фичи:')
for col in ['sale_hour', 'sale_day_of_week', 'sale_month', 'lead_created_hour', 'lead_created_day_of_week', 'lead_created_month']:
    if col in df.columns:
        print(f'  {col}: непустых={df[col].notna().sum()}, мода={df[col].mode().iloc[0] if df[col].notna().any() else "—"}')

print('\nИнтервалы (дней):')
for col_name, _, _ in INTERVALS:
    if col_name in df.columns:
        s = df[col_name].dropna()
        print(f'  {col_name}: n={len(s)}, среднее={s.mean():.1f}, медиана={s.median():.1f}')


# delivery_group: создаем группы в зависимости от кол-ва дней от момента передачи в доставку до ПВЗ/курьер
df.loc[df['days_handed_to_issued_pvz'] < 0, 'days_handed_to_issued_pvz'] = None

def assign_delivery_group(days):
    if pd.isna(days) or days < 0:
        return None
    if days <= 3:
        return 'быстрая'
    elif days <= 5:
        return 'средняя'
    elif days <= 10:
        return 'долгая'
    else:
        return 'очень долгая'

df['delivery_group'] = df['days_handed_to_issued_pvz'].apply(assign_delivery_group).astype('category')

# price_group: создаем группы в зависимости от стоимости заказа
def assign_price_group(price):
    if pd.isna(price) or price < 0:
        return None
    if price <= 5000:
        return 'до 5к'
    elif price <= 15000:
        return '5к-15к'
    elif price <= 20000:
        return '15к-20к'
    else:
        return '20к+'

df['price_group'] = df['lead_price'].apply(assign_price_group).astype('category')

# сейв

df.to_excel('data/clean/clean_data.xlsx', index=False, engine='openpyxl')
print(f'\nСохранено: clean_data.xlsx — {len(df)} строк, {df.shape[1]} столбцов')
