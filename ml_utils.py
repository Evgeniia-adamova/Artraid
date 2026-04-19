import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cb
from pathlib import Path
from sklearn.model_selection import train_test_split
import streamlit as st

_BASE = Path(__file__).parent
_ML_PATH = _BASE / 'ML'
_DATA_FILE = _BASE / 'data_preparation' / 'data' / 'clean' / 'clean_data.xlsx'

_CAT_COLS_MK1 = [
    'lead_region', 'lead_source_category', 'lead_Тариф Доставки',
    'lead_Квалификация лида', 'lead_Вид оплаты', 'lead_Проблема',
    'lead_Категория и варианты выбора', 'lead_Служба доставки',
    'lead_responsible_user_id',
]
_CAT_COLS_MK2 = _CAT_COLS_MK1 + ['delivery_group']

_DROP1 = ['lead_tags', 'contact_Город', 'contact_id', 'lead_id', 'lead_Состав заказа', 'lead_yclid']

_DROP2_MK1 = [
    'sale_date', 'closed_ts', 'received_ts', 'rejected_ts', 'returned_ts', 'days_to_outcome',
    'lead_Условный отказ', 'lead_Оплата МОП', 'is_paid_mop',
    'lead_Дата получения денег на Р/С', 'lead_Источник', 'lead_Дата создания сделки',
    'lead_Скидка', 'handed_to_delivery_ts',
    'days_sale_to_received', 'days_sale_to_rejected', 'days_sale_to_returned', 'days_sale_to_closed',
    'days_sale_to_handed', 'days_handed_to_issued_pvz', 'delivery_group',
    'issued_or_pvz_ts', 'days_received_to_issued',
    'contact_LTV', 'contact_Число сделок',
    'price_group', 'sale_ts',
]

_DROP2_MK2 = [
    'sale_date', 'closed_ts', 'received_ts', 'rejected_ts', 'returned_ts', 'days_to_outcome',
    'lead_Условный отказ', 'lead_Оплата МОП', 'is_paid_mop',
    'lead_Дата получения денег на Р/С', 'lead_Источник', 'lead_Дата создания сделки',
    'lead_Скидка',
    'days_sale_to_received', 'days_sale_to_rejected', 'days_sale_to_returned', 'days_sale_to_closed',
    'days_received_to_issued',
    'contact_LTV', 'contact_Число сделок',
    'price_group', 'sale_ts',
]


def _bool2flag(DATA):
    DAT = DATA.copy()
    cols = DAT.select_dtypes(['bool']).columns.tolist()
    DAT[cols] = DAT[cols] * 1
    return DAT


def _sig3(DATA, columns):
    Stds = DATA[columns].std()
    Means = DATA[columns].mean()
    mask = DATA[columns]
    mask = mask[((-3*Stds+Means <= mask) & (3*Stds+Means >= mask))[columns[0]]]
    return DATA.iloc[mask.index.tolist()].reset_index(drop=True)


def _nan2missing(DATA):
    DAT = DATA.copy()
    cols = DAT.select_dtypes(['object']).columns.tolist()
    DAT[cols] = DAT[cols].fillna('Неизвестно')
    return DAT


def _replacing_ts(DATA, stages):
    st_r = stages[::-1]
    DAT = DATA.copy()
    for idx, ts_col_1 in enumerate(st_r[:-1]):
        for ts_col_2 in st_r[idx+1:]:
            DAT[ts_col_1] = DAT[ts_col_1].fillna(DAT[ts_col_2][DAT[ts_col_1].isna()])
            if DAT[ts_col_1].isna().sum() == 0:
                break
    return DAT


def _nan2adequate(DATA):
    DAT = DATA.copy()
    if 'lead_Скидка' in DAT.columns:
        DAT[['lead_Скидка']] = DAT[['lead_Скидка']].fillna(0)
    DAT[['lead_Квалификация лида']] = DAT[['lead_Квалификация лида']].fillna('ПРОПУСК')
    DAT[['lead_source_category']] = DAT[['lead_source_category']].fillna('ПРОПУСК')
    ltv_cols = [c for c in ['contact_LTV', 'contact_Число сделок'] if c in DAT.columns]
    DAT[ltv_cols] = DAT[ltv_cols].fillna(0)
    delivery_cols = [c for c in ['delivery_group', 'days_sale_to_handed', 'days_handed_to_issued_pvz', 'lead_Тариф Доставки'] if c in DAT.columns]
    DAT.dropna(subset=delivery_cols, inplace=True, ignore_index=True)
    DAT = _nan2missing(DAT)
    stages = [c for c in ['sale_ts', 'lead_Дата перехода в Сборку', 'handed_to_delivery_ts', 'issued_or_pvz_ts'] if c in DAT.columns]
    DAT = _replacing_ts(DAT, stages)
    return DAT


def _dt2epoch(df):
    DAT = df.copy()
    dt_cols = DAT.select_dtypes(['datetime64[ns]']).columns.tolist()
    for col in dt_cols:
        DAT[col] = (DAT[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    return DAT


def _add_history(DATA, mk1_extras=False):
    CD = DATA.copy()
    CD = CD.sort_values('sale_ts').reset_index(drop=True)
    CD['_buyout_price'] = CD['lead_price'] * (CD['buyout_flag'] == True).astype(int)
    CD['ltv_before'] = CD.groupby('contact_id')['_buyout_price'].transform(
        lambda x: x.shift(1).cumsum().fillna(0))
    CD['orders_before'] = CD.groupby('contact_id')['lead_price'].transform(
        lambda x: (~x.isna()).cumsum().shift(1).fillna(0))
    _buyouts = CD['buyout_flag'].astype(float)
    CD['buyouts_before'] = CD.groupby('contact_id')[_buyouts.name].transform(
        lambda x: x.shift(1).cumsum().fillna(0))
    CD['buyout_rate_before'] = np.where(
        CD['orders_before'] > 0,
        CD['buyouts_before'] / CD['orders_before'], -1.0)
    CD['mgr_orders_before'] = CD.groupby('lead_responsible_user_id')['lead_price'].transform(
        lambda x: (~x.isna()).cumsum().shift(1).fillna(0))
    if mk1_extras:
        CD['days_since_last_order'] = CD.groupby('contact_id')['sale_ts'].transform(
            lambda x: x.diff().dt.total_seconds() / 86400).fillna(-1.0)
    CD.drop(columns=['_buyout_price', 'buyouts_before'], inplace=True)
    return CD


@st.cache_data(show_spinner="Подготовка данных для прогноза...")
def _get_encoders(version: str):
    drop2 = _DROP2_MK1 if version == 'mk1' else _DROP2_MK2
    cat_cols_all = _CAT_COLS_MK1 if version == 'mk1' else _CAT_COLS_MK2

    DATA = pd.read_excel(_DATA_FILE)
    DATA = _add_history(DATA, mk1_extras=(version == 'mk1'))
    DATA = DATA.drop(columns=[c for c in _DROP1 + drop2 if c in DATA.columns])
    DATA = _bool2flag(DATA)
    DATA = _sig3(DATA, ['lead_price'])
    DATA = _nan2adequate(DATA)
    DATA = _dt2epoch(DATA)
    DATA['sale_quarter'] = ((DATA['sale_month'] - 1) // 3 + 1).astype(int)
    if version == 'mk1':
        DATA['is_weekend'] = DATA['sale_day_of_week'].isin([5, 6]).astype(int)

    num_cols = DATA.select_dtypes('number').columns.tolist()
    DATA[num_cols] = DATA[num_cols].fillna(DATA[num_cols].median())

    Y = DATA['buyout_flag']
    X_df = DATA.drop('buyout_flag', axis=1)
    features = X_df.columns.tolist()

    X_train_df, _, y_train, _ = train_test_split(
        X_df, Y, test_size=0.2, random_state=42, stratify=Y)

    global_mean = float(y_train.mean())
    cat_cols = [c for c in cat_cols_all if c in features]

    encode_maps = {}
    for col in cat_cols:
        full_map = y_train.groupby(X_train_df[col]).mean()
        encode_maps[col] = full_map.to_dict()

    train_num_cols = [c for c in num_cols if c in X_train_df.columns]
    medians = X_train_df[train_num_cols].median().to_dict()

    return {
        'features': features,
        'cat_cols': cat_cols,
        'encode_maps': encode_maps,
        'global_mean': global_mean,
        'medians': medians,
    }


@st.cache_resource(show_spinner="Загрузка моделей MK1...")
def _load_models_mk1():
    ml = _ML_PATH / 'Models'
    with open(ml / 'LogReg-MK1.pkl', 'rb') as f:
        lr_data = pickle.load(f)
    with open(ml / 'RF-MK1.pkl', 'rb') as f:
        rf_data = pickle.load(f)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(ml / 'XGBoost-MK1.json'))
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(ml / 'CatBoost-MK1.cbm'))
    return {
        'lr': lr_data['model'],
        'lr_scaler': lr_data['scaler'],
        'rf': rf_data['model'],
        'xgb': xgb_model,
        'cb': cb_model,
    }


@st.cache_resource(show_spinner="Загрузка моделей MK2...")
def _load_models_mk2():
    ml = _ML_PATH / 'Models_mk2'
    with open(ml / 'LogReg-MK2.pkl', 'rb') as f:
        lr_data = pickle.load(f)
    with open(ml / 'RF-MK2.pkl', 'rb') as f:
        rf_data = pickle.load(f)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(ml / 'XGBoost-MK2.json'))
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(ml / 'CatBoost-MK2.cbm'))
    return {
        'lr': lr_data['model'],
        'lr_scaler': lr_data['scaler'],
        'rf': rf_data['model'],
        'xgb': xgb_model,
        'cb': cb_model,
    }


def _build_row(input_raw: dict, encoders: dict) -> np.ndarray:
    features = encoders['features']
    encode_maps = encoders['encode_maps']
    global_mean = encoders['global_mean']
    medians = encoders['medians']
    cat_cols = encoders['cat_cols']

    row = {f: medians.get(f, 0.0) for f in features}

    # defaults for customers with no history
    if not int(input_raw.get('is_repeat_client', 0)):
        row['ltv_before'] = 0.0
        row['orders_before'] = 0.0
        row['buyout_rate_before'] = -1.0
        if 'days_since_last_order' in features:
            row['days_since_last_order'] = -1.0

    # numeric user inputs
    for k, v in input_raw.items():
        if k in features and k not in cat_cols:
            row[k] = float(v)

    # derived features
    if 'sale_month' in input_raw and 'sale_quarter' in features:
        row['sale_quarter'] = int((int(input_raw['sale_month']) - 1) // 3 + 1)
    if 'sale_day_of_week' in input_raw and 'is_weekend' in features:
        row['is_weekend'] = int(int(input_raw['sale_day_of_week']) in (5, 6))

    # target encoding for categorical inputs
    for col in cat_cols:
        if col in input_raw:
            row[col] = encode_maps[col].get(input_raw[col], global_mean)
        else:
            row[col] = global_mean

    return np.array([row[f] for f in features], dtype=float).reshape(1, -1)


def predict_mk1(input_raw: dict) -> dict:
    encoders = _get_encoders('mk1')
    models = _load_models_mk1()
    X = _build_row(input_raw, encoders)
    X_scaled = models['lr_scaler'].transform(X)
    return {
        'LogReg': float(models['lr'].predict_proba(X_scaled)[0, 1]),
        'RF': float(models['rf'].predict_proba(X)[0, 1]),
        'XGBoost': float(models['xgb'].predict_proba(X)[0, 1]),
        'CatBoost': float(models['cb'].predict_proba(X)[0, 1]),
    }


def predict_mk2(input_raw: dict) -> dict:
    encoders = _get_encoders('mk2')
    models = _load_models_mk2()
    X = _build_row(input_raw, encoders)
    X_scaled = models['lr_scaler'].transform(X)
    return {
        'LogReg': float(models['lr'].predict_proba(X_scaled)[0, 1]),
        'RF': float(models['rf'].predict_proba(X)[0, 1]),
        'XGBoost': float(models['xgb'].predict_proba(X)[0, 1]),
        'CatBoost': float(models['cb'].predict_proba(X)[0, 1]),
    }


def render_results(probs: dict):
    avg = np.mean(list(probs.values()))

    rows = []
    for model, p in probs.items():
        if p >= 0.7:
            verdict = "Скорее выкупит"
        elif p >= 0.5:
            verdict = "Вероятно выкупит"
        else:
            verdict = "Риск невыкупа"
        rows.append({'Модель': model, 'P(выкуп)': f'{p:.1%}', 'Вердикт': verdict})

    import streamlit as st
    st.markdown("### Результаты моделей")
    st.table(pd.DataFrame(rows))

    st.markdown("### Итог")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Средняя вероятность выкупа", f'{avg:.1%}')
    with col2:
        if avg >= 0.7:
            st.success("Скорее всего выкупит")
        elif avg >= 0.5:
            st.info("Вероятно выкупит")
        else:
            st.warning("Высокий риск невыкупа")
    st.progress(float(avg))
