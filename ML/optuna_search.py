"""
Подбор гиперпараметров XGBoost и CatBoost через Optuna (байесовская оптимизация).
Запускать отдельно от models_mk1.py. Найденные параметры вставлять вручную.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
from pathlib import Path

import xgboost as xgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

# пути
ml_path = Path(__file__).parent
base_path = ml_path.parent
clean_data_file = base_path / 'data_preparation' / 'data' / 'clean' / 'clean_data.xlsx'

N_TRIALS = 80
CV_FOLDS = 5
SEED = 42


# препроцессинг (дублируем из models_mk1.py)

def sig3OutlDetector(DATA, columns):
    Stds = DATA[columns].std()
    Means = DATA[columns].mean()
    mask = DATA[columns]
    mask = mask[((-3*Stds+Means <= mask) & (3*Stds+Means >= mask))[columns[0]]]
    return DATA.iloc[mask.index.tolist()].reset_index(drop=True)

def bool2flag(DATA):
    DAT = DATA.copy()
    cols = DAT.select_dtypes(['bool']).columns.tolist()
    DAT[cols] = DAT[cols] * 1
    return DAT

def Nan2Missing(DATA):
    DAT = DATA.copy()
    cols = DAT.select_dtypes(['object']).columns.tolist()
    DAT[cols] = DAT[cols].fillna('Неизвестно')
    return DAT

def ReplacingTS(DATA, stages):
    st = stages[::-1]
    DAT = DATA.copy()
    for idx, ts_col_1 in enumerate(st[:-1]):
        for ts_col_2 in st[idx+1:]:
            DAT[ts_col_1] = DAT[ts_col_1].fillna(DAT[ts_col_2][DAT[ts_col_1].isna()])
            if DAT[ts_col_1].isna().sum() == 0:
                break
    return DAT

def Nan2Adequate(DATA, last_stage=None):
    DAT = DATA.copy()
    if 'lead_Скидка' in DAT.columns:
        DAT[['lead_Скидка']] = DAT[['lead_Скидка']].fillna(0)
    DAT[['lead_Квалификация лида']] = DAT[['lead_Квалификация лида']].fillna('ПРОПУСК')
    DAT[['lead_source_category']] = DAT[['lead_source_category']].fillna('ПРОПУСК')
    ltv_cols = [c for c in ['contact_LTV', 'contact_Число сделок'] if c in DAT.columns]
    DAT[ltv_cols] = DAT[ltv_cols].fillna(0)
    delivery_cols = [c for c in ['delivery_group', 'days_sale_to_handed', 'days_handed_to_issued_pvz', 'lead_Тариф Доставки'] if c in DAT.columns]
    DAT.dropna(subset=delivery_cols, inplace=True, ignore_index=True)
    DAT = Nan2Missing(DAT)
    stages = [c for c in ['sale_ts', 'lead_Дата перехода в Сборку', 'handed_to_delivery_ts', 'issued_or_pvz_ts'] if c in DAT.columns]
    if last_stage is not None and last_stage in stages:
        drop = stages[stages.index(last_stage):]
        DAT.drop([c for c in drop if c in DAT.columns], axis=1, inplace=True)
        stages = stages[:stages.index(last_stage)]
    DAT = ReplacingTS(DAT, stages)
    return DAT

def datetime2EpSec(df):
    DAT = df.copy()
    dt_cols = DAT.select_dtypes(['datetime64[ns]']).columns.tolist()
    for col in dt_cols:
        DAT[col] = (DAT[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    return DAT

def target_encode(X_train_df, X_test_df, y_train_s, cat_cols, n_splits=5, seed=42):
    X_tr = X_train_df.copy().reset_index(drop=True)
    X_te = X_test_df.copy().reset_index(drop=True)
    y_tr = pd.Series(y_train_s).reset_index(drop=True).astype(float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for col in [c for c in cat_cols if c in X_tr.columns]:
        global_mean = float(y_tr.mean())
        oof = np.full(len(X_tr), global_mean)
        for fold_tr_idx, fold_val_idx in kf.split(X_tr):
            fold_map = y_tr.iloc[fold_tr_idx].groupby(X_tr[col].iloc[fold_tr_idx]).mean()
            oof[fold_val_idx] = X_tr[col].iloc[fold_val_idx].map(fold_map).fillna(global_mean).values
        full_map = y_tr.groupby(X_tr[col]).mean()
        X_te[col] = X_te[col].map(full_map).fillna(global_mean).values
        X_tr[col] = oof
    return X_tr, X_te


# загрузка и подготовка данных

print('Загрузка данных...')
CLEAN_DATA = pd.read_excel(clean_data_file)
CLEAN_DATA = CLEAN_DATA.sort_values('sale_ts').reset_index(drop=True)

CLEAN_DATA['_buyout_price'] = CLEAN_DATA['lead_price'] * (CLEAN_DATA['buyout_flag'] == True)
CLEAN_DATA['ltv_before'] = CLEAN_DATA.groupby('contact_id')['_buyout_price'].transform(lambda x: x.shift(1).cumsum().fillna(0))
CLEAN_DATA['orders_before'] = CLEAN_DATA.groupby('contact_id')['lead_price'].transform(lambda x: (~x.isna()).cumsum().shift(1).fillna(0))
_buyouts = CLEAN_DATA['buyout_flag'].astype(float)
CLEAN_DATA['buyouts_before'] = CLEAN_DATA.groupby('contact_id')[_buyouts.name].transform(lambda x: x.shift(1).cumsum().fillna(0))
CLEAN_DATA['buyout_rate_before'] = np.where(CLEAN_DATA['orders_before'] > 0, CLEAN_DATA['buyouts_before'] / CLEAN_DATA['orders_before'], -1.0)
CLEAN_DATA['days_since_last_order'] = CLEAN_DATA.groupby('contact_id')['sale_ts'].transform(lambda x: x.diff().dt.total_seconds() / 86400).fillna(-1.0)
CLEAN_DATA['mgr_orders_before'] = CLEAN_DATA.groupby('lead_responsible_user_id')['lead_price'].transform(lambda x: (~x.isna()).cumsum().shift(1).fillna(0))
CLEAN_DATA.drop(columns=['_buyout_price', 'buyouts_before'], inplace=True)

drop1 = ['lead_tags', 'contact_Город', 'contact_id', 'lead_id', 'lead_Состав заказа', 'lead_yclid']
drop2 = [
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

DATA = CLEAN_DATA.drop(columns=[c for c in drop1 + drop2 if c in CLEAN_DATA.columns])
del CLEAN_DATA
gc.collect()

DATA = bool2flag(DATA)
DATA = sig3OutlDetector(DATA, ['lead_price'])
DATA = Nan2Adequate(DATA)
DATA = datetime2EpSec(DATA)
DATA['sale_quarter'] = ((DATA['sale_month'] - 1) // 3 + 1).astype(int)
DATA['is_weekend'] = DATA['sale_day_of_week'].isin([5, 6]).astype(int)

num_cols = DATA.select_dtypes('number').columns
DATA[num_cols] = DATA[num_cols].fillna(DATA[num_cols].median())

Y = DATA['buyout_flag']
X_df = DATA.drop('buyout_flag', axis=1)
features = X_df.columns.tolist()
del DATA
gc.collect()

X_df_train, X_df_test, y_train, y_test = train_test_split(X_df, Y, test_size=0.2, random_state=SEED, stratify=Y)

CAT_COLS = [c for c in [
    'lead_region', 'lead_source_category', 'lead_Тариф Доставки',
    'lead_Квалификация лида', 'lead_Вид оплаты', 'lead_Проблема',
    'lead_Категория и варианты выбора', 'lead_Служба доставки',
    'lead_responsible_user_id',
] if c in features]

X_df_train, X_df_test = target_encode(X_df_train, X_df_test, y_train, CAT_COLS)
X_train = X_df_train.values.astype(float)
X_test = X_df_test.values.astype(float)

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale = n_neg / n_pos
print(f'Train: {len(X_train)} | Test: {len(X_test)} | Признаков: {len(features)}\n')


# CV-оценка (чтобы не переобучаться на тест)

def cv_auc(model_fn, X, y, n_splits=CV_FOLDS):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[val_idx], m.predict_proba(X[val_idx])[:, 1]))
    return float(np.mean(aucs))


# XGBoost Optuna

print(f'XGBoost: поиск ({N_TRIALS} trials)...')

def xgb_objective(trial):
    params = {
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators':     trial.suggest_int('n_estimators', 200, 1000),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0),
    }
    def make():
        return xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale,
            eval_metric='auc',
            seed=SEED,
            verbosity=0,
            **params
        )
    return cv_auc(make, X_train, np.array(y_train))

xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
xgb_study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)

best_xgb = xgb_study.best_params
best_xgb_auc = xgb_study.best_value

# финальная оценка на тест с лучшими параметрами
final_xgb = xgb.XGBClassifier(
    objective='binary:logistic', scale_pos_weight=scale,
    eval_metric='auc', seed=SEED, verbosity=0, **best_xgb
)
final_xgb.fit(X_train, np.array(y_train))
test_xgb_auc = roc_auc_score(np.array(y_test), final_xgb.predict_proba(X_test)[:, 1])

print(f'\nXGBoost лучший CV AUC:   {best_xgb_auc:.4f}')
print(f'XGBoost тест AUC:        {test_xgb_auc:.4f}')
print(f'Лучшие параметры XGBoost:')
for k, v in best_xgb.items():
    print(f'  {k}: {v}')


# CatBoost Optuna

print(f'\nCatBoost: поиск ({N_TRIALS} trials)...')

def cb_objective(trial):
    params = {
        'depth':          trial.suggest_int('depth', 3, 8),
        'learning_rate':  trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_trees':      trial.suggest_int('num_trees', 200, 1000),
        'subsample':      trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg':    trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
    }
    def make():
        return cb.CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            auto_class_weights='Balanced',
            random_seed=SEED,
            verbose=False,
            **params
        )
    return cv_auc(make, X_train, np.array(y_train))

cb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.optimize(cb_objective, n_trials=N_TRIALS, show_progress_bar=True)

best_cb = cb_study.best_params
best_cb_auc = cb_study.best_value

final_cb = cb.CatBoostClassifier(
    loss_function='Logloss', eval_metric='AUC',
    auto_class_weights='Balanced', random_seed=SEED, verbose=False, **best_cb
)
final_cb.fit(X_train, np.array(y_train))
test_cb_auc = roc_auc_score(np.array(y_test), final_cb.predict_proba(X_test)[:, 1])

print(f'\nCatBoost лучший CV AUC:  {best_cb_auc:.4f}')
print(f'CatBoost тест AUC:       {test_cb_auc:.4f}')
print(f'Лучшие параметры CatBoost:')
for k, v in best_cb.items():
    print(f'  {k}: {v}')

print('\n=== ИТОГ ===')
print(f'XGBoost  тест AUC: {test_xgb_auc:.4f}')
print(f'CatBoost тест AUC: {test_cb_auc:.4f}')
print('\nВставь лучшие параметры в models_mk1.py.')

# сохраняем результаты в файл
import json
results = {
    'xgboost': {
        'best_params': best_xgb,
        'cv_auc': round(best_xgb_auc, 6),
        'test_auc': round(test_xgb_auc, 6),
    },
    'catboost': {
        'best_params': best_cb,
        'cv_auc': round(best_cb_auc, 6),
        'test_auc': round(test_cb_auc, 6),
    },
}
out_path = ml_path / 'optuna_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'\nРезультаты сохранены: {out_path}')
