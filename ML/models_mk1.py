"""
Модели MK1 - прогноз выкупа в момент создания заказа
Logistic Regression, Random Forest, XGBoost, CatBoost
"""

import pickle
import numpy as np
import pandas as pd
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.inspection import permutation_importance

# пути
ml_path = Path(__file__).parent
base_path = ml_path.parent
clean_data_file = base_path / 'data_preparation' / 'data' / 'clean' / 'clean_data.xlsx'
model_path = ml_path / 'Models'
log_path = ml_path / 'Logs'
img_path = ml_path / 'Images'


# препроцессинг

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
    # NaN в категориальных признаках помечаем отдельно, а не смешиваем с реальными значениями
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

def CategoricalOrdinal(DATA):
    DAT = DATA.copy()
    for col in DAT.select_dtypes(['object']).columns.tolist():
        unique = pd.unique(DAT[col]).tolist()
        mapping = {v: i for i, v in enumerate(unique)}
        DAT[col] = DAT[col].map(mapping)
    return DAT


# загрузка и подготовка данных

print('Загрузка данных...')
CLEAN_DATA = pd.read_excel(clean_data_file)

drop1 = ['lead_tags', 'contact_Город', 'contact_id', 'lead_id', 'lead_Состав заказа', 'lead_yclid', 'lead_responsible_user_id']
drop2 = [
    'sale_date', 'closed_ts', 'received_ts', 'rejected_ts', 'returned_ts', 'days_to_outcome',
    'lead_Условный отказ', 'lead_Оплата МОП', 'is_paid_mop',
    'lead_Дата получения денег на Р/С', 'lead_Источник', 'lead_Дата создания сделки',
    'lead_Скидка', 'handed_to_delivery_ts',
    # утечка: интервалы известны только после исхода сделки
    'days_sale_to_received', 'days_sale_to_rejected', 'days_sale_to_returned', 'days_sale_to_closed',
    # модель 1: логистика неизвестна на момент создания заказа
    'days_sale_to_handed', 'days_handed_to_issued_pvz', 'delivery_group',
    'issued_or_pvz_ts', 'days_received_to_issued',
    # утечка: LTV обновляется в CRM после исхода сделки
    'contact_LTV', 'contact_Число сделок',
    # price_group дублирует lead_price, sale_ts разложен на hour/dow/month
    'price_group', 'sale_ts',
]

DATA = CLEAN_DATA.drop(columns=[c for c in drop1 + drop2 if c in CLEAN_DATA.columns])
del CLEAN_DATA
gc.collect()

DATA = bool2flag(DATA)
DATA = sig3OutlDetector(DATA, ['lead_price'])
DATA = Nan2Adequate(DATA)
DATA = datetime2EpSec(DATA)
DATA = CategoricalOrdinal(DATA)

num_cols = DATA.select_dtypes('number').columns
DATA[num_cols] = DATA[num_cols].fillna(DATA[num_cols].median())

Y = DATA['buyout_flag'].values
X = DATA.drop('buyout_flag', axis=1).values
features = DATA.drop('buyout_flag', axis=1).columns.tolist()
del DATA
gc.collect()

seed = 42
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
print(f'Train: {len(X_train)} строк | class 0: {n_neg}, class 1: {n_pos}')
print(f'Признаки ({len(features)}): {features}\n')


# Logistic Regression

print('Logistic Regression')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=seed,
    solver='lbfgs'
)

start = datetime.now()
lr_model.fit(X_train_scaled, y_train)
lr_time = str(datetime.now() - start).split('.')[0]

lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
LR_THRESHOLD = 0.5
lr_preds = (lr_proba >= LR_THRESHOLD).astype(int)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_preds, average='macro')

print(classification_report(y_test, lr_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {lr_auc:.4f} | F1 macro: {lr_f1:.4f} | Время: {lr_time}\n')


# Random Forest

print('Random Forest')
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight='balanced',
    random_state=seed,
    n_jobs=-1
)

start = datetime.now()
rf_model.fit(X_train, y_train)
rf_time = str(datetime.now() - start).split('.')[0]

rf_proba = rf_model.predict_proba(X_test)[:, 1]
RF_THRESHOLD = 0.5
rf_preds = (rf_proba >= RF_THRESHOLD).astype(int)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_f1 = f1_score(y_test, rf_preds, average='macro')

print(classification_report(y_test, rf_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {rf_auc:.4f} | F1 macro: {rf_f1:.4f} | Время: {rf_time}\n')


# XGBoost

print('XGBoost')
scale = n_neg / n_pos  # компенсация дисбаланса классов

xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.01,
    n_estimators=5000,
    objective='binary:logistic',
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    early_stopping_rounds=50,
    seed=seed,
    verbosity=0
)

start = datetime.now()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
xgb_time = str(datetime.now() - start).split('.')[0]

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
XGB_THRESHOLD = 0.5
xgb_preds = (xgb_proba >= XGB_THRESHOLD).astype(int)
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_f1 = f1_score(y_test, xgb_preds, average='macro')

print(classification_report(y_test, xgb_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {xgb_auc:.4f} | F1 macro: {xgb_f1:.4f} | Итераций: {xgb_model.best_iteration} | Время: {xgb_time}\n')


# CatBoost

print('CatBoost')
cb_model = cb.CatBoostClassifier(
    num_trees=5000,
    depth=6,
    learning_rate=0.01,
    loss_function='Logloss',
    auto_class_weights='Balanced',
    l2_leaf_reg=5,
    subsample=0.8,
    eval_metric='AUC',
    early_stopping_rounds=50,
    random_seed=seed,
    verbose=False
)

start = datetime.now()
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test))
cb_time = str(datetime.now() - start).split('.')[0]

cb_proba = cb_model.predict_proba(X_test)[:, 1]
CB_THRESHOLD = 0.5
cb_preds = (cb_proba >= CB_THRESHOLD).astype(int)
cb_auc = roc_auc_score(y_test, cb_proba)
cb_f1 = f1_score(y_test, cb_preds, average='macro')

print(classification_report(y_test, cb_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {cb_auc:.4f} | F1 macro: {cb_f1:.4f} | Итераций: {cb_model.best_iteration_} | Время: {cb_time}\n')


# важность признаков

FEATURE_GROUPS = {
    'Клиент':    ['lead_Категория и варианты выбора', 'lead_Квалификация лида', 'is_repeat_client',
                  'is_yur', 'lead_Проблема', 'lead_Вид оплаты', 'lead_region'],
    'Заказ':     ['lead_price', 'has_discount', 'n_product_categories',
                  'has_маска', 'has_наколенник', 'has_бандаж_шейный', 'has_повязка',
                  'has_напульсник', 'has_обувь', 'has_подушка', 'has_матрас',
                  'has_постельное', 'has_пояс', 'has_аксессуары', 'has_крем', 'has_бады'],
    'Маркетинг': ['has_yclid', 'has_promo', 'lead_source_category'],
    'Логистика': ['lead_Служба доставки', 'lead_Тариф Доставки'],
    'Время':     ['sale_hour', 'sale_day_of_week', 'sale_month',
                  'lead_created_hour', 'lead_created_day_of_week', 'lead_created_month',
                  'days_creation_to_sale'],
}

def drawGroupedImp(imp_series, model_name, save_path):
    group_imp = {}
    for group, cols in FEATURE_GROUPS.items():
        vals = imp_series[[c for c in cols if c in imp_series.index]]
        group_imp[group] = vals.sum()
    group_series = pd.Series(group_imp).sort_values()
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    _, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(group_series.index, group_series.values, color=colors[:len(group_series)])
    ax.set_xlabel('Суммарная важность')
    ax.set_title(f'Важность групп признаков - {model_name}')
    ax.set_xlim(0, group_series.max() * 1.15)
    for bar, val in zip(bars, group_series.values):
        ax.text(val + group_series.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

def drawTopFeatures(imp_series, model_name, save_path, top_n=20):
    top = imp_series.sort_values(ascending=True).tail(top_n)
    _, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), top.values, color='#4e79a7')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('Важность признака')
    ax.set_title(f'Топ-{top_n} признаков - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

def drawCoef(model, features, save_path):
    coef = pd.Series(model.coef_[0], index=features)
    coef_sorted = coef.reindex(coef.abs().sort_values(ascending=False).index)
    colors = ['steelblue' if c > 0 else 'tomato' for c in coef_sorted]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coef_sorted)), coef_sorted.values, color=colors)
    plt.yticks(range(len(coef_sorted)), coef_sorted.index, fontsize=8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Коэффициент (синий = + к выкупу, красный = - к выкупу)')
    plt.title('Feature Coefficients - LogReg-MK1')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

def drawGroupedCoef(model, features, save_path):
    coef = pd.Series(np.abs(model.coef_[0]), index=features)
    group_imp = {}
    for group, cols in FEATURE_GROUPS.items():
        vals = coef[[c for c in cols if c in coef.index]]
        group_imp[group] = vals.sum()
    group_series = pd.Series(group_imp).sort_values()
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    _, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(group_series.index, group_series.values, color=colors[:len(group_series)])
    ax.set_xlabel('Суммарный |коэффициент|')
    ax.set_title('Важность групп признаков - LogReg-MK1')
    ax.set_xlim(0, group_series.max() * 1.15)
    for bar, val in zip(bars, group_series.values):
        ax.text(val + group_series.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

print('Графики важности признаков...')

drawCoef(lr_model, features, img_path / 'LogReg-MK1_coef.png')
drawGroupedCoef(lr_model, features, img_path / 'LogReg-MK1_grouped.png')

rf_perimp = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=seed, scoring='roc_auc')
drawTopFeatures(pd.Series(rf_perimp.importances_mean, index=features), 'RF-MK1', img_path / 'RF-MK1_perimp.png')
drawGroupedImp(pd.Series(rf_perimp.importances_mean, index=features), 'RF-MK1', img_path / 'RF-MK1_grouped.png')

xgb_imp = pd.Series(xgb_model.feature_importances_, index=features)
drawTopFeatures(xgb_imp, 'XGBoost-MK1', img_path / 'XGBoost-MK1_features.png')
drawGroupedImp(xgb_imp, 'XGBoost-MK1', img_path / 'XGBoost-MK1_grouped.png')

cb_imp = pd.Series(cb_model.feature_importances_, index=features)
drawTopFeatures(cb_imp, 'CatBoost-MK1', img_path / 'CatBoost-MK1_features.png')
drawGroupedImp(cb_imp, 'CatBoost-MK1', img_path / 'CatBoost-MK1_grouped.png')

importances_rf = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print('\nТоп-10 признаков (RF gini):')
print(importances_rf.head(10).round(4).to_string())


# сохранение моделей

with open(model_path / 'LogReg-MK1.pkl', 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler, 'features': features}, f)
print('\nМодель сохранена: LogReg-MK1.pkl')

with open(model_path / 'RF-MK1.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'features': features}, f)
print('Модель сохранена: RF-MK1.pkl')

xgb_model.save_model(str(model_path / 'XGBoost-MK1.json'))
print('Модель сохранена: XGBoost-MK1.json')

cb_model.save_model(str(model_path / 'CatBoost-MK1.cbm'))
print('Модель сохранена: CatBoost-MK1.cbm')


# запись результатов

records_file = log_path / 'records.xls'
new_rows = pd.DataFrame([
    {'Model': 'LogReg-MK1',   'bestTime': lr_time,  'totalTime': lr_time,  'AUROCIndex': round(lr_auc, 4),  'F1Index': round(lr_f1, 4)},
    {'Model': 'RF-MK1',       'bestTime': rf_time,  'totalTime': rf_time,  'AUROCIndex': round(rf_auc, 4),  'F1Index': round(rf_f1, 4)},
    {'Model': 'XGBoost-MK1',  'bestTime': xgb_time, 'totalTime': xgb_time, 'AUROCIndex': round(xgb_auc, 4), 'F1Index': round(xgb_f1, 4)},
    {'Model': 'CatBoost-MK1', 'bestTime': cb_time,  'totalTime': cb_time,  'AUROCIndex': round(cb_auc, 4),  'F1Index': round(cb_f1, 4)},
])

if not records_file.exists():
    new_rows.to_excel(records_file, index=False)
else:
    existing = pd.read_excel(records_file)
    other = existing[~existing['Model'].isin(['LogReg-MK1', 'RF-MK1', 'XGBoost-MK1', 'CatBoost-MK1'])]
    updated = pd.concat([new_rows, other], ignore_index=True)
    updated.to_excel(records_file, index=False)

print(f'\nРезультаты записаны в {records_file}')
