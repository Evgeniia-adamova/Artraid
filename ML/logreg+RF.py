"""
скрипт для моделей Logistic Regression и Random Forest

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
    DAT[['lead_Квалификация лида']] = DAT[['lead_Квалификация лида']].fillna('Неизвестно')
    DAT[['lead_source_category']] = DAT[['lead_source_category']].fillna('Неизвестно')
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
drop2 = ['sale_date', 'closed_ts', 'received_ts', 'rejected_ts', 'returned_ts', 'days_to_outcome',
         'lead_Условный отказ', 'lead_Оплата МОП', 'is_paid_mop',
         'lead_Дата получения денег на Р/С', 'lead_Источник', 'lead_Дата создания сделки',
         'lead_Скидка', 'contact_Число сделок', 'handed_to_delivery_ts',
         # утечка: интервалы известны только после исхода сделки
         'days_sale_to_received', 'days_sale_to_rejected', 'days_sale_to_returned', 'days_sale_to_closed',
         # утечка: LTV обновляется после выкупа
         'contact_LTV',
         # модель 1: прогноз в момент создания заказа - логистика ещё не известна
         'days_sale_to_handed', 'days_handed_to_issued_pvz', 'delivery_group',
         'issued_or_pvz_ts', 'days_received_to_issued']

DATA = CLEAN_DATA.drop(columns=[c for c in drop1 + drop2 if c in CLEAN_DATA.columns])
del CLEAN_DATA
gc.collect()

DATA = bool2flag(DATA)
DATA = sig3OutlDetector(DATA, ['lead_price'])
DATA = Nan2Adequate(DATA)
DATA = datetime2EpSec(DATA)
DATA = CategoricalOrdinal(DATA)

# добиваем оставшиеся NaN в числовых колонках (timestamps без данных, delivery_days и др.)
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


# логистическая регрессия

print('\nLogistic Regression')
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
LR_THRESHOLD = 0.5  # порог 0.5: balanced даёт recall невыкупов ~78%; при 0.6 - больше невыкупов, меньше выкупов
lr_preds = (lr_proba >= LR_THRESHOLD).astype(int)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_preds, average='macro')

print(classification_report(y_test, lr_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {lr_auc:.4f}')
print(f'F1 macro: {lr_f1:.4f}')
print(f'Время: {lr_time}')


# Random Forest

print('\nRandom Forest')
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
RF_THRESHOLD = 0.5  # порог 0.5: recall невыкупов 84%, recall выкупов 65%; для макс. ловли невыкупов - 0.6
rf_preds = (rf_proba >= RF_THRESHOLD).astype(int)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_f1 = f1_score(y_test, rf_preds, average='macro')

print(classification_report(y_test, rf_preds, target_names=['0 (не выкуп)', '1 (выкуп)']))
print(f'AUC: {rf_auc:.4f}')
print(f'F1 macro: {rf_f1:.4f}')
print(f'Время: {rf_time}')


# важность признаков

def drawPerImp(perimp, features, model_name, save_path):
    sorted_idx = perimp.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), perimp.importances_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx], fontsize=8)
    plt.xlabel('Permutation Importance')
    plt.title(f'Permutation Importance {model_name}')
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
    plt.title('Feature Coefficients - LogReg-baseline')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

print('\nКоэффициенты лог. регрессии...')
drawCoef(lr_model, features, img_path / 'LogReg-baseline_coef.png')

print('Permutation Importance для Random Forest...')
rf_perimp = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=seed, scoring='roc_auc')
drawPerImp(rf_perimp, features, 'RF-baseline-500', img_path / 'RF-baseline-500_perimp.png')

# группы признаков (по target_dtypes.yaml)
FEATURE_GROUPS = {
    'Клиент': ['lead_Категория и варианты выбора', 'lead_Квалификация лида', 'is_repeat_client', 'is_yur', 'lead_Проблема', 'lead_Вид оплаты', 'lead_region'],
    'Заказ': ['lead_price', 'has_discount', 'n_product_categories', 'price_group', 'has_маска', 'has_наколенник', 'has_бандаж_шейный', 'has_повязка', 'has_напульсник', 'has_обувь', 'has_подушка', 'has_матрас', 'has_постельное', 'has_пояс', 'has_аксессуары', 'has_крем', 'has_бады'],
    'Маркетинг': ['has_yclid', 'has_promo', 'lead_source_category'],
    'Логистика': ['lead_Служба доставки', 'lead_Тариф Доставки'],
    'Время': ['sale_ts', 'sale_hour', 'sale_day_of_week', 'sale_month', 'lead_created_hour', 'lead_created_day_of_week', 'lead_created_month', 'days_creation_to_sale'],
}

def drawGroupedPerImp(perimp_means, features, model_name, save_path):
    imp = pd.Series(perimp_means, index=features)
    group_imp = {}
    for group, cols in FEATURE_GROUPS.items():
        vals = imp[[c for c in cols if c in imp.index]]
        group_imp[group] = vals.sum()
    group_series = pd.Series(group_imp).sort_values()
    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    _, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(group_series.index, group_series.values, color=colors[:len(group_series)])
    ax.set_xlabel('Суммарный Permutation Importance (AUC)')
    ax.set_title(f'Важность групп признаков - {model_name}')
    ax.set_xlim(0, group_series.max() * 1.15)
    for bar, val in zip(bars, group_series.values):
        ax.text(val + group_series.max() * 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=9)
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
    ax.set_xlabel('Суммарный коэффициент')
    ax.set_title('Важность групп признаков - LogReg-baseline')
    ax.set_xlim(0, group_series.max() * 1.15)
    for bar, val in zip(bars, group_series.values):
        ax.text(val + group_series.max() * 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'График сохранен: {save_path.name}')

print('\nГрупповая важность признаков...')
drawGroupedPerImp(rf_perimp.importances_mean, features, 'RF-baseline-500', img_path / 'RF-baseline-500_grouped.png')
drawGroupedCoef(lr_model, features, img_path / 'LogReg-baseline_grouped.png')

importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print('\nТоп-10 признаков (Random Forest, gini):')
print(importances.head(10).round(4).to_string())


# сохранение моделей

with open(model_path / 'LogReg-baseline.pkl', 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler, 'features': features}, f)
print('\nМодель сохранена: LogReg-baseline.pkl')

with open(model_path / 'RF-baseline-500.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'features': features}, f)
print('Модель сохранена: RF-baseline-500.pkl')


# запись результатов в таблицу

records_file = log_path / 'records.xls'
new_rows = pd.DataFrame([
    {'Model': 'LogReg-baseline', 'bestTime': lr_time, 'totalTime': lr_time, 'AUROCIndex': round(lr_auc, 4), 'F1Index': round(lr_f1, 4)},
    {'Model': 'RF-baseline-500', 'bestTime': rf_time, 'totalTime': rf_time, 'AUROCIndex': round(rf_auc, 4), 'F1Index': round(rf_f1, 4)},
])

if not records_file.exists():
    new_rows.to_excel(records_file, index=False)
else:
    existing = pd.read_excel(records_file)
    # убираем старые строки этих двух моделей, вставляем свежие в начало
    other = existing[~existing['Model'].isin(['LogReg-baseline', 'RF-baseline-500'])]
    updated = pd.concat([new_rows, other], ignore_index=True)
    updated.to_excel(records_file, index=False)

print(f'\nРезультаты записаны в {records_file}')
