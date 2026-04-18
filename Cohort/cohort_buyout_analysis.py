from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

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

EARLY_COLORS = ["#AED9E0", "#73C2D4", "#45B7D1", "#1B9AAA", "#13678A", "#134074"]
LATE_COLORS  = ["#F5B7B1", "#F1948A", "#E74C3C", "#C0392B", "#A93226", "#922B21"]

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

MAX_DAY = 30
HEATMAP_WEEKS = 36
LINE_COHORTS_PER_SIDE = 4
KEY_DAYS = (7, 14, 30)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data_preparation" / "data" / "clean" / "clean_data.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"

REQUIRED_COLUMNS = [
    "sale_date",
    "buyout_flag",
    "received_ts",
    "rejected_ts",
    "returned_ts",
    "lead_region",
    "lead_source_category"
]
DATE_COLUMNS = ["sale_date", "received_ts", "rejected_ts", "returned_ts"]
BUSINESS_KEY_CANDIDATES = ["lead_id", "order_id", "lead_Номер заказа на сайте"]


def to_truthy_mask(series: pd.Series) -> pd.Series:
    """Convert heterogeneous bool-like values to a reliable boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)

    normalized = series.astype("string").str.strip().str.lower().fillna("false")
    return normalized.isin({"true", "1", "yes", "y", "да"})


def format_pct(value: float | int | np.floating | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные поля: {missing}")


def choose_business_key(df: pd.DataFrame) -> str | None:
    for column in BUSINESS_KEY_CANDIDATES:
        if column in df.columns:
            return column
    return None


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    columns_to_load = list(dict.fromkeys(REQUIRED_COLUMNS + BUSINESS_KEY_CANDIDATES))
    df = pd.read_excel(path, usecols=lambda col: col in columns_to_load)
    ensure_columns(df, REQUIRED_COLUMNS)

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["sale_date"] = df["sale_date"].dt.floor("D")

    rows_before = len(df)
    rows_after_unknown = rows_before  # outcome_unknown уже отфильтрован в data_preparation

    df = df.loc[df["sale_date"].notna()].copy()
    rows_after_sale_date = len(df)

    business_key = choose_business_key(df)
    duplicates_removed = 0
    if business_key is not None:
        duplicates_removed = int(df.duplicated(subset=[business_key]).sum())
        df = df.drop_duplicates(subset=[business_key], keep="first").copy()
    else:
        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()

    df["is_buyout"] = to_truthy_mask(df["buyout_flag"])
    df["buyout_event_date"] = df["received_ts"].where(df["is_buyout"])
    df["non_buyout_event_date"] = pd.concat([df["rejected_ts"], df["returned_ts"]], axis=1).min(axis=1)
    df["result_event_date"] = df["buyout_event_date"].fillna(df["non_buyout_event_date"])

    df["result_event_type"] = np.select(
        [
            df["buyout_event_date"].notna(),
            df["rejected_ts"].notna() & (df["returned_ts"].isna() | (df["rejected_ts"] <= df["returned_ts"])),
            df["returned_ts"].notna(),
        ],
        ["buyout_received", "rejected", "returned"],
        default="unknown",
    )

    days_to_buyout = np.floor((df["buyout_event_date"] - df["sale_date"]).dt.total_seconds() / 86400)
    df["days_to_buyout"] = pd.Series(days_to_buyout, index=df.index, dtype="Float64")
    df.loc[df["days_to_buyout"] < 0, "days_to_buyout"] = pd.NA

    result_days = np.floor((df["result_event_date"] - df["sale_date"]).dt.total_seconds() / 86400)
    df["days_since_sale"] = pd.Series(result_days, index=df.index, dtype="Float64")
    df.loc[df["days_since_sale"] < 0, "days_since_sale"] = pd.NA

    observed_cutoff = pd.concat(
        [df["sale_date"], df["received_ts"], df["rejected_ts"], df["returned_ts"]],
        ignore_index=True,
    ).max()
    age_at_cutoff = np.floor((observed_cutoff - df["sale_date"]).dt.total_seconds() / 86400)
    df["age_at_cutoff_days"] = pd.Series(age_at_cutoff, index=df.index, dtype="Int64")

    # Недельные когорты с началом недели в понедельник.
    df["cohort_week"] = df["sale_date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    df["cohort_month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()

    meta = {
        "rows_before": rows_before,
        "rows_after_unknown": rows_after_unknown,
        "rows_after_sale_date": rows_after_sale_date,
        "rows_after_dedup": len(df),
        "excluded_outcome_unknown": rows_before - rows_after_unknown,
        "dropped_missing_sale_date": rows_after_unknown - rows_after_sale_date,
        "duplicates_removed": duplicates_removed,
        "business_key": business_key or "full_row",
        "buyout_orders": int(df["buyout_event_date"].notna().sum()),
        "buyout_flag_true": int(df["is_buyout"].sum()),
        "buyout_without_received_ts": int((df["is_buyout"] & df["buyout_event_date"].isna()).sum()),
        "weekly_cohorts": int(df["cohort_week"].nunique()),
        "monthly_cohorts": int(df["cohort_month"].nunique()),
        "sale_date_min": df["sale_date"].min(),
        "sale_date_max": df["sale_date"].max(),
        "observed_cutoff": observed_cutoff,
    }
    return df, meta


def build_cohort_metrics(df: pd.DataFrame, cohort_col: str, max_day: int = MAX_DAY) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for cohort_date, cohort_df in df.groupby(cohort_col, sort=True):
        total_orders = int(len(cohort_df))
        buyout_days = cohort_df["days_to_buyout"]
        age_days = cohort_df["age_at_cutoff_days"].astype("Int64")

        for day in range(max_day + 1):
            observed_mask = age_days >= day
            orders_observed = int(observed_mask.fillna(False).sum())
            if orders_observed == 0:
                orders_bought_out = 0
                buyout_rate_cum = np.nan
            else:
                orders_bought_out = int((observed_mask & buyout_days.notna() & (buyout_days <= day)).sum())
                buyout_rate_cum = orders_bought_out / orders_observed

            rows.append(
                {
                    "cohort_date": cohort_date,
                    "cohort_period": "week" if cohort_col == "cohort_week" else "month",
                    "days_since_sale": day,
                    "orders_total": total_orders,
                    "orders_observed": orders_observed,
                    "orders_bought_out": orders_bought_out,
                    "observation_share": orders_observed / total_orders if total_orders else np.nan,
                    "buyout_rate_cum": buyout_rate_cum,
                }
            )

    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="cohort_date", columns="days_since_sale", values="buyout_rate_cum").sort_index()

    day0 = long_df.loc[long_df["days_since_sale"] == 0, ["cohort_date", "orders_total"]].set_index("cohort_date")
    day_max = long_df.loc[long_df["days_since_sale"] == max_day].set_index("cohort_date")

    summary = pd.DataFrame(index=pivot.index)
    summary["orders_total"] = day0["orders_total"]
    summary[f"orders_observed_day_{max_day}"] = day_max["orders_observed"]
    summary[f"observation_share_day_{max_day}"] = day_max["observation_share"]
    summary[f"buyout_rate_day_{max_day}"] = day_max["buyout_rate_cum"]
    summary["fully_mature"] = summary[f"observation_share_day_{max_day}"].fillna(0).ge(1)

    return long_df, pivot, summary


def select_line_cohorts(summary: pd.DataFrame, skip_first: int = 1) -> list[pd.Timestamp]:
    mature = summary.loc[summary["fully_mature"]].sort_index()
    if mature.empty:
        return list(summary.sort_index().tail(min(8, len(summary))).index)

    mature_sorted = list(mature.index)
    
    # Пропускаем первые skip_first когорт (часто в них мало данных)
    early_start = min(skip_first, len(mature_sorted) - LINE_COHORTS_PER_SIDE)
    early_cohorts = mature_sorted[early_start:early_start+LINE_COHORTS_PER_SIDE]
    
    # Выбираем поздние когорты
    late_cohorts = mature_sorted[-LINE_COHORTS_PER_SIDE:]
    
    # Объединяем и убираем дубликаты
    selected = early_cohorts + late_cohorts
    result = []
    for cohort in selected:
        if cohort not in result:
            result.append(cohort)
    return result


def get_top_regions(df: pd.DataFrame, n: int = 10) -> list[str]:
    """Получить топ-N регионов по количеству заказов."""
    return df["lead_region"].value_counts().head(n).index.tolist()


def plot_heatmap(cohort_pivot: pd.DataFrame, output_path: Path) -> None:
    heatmap_data = cohort_pivot.tail(HEATMAP_WEEKS).copy()
    if heatmap_data.empty:
        return

    heatmap_pct = (heatmap_data * 100).round(1)
    heatmap_pct.index = heatmap_pct.index.strftime("%Y-%m-%d")

    plt.figure(figsize=(20, max(10, 0.4 * len(heatmap_pct))))
    ax = sns.heatmap(
        heatmap_pct,
        mask=heatmap_pct.isna(),
        annot=True,
        fmt=".1f",
        cmap=ARTRAID_CMAP,
        linewidths=0.3,
        linecolor=LIGHT,
        cbar_kws={"label": "Cumulative buyout rate, %"},
        annot_kws={"size": 10, "color": LIGHT},
        vmin=0,
        vmax=max(75, float(np.nanmax(heatmap_pct.values)) if np.isfinite(np.nanmax(heatmap_pct.values)) else 75),
    )

    ax.set_title("Недельные когорты: накопительный выкуп по дням (последние недели, D0–D30)",
                 fontsize=16,
                 fontweight="bold",
                 color=DARK,
                 pad=16)
    ax.set_xlabel("Days after order", fontsize=12, color=TEXT)
    ax.set_ylabel("Cohort week start", fontsize=12, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_heatmap_first(cohort_pivot: pd.DataFrame, output_path: Path) -> None:
    heatmap_data = cohort_pivot.head(HEATMAP_WEEKS).copy()
    if heatmap_data.empty:
        return

    heatmap_pct = (heatmap_data * 100).round(1)
    heatmap_pct.index = heatmap_pct.index.strftime("%Y-%m-%d")

    plt.figure(figsize=(20, max(10, 0.4 * len(heatmap_pct))))
    ax = sns.heatmap(
        heatmap_pct,
        mask=heatmap_pct.isna(),
        annot=True,
        fmt=".1f",
        cmap=ARTRAID_CMAP,
        linewidths=0.3,
        linecolor=LIGHT,
        cbar_kws={"label": "Cumulative buyout rate, %"},
        annot_kws={"size": 10, "color": LIGHT},
        vmin=0,
        vmax=max(75, float(np.nanmax(heatmap_pct.values)) if np.isfinite(np.nanmax(heatmap_pct.values)) else 75),
    )

    ax.set_title("Недельные когорты: накопительный выкуп по дням (первые недели, D0–D30)",
                 fontsize=16,
                 fontweight="bold",
                 color=DARK,
                 pad=16)
    ax.set_xlabel("Days after order", fontsize=12, color=TEXT)
    ax.set_ylabel("Cohort week start", fontsize=12, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_curves(cohort_pivot: pd.DataFrame, selected_cohorts: list[pd.Timestamp], output_path: Path) -> pd.Series:
    mature_mask = cohort_pivot.notna().all(axis=1)
    avg_curve = cohort_pivot.loc[mature_mask].mean(axis=0, skipna=True) if mature_mask.any() else cohort_pivot.mean(axis=0, skipna=True)
    
    # Разделяем когорты на ранние и поздние
    midpoint = len(selected_cohorts) // 2
    early_cohorts = selected_cohorts[:midpoint]
    late_cohorts = selected_cohorts[midpoint:]
    
    # Цвета: ранние - голубые (от светлого к темному), поздние - красные (от светлого к темному)
    early_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(early_cohorts)))
    late_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(late_cohorts)))

    plt.figure(figsize=(12.5, 7))
    
    for i, cohort_date in enumerate(early_cohorts):
        if cohort_date not in cohort_pivot.index:
            continue
        row = cohort_pivot.loc[cohort_date]
        plt.plot(row.index, row.values * 100, marker="o", linewidth=2, label=cohort_date.strftime("%Y-%m-%d"), color=early_colors[i])

    for i, cohort_date in enumerate(late_cohorts):
        if cohort_date not in cohort_pivot.index:
            continue
        row = cohort_pivot.loc[cohort_date]
        plt.plot(row.index, row.values * 100, marker="o", linewidth=2, label=cohort_date.strftime("%Y-%m-%d"), color=late_colors[i])

    plt.plot(avg_curve.index, avg_curve.values * 100, linestyle="--", linewidth=3, color=DARK, label="Средняя по всем когортам")
    for day in KEY_DAYS:
        plt.axvline(day, color="gray", linestyle=":", linewidth=1)

    plt.title("Кривые накопительного выкупа: ранние vs поздние когорты")
    plt.xlabel("Days after order")
    plt.ylabel("Cumulative buyout rate, %")
    plt.xlim(0, MAX_DAY)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend(title="Когорта", ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return avg_curve


def plot_cohort_sizes(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    labels = [idx.strftime("%Y-%m-%d") for idx in summary.index]
    sizes = summary["orders_total"].values
    n = len(labels)

    bar_colors = [GRADIENT_POOL[int(i)] for i in np.linspace(0, len(GRADIENT_POOL)-1, n)]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(labels, sizes, color=bar_colors, edgecolor=BG, linewidth=0.5)

    ax.set_title(title, fontsize=16, fontweight="bold", color=DARK, pad=16)
    ax.set_xlabel("Когорта", fontsize=12, color=TEXT)
    ax.set_ylabel("Количество заказов", fontsize=12, color=TEXT)

    # Показываем каждую 4-ю подпись, чтобы не было каши
    plt.xticks(rotation=70, ha="right")
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 4 != 0:
            label.set_visible(False)

    ax.grid(axis="y", alpha=0.15, color=GRAY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=TEXT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close()

def plot_regional_daily_comparison(region_series: dict[str, pd.Series], output_path: Path) -> None:
    """Сравнительные кривые среднего выкупа по топ регионам."""
    if not region_series:
        return

    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(region_series)))
    
    for i, (region, curve) in enumerate(sorted(region_series.items())):
        curve_clean = curve.dropna().sort_index()
        plt.plot(curve_clean.index, curve_clean.values * 100, marker="o", linewidth=2.5, label=region, color=colors[i])
    
    for day in KEY_DAYS:
        plt.axvline(day, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    
    plt.title("Накопительный выкуп: сравнение региональных когорт")
    plt.xlabel("Days after order")
    plt.ylabel("Cumulative buyout rate, %")
    plt.xlim(0, MAX_DAY)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    plt.legend(title="Регион", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_regional_final_buyout_comparison(regions_data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Сравнение финальных выкупов по регионам как столбчатая диаграмма."""
    final_rates = {}
    region_orders = {}
    
    for region, summary in regions_data.items():
        mature = summary.loc[summary["fully_mature"]]
        if not mature.empty:
            avg_rate = mature[f"buyout_rate_day_{MAX_DAY}"].mean()
            final_rates[region] = avg_rate * 100
            region_orders[region] = int(mature["orders_total"].sum())
    
    if not final_rates:
        return
    
    regions = sorted(final_rates.keys(), key=lambda x: final_rates[x], reverse=True)
    rates = [final_rates[r] for r in regions]
    
    plt.figure(figsize=(14, 7))
    bars = plt.bar(regions, rates, color=plt.cm.Spectral(np.linspace(0.2, 0.8, len(regions))))
    plt.axhline(np.mean(rates), color=DARK, linestyle="--", linewidth=1.5, label=f"Среднее: {np.mean(rates):.1f}%")
    
    for i, (region, rate) in enumerate(zip(regions, rates)):
        plt.text(i, rate + 1, f"{rate:.1f}%\n({region_orders.get(region, 0):,})", 
                ha="center", va="bottom", fontsize=9)
    
    plt.title(f"Финальный buyout rate по регионам (D{MAX_DAY}, только зрелые когорты)")
    plt.ylabel(f"Buyout rate, %")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_regional_heatmap(region_pivots: dict[str, pd.DataFrame], region_summaries: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Хитмап выкупа по неделям и регионам."""
    # Берем только зрелые когорты для каждого региона
    heatmap_rows = {}
    for region, pivot in region_pivots.items():
        summary = region_summaries[region]
        mature_cohorts = summary.loc[summary["fully_mature"]].index
        mature_pivot = pivot.loc[pivot.index.isin(mature_cohorts)]
        if not mature_pivot.empty:
            avg_by_day = mature_pivot.mean(axis=0, skipna=True)
            heatmap_rows[region] = avg_by_day * 100
    
    if not heatmap_rows:
        return
    
    heatmap_df = pd.DataFrame(heatmap_rows).T
    heatmap_df = heatmap_df.iloc[:, :31]  # До дня 30
    
    plt.figure(figsize=(18, 8))
    sns.heatmap(heatmap_df,
                annot=True,
                fmt=".1f",
                cmap=ARTRAID_CMAP,
                linewidths=0.3,
                linecolor=LIGHT,
                cbar_kws={"label": "Avg buyout rate, %"},
                annot_kws={"size": 10, "color": LIGHT},
                vmin=0, vmax=max(75, float(heatmap_df.max().max())))
    plt.title("Средний накопительный выкуп по регионам и дням (D0–D30)",
              fontsize=16,
              fontweight="bold",
              color=DARK,
              pad=16)
    plt.xlabel("Days after order", fontsize=12, color=TEXT)
    plt.ylabel("Region", fontsize=12, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_final_buyout(summary: pd.DataFrame, output_path: Path, title: str) -> pd.DataFrame:
    plot_df = summary.copy().sort_index()
    mature_df = plot_df.loc[plot_df["fully_mature"]].copy()
    if mature_df.empty:
        return mature_df

    mature_df["buyout_pct"] = mature_df[f"buyout_rate_day_{MAX_DAY}"] * 100
    labels = [idx.strftime("%Y-%m-%d") for idx in mature_df.index]

    q1 = mature_df["buyout_pct"].quantile(0.25)
    q3 = mature_df["buyout_pct"].quantile(0.75)
    colors = []
    for value in mature_df["buyout_pct"]:
        if value >= q3:
            colors.append("#134074")
        elif value <= q1:
            colors.append(RED)
        else:
            colors.append("#45B7D1")

    fig, ax = plt.subplots(figsize=(14, 6.5))

    ax.bar(labels, mature_df["buyout_pct"], color=colors, edgecolor=BG, linewidth=0.5)

    ax.axhline(
        mature_df["buyout_pct"].mean(),
        color=DARK, linestyle="--", linewidth=1.5,
        label=f"Среднее: {mature_df['buyout_pct'].mean():.1f}%"
    )

    ax.set_title(title, fontsize=16, fontweight="bold", color=DARK, pad=16)
    ax.set_xlabel("Когорта", fontsize=12, color=TEXT)
    ax.set_ylabel(f"Buyout rate к D{MAX_DAY}, %", fontsize=12, color=TEXT)

    plt.xticks(rotation=70, ha="right")

    ax.grid(axis="y", alpha=0.15, color=GRAY)
    ax.legend(loc="upper right", fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=TEXT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close()
    return mature_df



def plot_regional_daily_comparison(region_series: dict[str, pd.Series], output_path: Path) -> None:
    """Сравнительные кривые среднего выкупа по топ регионам."""
    if not region_series:
        return

    # Расширенная палитра для 10+ линий — контрастные цвета
    REGIONAL_COLORS = [
        "#0B2545",  # тёмно-синий
        "#1B9AAA",  # бирюзовый
        "#C0392B",  # красный
        "#E67E22",  # оранжевый
        "#2D936C",  # зелёный
        "#7D3C98",  # фиолетовый
        "#45B7D1",  # светло-бирюзовый
        "#D4AC0D",  # золотой
        "#134074",  # средне-синий
        "#A93226",  # тёмно-красный
        "#1E8449",  # тёмно-зелёный
        "#6C3483",  # тёмно-фиолетовый
    ]

    n = len(region_series)
    colors = REGIONAL_COLORS[:n]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (region, curve) in enumerate(sorted(region_series.items())):
        curve_clean = curve.dropna().sort_index()
        ax.plot(
            curve_clean.index, curve_clean.values * 100,
            marker="o", markersize=4, linewidth=2.5,
            label=region, color=colors[i]
        )

    for day in KEY_DAYS:
        ax.axvline(day, color=GRAY, linestyle=":", linewidth=1, alpha=0.5)

    ax.set_title(
        "Накопительный выкуп: сравнение региональных когорт",
        fontsize=16, fontweight="bold", color=DARK, pad=16
    )
    ax.set_xlabel("Дней после заказа", fontsize=12, color=TEXT)
    ax.set_ylabel("Накопительный выкуп, %", fontsize=12, color=TEXT)
    ax.set_xlim(0, MAX_DAY)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.15, color=GRAY)
    ax.legend(title="Регион", fontsize=10, title_fontsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=TEXT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close()

def plot_regional_final_buyout_comparison(regions_data: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Сравнение финальных выкупов по регионам как столбчатая диаграмма."""
    final_rates = {}
    region_orders = {}
    
    for region, summary in regions_data.items():
        mature = summary.loc[summary["fully_mature"]]
        if not mature.empty:
            avg_rate = mature[f"buyout_rate_day_{MAX_DAY}"].mean()
            final_rates[region] = avg_rate * 100
            region_orders[region] = int(mature["orders_total"].sum())
    
    if not final_rates:
        return
    
    regions = sorted(final_rates.keys(), key=lambda x: final_rates[x], reverse=True)
    rates = [final_rates[r] for r in regions]

    n = len(regions)
    bar_colors = [GRADIENT_POOL[int(i)] for i in np.linspace(0, len(GRADIENT_POOL) - 1, n)]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.bar(regions, rates, color=bar_colors, edgecolor=BG, linewidth=0.5)

    ax.axhline(
        np.mean(rates),
        color=DARK, linestyle="--", linewidth=1.5,
        label=f"Среднее: {np.mean(rates):.1f}%"
    )

    ax.set_title(
        "Сравнение финального buyout rate по регионам",
        fontsize=16, fontweight="bold", color=DARK, pad=16
    )
    ax.set_xlabel("Регион", fontsize=12, color=TEXT)
    ax.set_ylabel(f"Buyout rate к D{MAX_DAY}, %", fontsize=12, color=TEXT)

    plt.xticks(rotation=70, ha="right")

    ax.grid(axis="y", alpha=0.15, color=GRAY)
    ax.legend(loc="upper right", fontsize=10)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    ax.tick_params(colors=TEXT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close()

def plot_regional_heatmap(region_pivots: dict[str, pd.DataFrame], region_summaries: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Хитмап выкупа по неделям и регионам."""
    # Берем только зрелые когорты для каждого региона
    heatmap_rows = {}
    for region, pivot in region_pivots.items():
        summary = region_summaries[region]
        mature_cohorts = summary.loc[summary["fully_mature"]].index
        mature_pivot = pivot.loc[pivot.index.isin(mature_cohorts)]
        if not mature_pivot.empty:
            avg_by_day = mature_pivot.mean(axis=0, skipna=True)
            heatmap_rows[region] = avg_by_day * 100
    
    if not heatmap_rows:
        return
    
    heatmap_df = pd.DataFrame(heatmap_rows).T
    heatmap_df = heatmap_df.iloc[:, :31]  # До дня 30
    
    plt.figure(figsize=(18, 8))
    sns.heatmap(heatmap_df,
                annot=True,
                fmt=".1f",
                cmap=ARTRAID_CMAP,
                linewidths=0.3,
                linecolor=LIGHT,
                cbar_kws={"label": "Avg buyout rate, %"},
                annot_kws={"size": 10, "color": LIGHT},
                vmin=0, vmax=max(75, float(heatmap_df.max().max())))

    plt.title("Средний накопительный выкуп по регионам и дням (D0–D30)")
    plt.xlabel("Days after order")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()

# Тепловая карта: источник лида х месяц продажи
def plot_source_month_heatmap(df: pd.DataFrame, output_path: Path,
                               min_orders: int = 500) -> None:
    """Тепловая карта: источник лида × месяц продажи."""
    df_cohort = df.copy()
    df_cohort["sale_month"] = df_cohort["sale_date"].dt.to_period("M").astype(str)

    cohort = (
        df_cohort.groupby(["lead_source_category", "sale_month"])
        .agg(
            buyout_count=("is_buyout", "sum"),
            total_count=("is_buyout", "count")
        )
        .reset_index()
    )

    cohort["buyout_rate_pct"] = (
        cohort["buyout_count"] / cohort["total_count"] * 100
    ).round(1)

    pivot = cohort.pivot(
        index="lead_source_category",
        columns="sale_month",
        values="buyout_rate_pct"
    ).sort_index(axis=1)

    # Фильтруем источники с достаточным объёмом
    total_by_source = cohort.groupby("lead_source_category")["total_count"].sum()
    valid_sources = total_by_source[total_by_source >= min_orders].sort_values(ascending=False).index
    pivot = pivot.loc[valid_sources]

    if pivot.empty:
        return

    flat = pivot.values.flatten()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return
    VMAX = max(75.0, float(np.max(flat)))
    VMIN = max(0.0, float(np.min(flat)) * 0.9)

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.9),
                                     max(4, len(pivot) * 0.8)))

    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap=ARTRAID_CMAP,
        linewidths=0.3,
        linecolor=LIGHT,
        cbar_kws={"label": "Buyout rate, %"},
        annot_kws={"size": 11},
        vmin=VMIN, vmax=VMAX,
        ax=ax,
    )

    ax.set_title(
        "Доля выкупа: источник лида × месяц",
        fontsize=16, fontweight="bold", color=DARK, pad=16
    )
    ax.set_xlabel("Месяц продажи", fontsize=12, color=TEXT)
    ax.set_ylabel("Источник лида", fontsize=12, color=TEXT)
    ax.tick_params(axis="x", rotation=45, labelsize=11, colors=TEXT)
    ax.tick_params(axis="y", labelsize=11, colors=TEXT)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close()


def build_summary_report(
    meta: dict[str, object],
    weekly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    avg_curve: pd.Series,
    regional_data: dict[str, dict[str, object]] | None = None,
) -> str:
    def fmt_pct(v: float | int | np.floating | None) -> str:
        if v is None or pd.isna(v):
            return "n/a"
        return f"{float(v) * 100:.1f}%"

    def fmt_int(v: float | int | np.floating | None) -> str:
        if v is None or pd.isna(v):
            return "n/a"
        return f"{int(v):,}".replace(",", " ")

    def render_rank_table(df: pd.DataFrame, rate_col: str, n: int = 5) -> str:
        if df.empty:
            return "_Нет данных._"

        cols = ["orders_total", rate_col]
        view = df[cols].copy().sort_values(rate_col, ascending=False).head(n)

        lines = [
            "| Когорта | Размер когорты | Buyout к D30 |",
            "|---|---:|---:|",
        ]
        for idx, row in view.iterrows():
            lines.append(
                f"| {idx.strftime('%Y-%m-%d')} | {fmt_int(row['orders_total'])} | {fmt_pct(row[rate_col])} |"
            )
        return "\n".join(lines)

    avg_curve = avg_curve.dropna().sort_index()

    final_avg_rate = float(avg_curve.loc[MAX_DAY]) if MAX_DAY in avg_curve.index else float(avg_curve.iloc[-1])
    day_80 = int(avg_curve[avg_curve >= final_avg_rate * 0.80].index.min()) if final_avg_rate > 0 else 0
    day_95 = int(avg_curve[avg_curve >= final_avg_rate * 0.95].index.min()) if final_avg_rate > 0 else 0

    daily_increment = avg_curve.diff().fillna(avg_curve.iloc[0])
    peak_increment_day = int(daily_increment.idxmax()) if not daily_increment.empty else 0

    mature_weekly = weekly_summary.loc[weekly_summary["fully_mature"]].copy().sort_index()
    mature_monthly = monthly_summary.loc[monthly_summary["fully_mature"]].copy().sort_index()

    trend_note = "Недостаточно зрелых когорт для оценки тренда."
    trend_delta_pp = np.nan
    if len(mature_weekly) >= 8:
        half = len(mature_weekly) // 2
        recent = mature_weekly.tail(min(8, half))[f"buyout_rate_day_{MAX_DAY}"]
        previous = mature_weekly.head(min(8, half))[f"buyout_rate_day_{MAX_DAY}"]
        if len(previous) >= 4:
            trend_delta_pp = (recent.mean() - previous.mean()) * 100
            if trend_delta_pp >= 2:
                trend_note = f"У поздних зрелых когорт есть улучшение примерно на {trend_delta_pp:.1f} п.п. к D{MAX_DAY}."
            elif trend_delta_pp <= -2:
                trend_note = f"У поздних зрелых когорт заметно ухудшение примерно на {abs(trend_delta_pp):.1f} п.п. к D{MAX_DAY}."
            else:
                trend_note = f"Сильного тренда не видно: разница новых и предыдущих зрелых когорт около {trend_delta_pp:.1f} п.п."

    monthly_note = "_Нет зрелых месячных когорт для сравнения._"
    if not mature_monthly.empty:
        best_month = mature_monthly[f"buyout_rate_day_{MAX_DAY}"].idxmax()
        worst_month = mature_monthly[f"buyout_rate_day_{MAX_DAY}"].idxmin()
        monthly_note = (
            f"Лучший месяц: **{best_month.strftime('%Y-%m')}** ({fmt_pct(mature_monthly.loc[best_month, f'buyout_rate_day_{MAX_DAY}'])}); "
            f"самый слабый: **{worst_month.strftime('%Y-%m')}** ({fmt_pct(mature_monthly.loc[worst_month, f'buyout_rate_day_{MAX_DAY}'])})."
        )

    top_weekly = mature_weekly.nlargest(5, f"buyout_rate_day_{MAX_DAY}")
    bottom_weekly = mature_weekly.nsmallest(5, f"buyout_rate_day_{MAX_DAY}")

    regional_section = ""
    if regional_data:
        regional_section = "\n\n## 7) Анализ по регионам (топ 10)\n"
        for region, data in sorted(regional_data.items(), key=lambda x: x[1].get("avg_buyout_rate", 0), reverse=True):
            best_rate = data["best_buyout_rate"]
            avg_rate = data["avg_buyout_rate"]
            regional_section += f"\n**{region}**: среднее {fmt_pct(avg_rate)}, максимум {fmt_pct(best_rate)} (заказов: {fmt_int(data['total_orders'])})"

    lines = [
        "# Когортный анализ выкупа",
        "",
        "## 1) Обзор датасета",
        "",
        f"- Входной файл: `{INPUT_PATH}`",
        f"- Строк до фильтрации: **{fmt_int(meta['rows_before'])}**",
        f"- Исключено `outcome_unknown=True`: **{fmt_int(meta['excluded_outcome_unknown'])}**",
        f"- Удалено строк без `sale_date`: **{fmt_int(meta['dropped_missing_sale_date'])}**",
        f"- Удалено дубликатов по ключу `{meta['business_key']}`: **{fmt_int(meta['duplicates_removed'])}**",
        f"- Строк после очистки: **{fmt_int(meta['rows_after_dedup'])}**",
        f"- Недельных когорт: **{fmt_int(meta['weekly_cohorts'])}**",
        f"- Месячных когорт: **{fmt_int(meta['monthly_cohorts'])}**",
        f"- Выкупов с подтвержденным `received_ts`: **{fmt_int(meta['buyout_orders'])}**",
        f"- Всего `buyout_flag=True`: **{fmt_int(meta['buyout_flag_true'])}**",
        f"- Диапазон дат продаж: **{meta['sale_date_min']} → {meta['sale_date_max']}**",
        f"- Дата отсечки наблюдения: **{meta['observed_cutoff']}**",
        "",
        "## 2) Методология",
        "",
        "- Когорты построены по `sale_date` с агрегацией по неделям.",
        "- Для выкупа используется `received_ts`.",
        "- Для невыкупа используется наиболее ранняя из `rejected_ts` / `returned_ts`.",
        "- Метрика `buyout_rate_cum` считается с поправкой на зрелость когорты: в знаменателе только заказы, которые уже могли быть наблюдаемы на день D.",
        "",
        "## 3) Ключевые выводы",
        "",
        f"1. Основной рост выкупа приходится на **{peak_increment_day}-й день** после заказа.",
        f"2. Около 80% финального выкупа набирается к **{day_80}-му дню**, а плато (~95%) достигается к **{day_95}-му дню**.",
        f"3. Средний накопительный выкуп к D{MAX_DAY}: **{fmt_pct(final_avg_rate)}**.",
        f"4. {trend_note}",
        f"5. {monthly_note}",
        "",
        "## 4) Лучшие и слабейшие зрелые недельные когорты",
        "",
        "### Лучшие когорты",
        "",
        render_rank_table(top_weekly, f"buyout_rate_day_{MAX_DAY}", n=5),
        "",
        "### Слабейшие когорты",
        "",
        render_rank_table(bottom_weekly, f"buyout_rate_day_{MAX_DAY}", n=5),
        "",
        "## 5) Интерпретация",
        "",
        f"- Основной прирост выкупа сконцентрирован в первые **{day_95} дней**.",
        f"- После D{MAX_DAY} кривая выходит на зрелый уровень.",
        f"- Заметный разброс между когортами указывает на то, что на следующем этапе стоит проверить влияние доставки, региона и менеджера.",
        "",
        regional_section,
        "",
        "## 6) Сформированные файлы",
        "",
        "- `cohort_analysis_data.xlsx` (один файл с 4 листами):",
        "  - Лист 1: Метрики по неделям",
        "  - Лист 2: Итоги по неделям",
        "  - Лист 3: Итоги по месяцам",
        "  - Лист 4: Накопительный выкуп",
        "- `cohort_heatmap_weekly_d30.png` (последние недели)",
        "- `cohort_heatmap_weekly_d30_first.png` (первые недели)",
        "- `cohort_curves_weekly_selected.png`",
        "- `cohort_sizes_weekly.png`",
        "- `final_buyout_rate_weekly.png`",
        "- `final_buyout_rate_monthly.png`",
        "- `regional_curves_comparison.png` (накопительный выкуп по топ-10 регионам)",
        "- `regional_final_buyout_comparison.png` (сравнение финальных выкупов)",
        "- `regional_heatmap_by_days.png` (хитмап выкупа по неделям и регионам)",
    ]

    return "\n".join(lines)

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    print(f"Loading data from {INPUT_PATH}...")
    df, meta = load_and_prepare_data(INPUT_PATH)
    print(
        f"Rows after cleaning: {meta['rows_after_dedup']} | "
        f"weekly cohorts: {meta['weekly_cohorts']} | monthly cohorts: {meta['monthly_cohorts']}"
    )

    weekly_long, weekly_pivot, weekly_summary = build_cohort_metrics(df, "cohort_week", MAX_DAY)
    monthly_long, monthly_pivot, monthly_summary = build_cohort_metrics(df, "cohort_month", MAX_DAY)

    selected_cohorts = select_line_cohorts(weekly_summary, skip_first=3)
    avg_curve = plot_curves(weekly_pivot, selected_cohorts, CHARTS_DIR / "cohort_curves_weekly_selected.png")
    plot_heatmap(weekly_pivot, CHARTS_DIR / "cohort_heatmap_weekly_d30.png")
    plot_heatmap_first(weekly_pivot, CHARTS_DIR / "cohort_heatmap_weekly_d30_first.png")
    plot_cohort_sizes(weekly_summary, CHARTS_DIR / "cohort_sizes_weekly.png", "Размер недельных когорт по sale_date")
    plot_final_buyout(weekly_summary, CHARTS_DIR / "final_buyout_rate_weekly.png", f"Финальный buyout rate по недельным когортам (D{MAX_DAY}, только зрелые когорты)")
    plot_final_buyout(monthly_summary, CHARTS_DIR / "final_buyout_rate_monthly.png", f"Обзорный месячный buyout rate (D{MAX_DAY}, только зрелые когорты)")
    # Тепловая карта: источник × месяц
    plot_source_month_heatmap(
        df,
        output_path=CHARTS_DIR / "source_month_heatmap.png",
        min_orders=500
    )

    # Региональный анализ
    print("\nBuilding regional analysis...")
    top_regions = get_top_regions(df, n=10)
    region_pivots: dict[str, pd.DataFrame] = {}
    region_summaries: dict[str, pd.DataFrame] = {}
    region_avg_curves: dict[str, pd.Series] = {}
    regional_data: dict[str, dict[str, object]] = {}

    for region in top_regions:
        region_df = df[df["lead_region"] == region].copy()
        if len(region_df) > 0:
            _, region_pivot, region_summary = build_cohort_metrics(region_df, "cohort_week", MAX_DAY)
            region_pivots[region] = region_pivot
            region_summaries[region] = region_summary
            
            # Получаем среднюю кривую для региона
            mature_cohorts = region_summary.loc[region_summary["fully_mature"]]
            if len(mature_cohorts) > 0:
                region_avg_curve = region_pivot.loc[region_pivot.index.isin(mature_cohorts.index)].mean(axis=0, skipna=True)
            else:
                region_avg_curve = region_pivot.mean(axis=0, skipna=True)
            region_avg_curves[region] = region_avg_curve
            
            # Собираем статистику для отчета
            regional_data[region] = {
                "total_orders": int(len(region_df)),
                "buyout_orders": int(region_df["buyout_event_date"].notna().sum()),
                "avg_buyout_rate": float(region_pivot.mean().mean()) if region_pivot.size > 0 else 0.0,
                "best_buyout_rate": float(region_pivot.max().max()) if region_pivot.size > 0 else 0.0,
            }

    # Создаем региональные графики
    if region_avg_curves:
        plot_regional_daily_comparison(region_avg_curves, CHARTS_DIR / "regional_curves_comparison.png")
        plot_regional_final_buyout_comparison(region_summaries, CHARTS_DIR / "regional_final_buyout_comparison.png")
        plot_regional_heatmap(region_pivots, region_summaries, CHARTS_DIR / "regional_heatmap_by_days.png")

    # Сохраняем все таблицы в один Excel файл с несколькими листами
    with pd.ExcelWriter(OUTPUT_DIR / "cohort_analysis_data.xlsx", engine="openpyxl") as writer:
        weekly_long.to_excel(writer, sheet_name="Метрики по неделям", index=False, float_format="%.6f")
        weekly_summary.to_excel(writer, sheet_name="Итоги по неделям", float_format="%.6f")
        monthly_summary.to_excel(writer, sheet_name="Итоги по месяцам", float_format="%.6f")
        weekly_pivot.to_excel(writer, sheet_name="Накопительный выкуп", float_format="%.6f")

    summary_text = build_summary_report(meta, weekly_summary, monthly_summary, avg_curve, regional_data if regional_data else None)
    (OUTPUT_DIR / "summary.md").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print("\nSaved outputs:")
    for path in [
        OUTPUT_DIR / "cohort_analysis_data.xlsx",
        OUTPUT_DIR / "summary.md",
        CHARTS_DIR / "cohort_heatmap_weekly_d30.png",
        CHARTS_DIR / "cohort_heatmap_weekly_d30_first.png",
        CHARTS_DIR / "cohort_curves_weekly_selected.png",
        CHARTS_DIR / "cohort_sizes_weekly.png",
        CHARTS_DIR / "final_buyout_rate_weekly.png",
        CHARTS_DIR / "final_buyout_rate_monthly.png",
        CHARTS_DIR / "regional_curves_comparison.png",
        CHARTS_DIR / "regional_final_buyout_comparison.png",
        CHARTS_DIR / "source_month_heatmap.png",
        CHARTS_DIR / "regional_heatmap_by_days.png",
    ]:
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
