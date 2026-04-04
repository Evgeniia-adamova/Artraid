from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def select_line_cohorts(summary: pd.DataFrame) -> list[pd.Timestamp]:
    mature = summary.loc[summary["fully_mature"]].sort_index()
    if mature.empty:
        return list(summary.sort_index().tail(min(8, len(summary))).index)

    chosen = list(mature.head(LINE_COHORTS_PER_SIDE).index) + list(mature.tail(LINE_COHORTS_PER_SIDE).index)
    selected: list[pd.Timestamp] = []
    for cohort in chosen:
        if cohort not in selected:
            selected.append(cohort)
    return selected


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
        cmap="YlGnBu",
        linewidths=0.1,
        linecolor="white",
        cbar_kws={"label": "Cumulative buyout rate, %"},
        annot_kws={"size": 11},
        vmin=0,
        vmax=max(75, float(np.nanmax(heatmap_pct.values)) if np.isfinite(np.nanmax(heatmap_pct.values)) else 75),
    )
    ax.set_title("Недельные когорты: накопительный выкуп по дням (последние недели, D0–D30)", pad=14)
    ax.set_xlabel("Days after order")
    ax.set_ylabel("Cohort week start")
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

    plt.plot(avg_curve.index, avg_curve.values * 100, linestyle="--", linewidth=3, color="black", label="Средняя по всем когортам")
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
    plt.figure(figsize=(14, 6))
    plt.bar(labels, summary["orders_total"], color="#4c78a8")
    plt.title(title)
    plt.xlabel("Cohort")
    plt.ylabel("Orders count")
    plt.xticks(rotation=70, ha="right")
    plt.grid(axis="y", alpha=0.25)
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
            colors.append("#2b8cbe")
        elif value <= q1:
            colors.append("#de2d26")
        else:
            colors.append("#9ecae1")

    plt.figure(figsize=(14, 6.5))
    plt.bar(labels, mature_df["buyout_pct"], color=colors)
    plt.axhline(mature_df["buyout_pct"].mean(), color="black", linestyle="--", linewidth=1.5, label=f"Среднее: {mature_df['buyout_pct'].mean():.1f}%")
    plt.title(title)
    plt.xlabel("Cohort")
    plt.ylabel(f"Buyout rate by D{MAX_DAY}, %")
    plt.xticks(rotation=70, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return mature_df


def build_summary_report(
    meta: dict[str, object],
    weekly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    avg_curve: pd.Series,
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
        "## 6) Сформированные файлы",
        "",
        "- `cohort_metrics_weekly.csv`",
        "- `cohort_summary_weekly.csv`",
        "- `cohort_summary_monthly.csv`",
        "- `cohort_heatmap_table_weekly.csv`",
        "- `cohort_heatmap_weekly_d30.png`",
        "- `cohort_curves_weekly_selected.png`",
        "- `cohort_sizes_weekly.png`",
        "- `final_buyout_rate_weekly.png`",
        "- `final_buyout_rate_monthly.png`",
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

    selected_cohorts = select_line_cohorts(weekly_summary)
    avg_curve = plot_curves(weekly_pivot, selected_cohorts, CHARTS_DIR / "cohort_curves_weekly_selected.png")
    plot_heatmap(weekly_pivot, CHARTS_DIR / "cohort_heatmap_weekly_d30.png")
    plot_cohort_sizes(weekly_summary, CHARTS_DIR / "cohort_sizes_weekly.png", "Размер недельных когорт по sale_date")
    plot_final_buyout(weekly_summary, CHARTS_DIR / "final_buyout_rate_weekly.png", f"Финальный buyout rate по недельным когортам (D{MAX_DAY}, только зрелые когорты)")
    plot_final_buyout(monthly_summary, CHARTS_DIR / "final_buyout_rate_monthly.png", f"Обзорный месячный buyout rate (D{MAX_DAY}, только зрелые когорты)")

    weekly_long.to_csv(OUTPUT_DIR / "cohort_metrics_weekly.csv", index=False, float_format="%.6f")
    weekly_summary.to_csv(OUTPUT_DIR / "cohort_summary_weekly.csv", float_format="%.6f")
    monthly_summary.to_csv(OUTPUT_DIR / "cohort_summary_monthly.csv", float_format="%.6f")
    weekly_pivot.to_csv(OUTPUT_DIR / "cohort_heatmap_table_weekly.csv", float_format="%.6f")

    summary_text = build_summary_report(meta, weekly_summary, monthly_summary, avg_curve)
    (OUTPUT_DIR / "summary.md").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print("\nSaved outputs:")
    for path in [
        OUTPUT_DIR / "cohort_metrics_weekly.csv",
        OUTPUT_DIR / "cohort_summary_weekly.csv",
        OUTPUT_DIR / "cohort_summary_monthly.csv",
        OUTPUT_DIR / "cohort_heatmap_table_weekly.csv",
        OUTPUT_DIR / "summary.md",
        CHARTS_DIR / "cohort_heatmap_weekly_d30.png",
        CHARTS_DIR / "cohort_curves_weekly_selected.png",
        CHARTS_DIR / "cohort_sizes_weekly.png",
        CHARTS_DIR / "final_buyout_rate_weekly.png",
        CHARTS_DIR / "final_buyout_rate_monthly.png",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
