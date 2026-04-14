from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "data_preparation" / "data" / "clean" / "clean_data.xlsx"
OUTPUT_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"

REQUIRED_COLUMNS = [
    "sale_date",
    "lead_region",
]


def load_data(path: Path) -> pd.DataFrame:
    """Загружаем данные с избранными колонками."""
    columns_to_load = REQUIRED_COLUMNS
    df = pd.read_excel(path, usecols=lambda col: col in columns_to_load)
    
    # Проверяем наличие нужных колонок
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {missing}")
    
    # Конвертируем дату
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.dropna(subset=["sale_date"])
    
    # Извлекаем месяц
    df["month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()
    
    return df


def build_sales_pivot(df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, list[str]]:
    """
    Строим таблицу продаж по месяцам и регионам.
    Возвращает таблицу и список топ регионов.
    """
    # Получаем топ регионы по общему объему продаж
    top_regions = df["lead_region"].value_counts().head(top_n).index.tolist()
    
    # Фильтруем только топ регионы
    df_top = df[df["lead_region"].isin(top_regions)].copy()
    
    # Строим pivot: месяцы × регионы
    sales_counts = df_top.groupby(["month", "lead_region"]).size().unstack(fill_value=0)
    
    return sales_counts, top_regions


def add_metrics(sales_pivot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Добавляем метрики к таблице продаж:
    - Процент прироста месяц-к-месяцу
    - Доля региона от общего объема
    """
    # Копируем для итоговых метрик
    monthly_totals = sales_pivot.sum(axis=1)
    
    # Доля региона (%)
    region_share = sales_pivot.copy()
    for col in region_share.columns:
        region_share[col] = (region_share[col] / monthly_totals * 100).round(2)
    
    # Процент прироста месяц-к-месяцу
    mom_growth = sales_pivot.copy()
    mom_growth.iloc[0] = np.nan
    for i in range(1, len(mom_growth)):
        for col in mom_growth.columns:
            prev_val = sales_pivot.iloc[i-1][col]
            curr_val = sales_pivot.iloc[i][col]
            if prev_val > 0:
                mom_growth.iloc[i][col] = ((curr_val - prev_val) / prev_val * 100).round(2)
            elif curr_val > 0:
                mom_growth.iloc[i][col] = 100.0
            else:
                mom_growth.iloc[i][col] = np.nan
    
    return region_share, mom_growth, monthly_totals


def save_excel_report(sales_pivot: pd.DataFrame, region_share: pd.DataFrame, 
                      mom_growth: pd.DataFrame, monthly_totals: pd.Series, 
                      output_path: Path) -> None:
    """Сохраняем отчет в Excel с несколькими листами."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sales_pivot.to_excel(writer, sheet_name="Продажи по месяцам")
        
        # Переименуем индекс для читаемости
        region_share_display = region_share.copy()
        region_share_display.index = region_share_display.index.strftime("%Y-%m")
        region_share_display.to_excel(writer, sheet_name="Доля региона %")
        
        # MoM рост
        mom_growth_display = mom_growth.copy()
        mom_growth_display.index = mom_growth_display.index.strftime("%Y-%m")
        mom_growth_display.to_excel(writer, sheet_name="Прирост месяц-к-месяцу %")
        
        # Итого по месяцам
        summary = pd.DataFrame({
            "Месяц": monthly_totals.index.strftime("%Y-%m"),
            "Всего продаж": monthly_totals.values,
        })
        summary.to_excel(writer, sheet_name="Итого по месяцам", index=False)


def plot_monthly_sales(sales_pivot: pd.DataFrame, output_path: Path) -> None:
    """Графики продаж по месяцам для каждого региона."""
    # Переименуем индекс для читаемости
    x_labels = sales_pivot.index.strftime("%Y-%m")
    x_pos = np.arange(len(sales_pivot))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(sales_pivot.columns)))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    width = 0.8 / len(sales_pivot.columns)
    for i, region in enumerate(sales_pivot.columns):
        ax.bar(x_pos + i * width, sales_pivot[region], width, label=region, color=colors[i])
    
    ax.set_xlabel("Месяц", fontsize=12)
    ax.set_ylabel("Количество заказов", fontsize=12)
    ax.set_title("Динамика продаж по месяцам и регионам", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * (len(sales_pivot.columns) - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(title="Регион", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_stacked_area(sales_pivot: pd.DataFrame, output_path: Path) -> None:
    """Стакированная диаграмма площадей - динамика рынка."""
    x_labels = sales_pivot.index.strftime("%Y-%m")
    x_pos = np.arange(len(sales_pivot))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.stackplot(x_pos, *[sales_pivot[col] for col in sales_pivot.columns],
                 labels=sales_pivot.columns, alpha=0.8)
    
    ax.set_xlabel("Месяц", fontsize=12)
    ax.set_ylabel("Количество заказов", fontsize=12)
    ax.set_title("Общий объем продаж: вклад по регионам (стакированная площадь)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos[::max(1, len(x_pos)//12)])
    ax.set_xticklabels(x_labels[::max(1, len(x_labels)//12)], rotation=45, ha="right")
    ax.legend(title="Регион", loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_region_market_share(region_share: pd.DataFrame, output_path: Path) -> None:
    """График доли каждого региона (%) в течение времени."""
    x_labels = region_share.index.strftime("%Y-%m")
    x_pos = np.arange(len(region_share))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(region_share.columns)))
    width = 0.8 / len(region_share.columns)
    
    for i, region in enumerate(region_share.columns):
        ax.bar(x_pos + i * width, region_share[region], width, label=region, color=colors[i])
    
    ax.set_xlabel("Месяц", fontsize=12)
    ax.set_ylabel("Доля в общем объеме, %", fontsize=12)
    ax.set_title("Доля рынка по регионам (месячная динамика)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos + width * (len(region_share.columns) - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(title="Регион", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_total_sales_trend(monthly_totals: pd.Series, output_path: Path) -> None:
    """График общего объема продаж во времени."""
    x_labels = monthly_totals.index.strftime("%Y-%m")
    x_pos = np.arange(len(monthly_totals))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(x_pos, monthly_totals.values, marker="o", linewidth=2.5, markersize=8, 
            color="#2b8cbe", label="Общие продажи")
    ax.fill_between(x_pos, monthly_totals.values, alpha=0.3, color="#2b8cbe")
    
    # Добавляем значения на каждую точку
    for i, val in enumerate(monthly_totals.values):
        ax.text(i, val + 50, str(int(val)), ha="center", va="bottom", fontsize=9)
    
    ax.set_xlabel("Месяц", fontsize=12)
    ax.set_ylabel("Количество заказов", fontsize=12)
    ax.set_title("Общий тренд продаж по месяцам", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos[::max(1, len(x_pos)//12)])
    ax.set_xticklabels(x_labels[::max(1, len(x_labels)//12)], rotation=45, ha="right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_summary(df: pd.DataFrame, sales_pivot: pd.DataFrame, 
                     monthly_totals: pd.Series, top_regions: list[str]) -> str:
    """Генерируем текстовый отчет."""
    total_orders = len(df)
    total_regions = df["lead_region"].nunique()
    
    # Статистика по регионам
    top_5_regions = df["lead_region"].value_counts().head(5)
    
    # Динамика
    first_month = monthly_totals.iloc[0]
    last_month = monthly_totals.iloc[-1]
    growth = ((last_month - first_month) / first_month * 100) if first_month > 0 else 0
    
    avg_monthly = monthly_totals.mean()
    peak_month = monthly_totals.idxmax()
    peak_value = monthly_totals.max()
    
    lines = [
        "# Анализ динамики продаж по месяцам и регионам",
        "",
        "## Обзор",
        "",
        f"- **Всего заказов в датасете**: {total_orders:,}",
        f"- **Количество уникальных регионов**: {total_regions}",
        f"- **Анализируемые регионы (топ-{len(top_regions)})**: {', '.join(top_regions)}",
        "",
        "## Общие метрики",
        "",
        f"- **Среднее количество заказов в месяц**: {int(avg_monthly):,}",
        f"- **Пик продаж**: {peak_month.strftime('%Y-%m')} ({int(peak_value):,} заказов)",
        f"- **Динамика (первый месяц → последний)**: {int(first_month):,} → {int(last_month):,}",
        f"- **Темп роста**: {growth:+.1f}%",
        "",
        "## Топ-5 регионов по объему (всего за период)",
        "",
        "| Регион | Количество заказов | Доля от общего |",
        "|---|---:|---:|",
    ]
    
    for region, count in top_5_regions.items():
        pct = (count / total_orders * 100)
        lines.append(f"| {region} | {int(count):,} | {pct:.1f}% |")
    
    lines.extend([
        "",
        "## Генерированные файлы",
        "",
        "- `regional_sales_dynamics.xlsx` — основной отчет с метриками:",
        "  - Лист 1: Количество продаж по месяцам и регионам",
        "  - Лист 2: Доля каждого региона (%)",
        "  - Лист 3: Процент прироста месяц-к-месяцу",
        "  - Лист 4: Итого по месяцам",
        "- `sales_monthly_dynamics.png` — столбчатая диаграмма продаж",
        "- `sales_stacked_area.png` — стакированная диаграмма площадей",
        "- `market_share_dynamics.png` — доля регионов во времени",
        "- `total_sales_trend.png` — общий тренд продаж",
    ])
    
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="talk")
    
    print(f"Loading data from {INPUT_PATH}...")
    df = load_data(INPUT_PATH)
    print(f"Loaded {len(df)} records with {df['lead_region'].nunique()} unique regions")
    
    # Строим таблицы продаж
    print("\nBuilding sales pivot tables...")
    sales_pivot, top_regions = build_sales_pivot(df, top_n=10)
    region_share, mom_growth, monthly_totals = add_metrics(sales_pivot)
    
    print(f"Analyzed top {len(top_regions)} regions across {len(sales_pivot)} months")
    
    # Сохраняем в Excel
    print("Saving Excel report...")
    excel_path = OUTPUT_DIR / "regional_sales_dynamics.xlsx"
    save_excel_report(sales_pivot, region_share, mom_growth, monthly_totals, excel_path)
    print(f"✓ {excel_path}")
    
    # Создаем графики
    print("Creating visualizations...")
    plot_monthly_sales(sales_pivot, CHARTS_DIR / "sales_monthly_dynamics.png")
    print(f"✓ sales_monthly_dynamics.png")
    
    plot_stacked_area(sales_pivot, CHARTS_DIR / "sales_stacked_area.png")
    print(f"✓ sales_stacked_area.png")
    
    plot_region_market_share(region_share, CHARTS_DIR / "market_share_dynamics.png")
    print(f"✓ market_share_dynamics.png")
    
    plot_total_sales_trend(monthly_totals, CHARTS_DIR / "total_sales_trend.png")
    print(f"✓ total_sales_trend.png")
    
    # Генерируем и сохраняем отчет
    print("Generating summary report...")
    summary = generate_summary(df, sales_pivot, monthly_totals, top_regions)
    summary_path = OUTPUT_DIR / "regional_sales_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"✓ {summary_path}")
    
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
