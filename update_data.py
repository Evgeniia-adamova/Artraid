"""
Запускает полный пайплайн обновления данных:
  1. data_preparation       — чистит CSV → clean_data.xlsx
  2. feature_engineering    — добавляет lead_source_category, price_group и флаги
  3. feature_region         — добавляет lead_region
  4. cohort_buyout_analysis — перегенерирует графики когорт
  5. buyout_loss_analysis   — перегенерирует графики финансов
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

steps = [
    ("Подготовка данных",       ROOT / "data_preparation", "data_preparation.py"),
    ("Feature engineering",     ROOT / "data_preparation", "feature_engineering.py"),
    ("Регионы",                 ROOT / "data_preparation", "feature_region.py"),
    ("Когортный анализ",        ROOT / "Cohort",            "cohort_buyout_analysis.py"),
    ("Финансовые потери",       ROOT / "fin",               "buyout_loss_analysis.py"),
]

for label, cwd, script in steps:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, script],
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"\nОшибка на шаге «{label}». Пайплайн остановлен.")
        sys.exit(result.returncode)

print("\nГотово. Перезапустите Streamlit или нажми R в браузере для обновления.")
