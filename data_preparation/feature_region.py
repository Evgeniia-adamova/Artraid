#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Быстрое извлечение региона РФ из адресов на базе OSM NDJSON + Natasha.

Что делает:
1) строит city -> region из OSM;
2) строит алиасы регионов из OSM;
3) сначала ищет явный регион в строке;
4) потом пытается извлечь город через Natasha;
5) потом делает точное city -> region;
6) потом безопасную проверку опечатки на 1 символ;
7) отдельно и надёжно обрабатывает Крым и Севастополь;

Вход:
- place_city.ndjson
- place_town.ndjson
- place_village.ndjson
- place_hamlet.ndjson
- addresses.txt

Выход:
- osm_city_region_map.csv
- addresses_with_regions.csv
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple
import json
import re

import pandas as pd

try:
    from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
    NATASHA_AVAILABLE = True
except Exception:
    NATASHA_AVAILABLE = False


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"
REGION_DATASETS_DIR = BASE_DIR / "region_datasets"
CONFIG_DIR = BASE_DIR / "config"

# Input: processed clean data from data preparation pipeline
INPUT_EXCEL_PATH = CLEAN_DIR / "clean_data.xlsx"
# Output: data with extracted regions
OUTPUT_EXCEL_PATH = CLEAN_DIR / "clean_data.xlsx"
# Cache: precomputed city->region mapping from OSM
CITY_MAP_CACHE = REGION_DATASETS_DIR / "osm_city_region_map.csv"

SOURCE_FILES = [
    REGION_DATASETS_DIR / "place_city.ndjson",
    REGION_DATASETS_DIR / "place-town.ndjson",
    REGION_DATASETS_DIR / "place-village.ndjson",
    REGION_DATASETS_DIR / "place-hamlet.ndjson",
]


# =========================
# NATASHA INIT
# =========================
if NATASHA_AVAILABLE:
    try:
        segmenter = Segmenter()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)
    except Exception:
        NATASHA_AVAILABLE = False
        segmenter = None
        emb = None
        ner_tagger = None
else:
    segmenter = None
    emb = None
    ner_tagger = None


# =========================
# NORMALIZATION
# =========================
def normalize_key(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.lower().replace("ё", "е")
    t = re.sub(r"[.,;:/\\()\[\]{}|]", " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def canonical_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[а-яa-z0-9]+", normalize_key(text))


def stem_adjective(word: str) -> str:
    w = normalize_key(word)
    for suf in ("ская", "ский", "ской", "ское"):
        if w.endswith(suf):
            return w[: -len(suf)]
    return w


# =========================
# EDIT DISTANCE <= 1
# =========================
def is_edit_distance_leq_one(a: str, b: str) -> bool:
    a = normalize_key(a)
    b = normalize_key(b)

    if a == b:
        return True

    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        mismatches = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                mismatches += 1
                if mismatches > 1:
                    return False
        return True

    if la > lb:
        a, b = b, a
        la, lb = lb, la

    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue

        edits += 1
        if edits > 1:
            return False

        j += 1

    return True


def token_close(a: str, b: str) -> bool:
    a = normalize_key(a)
    b = normalize_key(b)

    if a == b:
        return True

    if len(a) < 4 or len(b) < 4:
        return False

    return is_edit_distance_leq_one(a, b)


def tokens_window_exact(window: List[str], alias_tokens: List[str]) -> bool:
    if len(window) != len(alias_tokens):
        return False
    return all(w == a for w, a in zip(window, alias_tokens))


def tokens_window_close(window: List[str], alias_tokens: List[str]) -> bool:
    if len(window) != len(alias_tokens):
        return False
    return all(token_close(w, a) for w, a in zip(window, alias_tokens))


# =========================
# IO
# =========================
def read_ndjson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_excel_data(path: Path) -> pd.DataFrame:
    """Читает Excel файл с данными о городах"""
    df = pd.read_excel(path)
    return df


# =========================
# CITY->REGION MAP
# =========================
def load_or_build_city_map() -> pd.DataFrame:
    if CITY_MAP_CACHE.exists():
        return pd.read_csv(CITY_MAP_CACHE, dtype=str)

    rows = []
    for file_path in SOURCE_FILES:
        if not file_path.exists():
            raise FileNotFoundError(f"Не найден файл: {file_path}")

        for obj in read_ndjson(file_path):
            address = obj.get("address") or {}
            region = address.get("state")
            if not region:
                continue

            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("hamlet")
                or obj.get("name")
            )

            if not city:
                continue

            rows.append(
                {
                    "city": canonical_text(str(city)),
                    "region": canonical_text(str(region)),
                }
            )

    if not rows:
        raise ValueError("Не удалось собрать city->region map из OSM файлов.")

    df = pd.DataFrame(rows).dropna()
    df.to_csv(CITY_MAP_CACHE, index=False, encoding="utf-8-sig")
    return df


# =========================
# BUILD INDEXES
# =========================
def build_region_aliases(region: str) -> List[str]:
    r = normalize_key(region)
    aliases = {r}

    if r in {"москва", "санкт петербург"}:
        return list(aliases)

    if r.startswith("республика "):
        name = r[len("республика "):].strip()
        if name:
            aliases.add(f"респ {name}")
            aliases.add(f"{name} республика")
            aliases.add(f"республика {name}")
        return list(aliases)

    if r.endswith("область"):
        base = r[: -len("область")].strip()
        if base:
            first = base.split()[0]
            root = stem_adjective(first)
            aliases.add(f"{base} обл")
            aliases.add(f"обл {base}")
            aliases.add(f"{first} обл")
            aliases.add(f"обл {first}")
            if root:
                aliases.add(f"{root} обл")
                aliases.add(f"обл {root}")
        return list(aliases)

    if r.endswith("край"):
        base = r[: -len("край")].strip()
        if base:
            first = base.split()[0]
            root = stem_adjective(first)
            aliases.add(f"{base} кр")
            aliases.add(f"край {base}")
            aliases.add(f"{first} кр")
            aliases.add(f"край {first}")
            if root:
                aliases.add(f"{root} кр")
                aliases.add(f"край {root}")
        return list(aliases)

    return list(aliases)


def build_indexes(city_map: pd.DataFrame):
    city_to_regions: Dict[str, List[str]] = defaultdict(list)
    region_names = set()

    for _, row in city_map.iterrows():
        city = normalize_key(row["city"])
        region = canonical_text(row["region"])
        if city and region:
            city_to_regions[city].append(region)
            region_names.add(region)

    city_exact_map: Dict[str, str] = {}
    for city, regions in city_to_regions.items():
        city_exact_map[city] = pd.Series(regions).value_counts().idxmax()

    region_bucket: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    city_bucket: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for region in sorted(region_names):
        for alias in build_region_aliases(region):
            alias_norm = normalize_key(alias)
            if not alias_norm:
                continue
            first = alias_norm.split()[0]
            region_bucket[first].append((alias_norm, region))

    for city, region in city_exact_map.items():
        first = city.split()[0]
        city_bucket[first].append((city, region))

    for bucket in region_bucket.values():
        bucket.sort(key=lambda x: len(x[0]), reverse=True)
    for bucket in city_bucket.values():
        bucket.sort(key=lambda x: len(x[0]), reverse=True)

    city_overrides = {
        "благовещенск": "Амурская область",
        "новочеркасск": "Ростовская область",
        "дмитров": "Московская область",
        "владивосток": "Приморский край",
        "находка": "Приморский край",
        "абакан": "Республика Хакасия",
        "дербент": "Республика Дагестан",
        "щелково": "Московская область",
        "островцы": "Московская область",
        "южно сахалинск": "Сахалинская область",
        "камень на оби": "Алтайский край",
        "каменск уральский": "Свердловская область",
        "казань": "Республика Татарстан",
        "челны": "Республика Татарстан",
        "нижний новгород": "Нижегородская область",
        "краснодар": "Краснодарский край",
        "симферополь": "Республика Крым",
        "севастополь": "Севастополь",
        "ялта": "Республика Крым",
        "керчь": "Республика Крым",
        "евпатория": "Республика Крым",
        "феодосия": "Республика Крым",
        "саки": "Республика Крым",
        "джанкой": "Республика Крым",
        "бахчисарай": "Республика Крым",
        "алушта": "Республика Крым",
        "крым": "Республика Крым",
        "crimea": "Республика Крым",
    }

    return region_bucket, city_bucket, city_overrides


# =========================
# MATCHING
# =========================
@lru_cache(maxsize=50000)
def approx_keys_for_token(token: str, keys_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    token = normalize_key(token)
    if len(token) < 4:
        return tuple()

    matches = []
    for key in keys_tuple:
        if abs(len(token) - len(key)) > 1:
            continue
        if is_edit_distance_leq_one(token, key):
            matches.append(key)
    return tuple(matches)


def collect_candidates(
    norm_text: str,
    tokens: List[str],
    bucket: Dict[str, List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    bucket_keys = tuple(bucket.keys())
    candidate_keys = set()

    for tok in tokens:
        if tok in bucket:
            candidate_keys.add(tok)

    if not candidate_keys:
        for tok in tokens:
            if len(tok) < 4:
                continue
            for key in approx_keys_for_token(tok, bucket_keys):
                candidate_keys.add(key)

    candidates: List[Tuple[str, str]] = []
    for key in candidate_keys:
        candidates.extend(bucket.get(key, []))

    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    return candidates


def match_from_candidates(
    norm_text: str,
    tokens: List[str],
    candidates: List[Tuple[str, str]],
) -> Tuple[str, str] | None:
    for alias, value in candidates:
        alias_tokens = alias.split()
        if not alias_tokens:
            continue

        if alias in norm_text:
            return value, "exact_substring"

        n = len(alias_tokens)
        if n <= len(tokens):
            for i in range(len(tokens) - n + 1):
                window = tokens[i : i + n]
                if tokens_window_exact(window, alias_tokens):
                    return value, "exact_window"

        if n <= len(tokens):
            for i in range(len(tokens) - n + 1):
                window = tokens[i : i + n]
                if tokens_window_close(window, alias_tokens):
                    return value, "typo_window"

    return None


def extract_locs_with_natasha(text: str) -> List[str]:
    if not NATASHA_AVAILABLE:
        return []

    try:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        locs: List[str] = []
        for span in doc.spans:
            if span.type == "LOC":
                loc = text[span.start:span.stop].strip()
                if loc:
                    locs.append(loc)
        return locs
    except Exception:
        return []


def extract_region(address: str,
                   region_bucket: Dict[str, List[Tuple[str, str]]],
                   city_bucket: Dict[str, List[Tuple[str, str]]],
                   city_overrides: Dict[str, str]) -> Tuple[str, str]:
    if not isinstance(address, str) or not address.strip():
        return "Не определен", "empty"

    norm = normalize_key(address)
    tokens = tokenize(address)

    # 0) Крым / Севастополь: отдельный, более надёжный слой
    if re.search(r"(?<!\w)(крым|crimea)(?!\w)", norm):
        return "Республика Крым", "hardcoded_crimea"
    if re.search(r"(?<!\w)севастополь(?!\w)", norm):
        return "Севастополь", "hardcoded_sevastopol"

    # 1) Явный регион в строке
    region_candidates = collect_candidates(norm, tokens, region_bucket)
    hit = match_from_candidates(norm, tokens, region_candidates)
    if hit:
        region, source = hit
        return region, f"region_{source}"

    # 2) Natasha: LOC -> city -> region
    if NATASHA_AVAILABLE:
        locs = extract_locs_with_natasha(address)
        for loc in locs:
            loc_norm = normalize_key(loc)

            if re.search(r"(?<!\w)(крым|crimea)(?!\w)", loc_norm):
                return "Республика Крым", "natasha_crimea"
            if re.search(r"(?<!\w)севастополь(?!\w)", loc_norm):
                return "Севастополь", "natasha_sevastopol"

            if loc_norm in city_overrides:
                return city_overrides[loc_norm], "natasha_city_override"

            city_candidates = collect_candidates(loc_norm, tokenize(loc), city_bucket)
            hit = match_from_candidates(loc_norm, tokenize(loc), city_candidates)
            if hit:
                region, source = hit
                return region, f"natasha_{source}"

            for city_key, region_name in city_overrides.items():
                if city_key in loc_norm:
                    return region_name, "natasha_city_substring_override"

    # 3) Быстрые overrides по строке
    for city_key, region_name in city_overrides.items():
        city_norm = normalize_key(city_key)
        if city_norm in norm:
            return region_name, "city_override"

    # 4) Точное / typo city matching через OSM map
    city_candidates = collect_candidates(norm, tokens, city_bucket)
    hit = match_from_candidates(norm, tokens, city_candidates)
    if hit:
        region, source = hit
        return region, f"city_{source}"

    return "Не определен", "unmatched"


# =========================
# MAIN
# =========================
def main():
    print("Строю/читаю city->region map...")
    city_map = load_or_build_city_map()
    print(f"Записей в map: {len(city_map):,}")

    print("Строю индексы...")
    region_bucket, city_bucket, city_overrides = build_indexes(city_map)

    print("Читаю Excel файл...")
    df = read_excel_data(INPUT_EXCEL_PATH)
    print(f"Строк в файле: {len(df):,}")

    if "contact_Город" not in df.columns:
        print("Ошибка: колонка 'contact_Город' не найдена в файле!")
        print(f"Доступные колонки: {list(df.columns)}")
        return

    # Инициализируем новые колонки
    df["lead_region"] = "Не определен"
    # df["lead_region_source"] = "unmatched"

    print("Обрабатываю города...")
    for i, city in enumerate(df["contact_Город"], 1):
        region, source = extract_region(
            str(city) if pd.notna(city) else "",
            region_bucket,
            city_bucket,
            city_overrides,
        )
        df.at[i - 1, "lead_region"] = region
        # df.at[i - 1, "lead_region_source"] = source

        if i % 1000 == 0:
            print(f"Обработано: {i:,}")

    print("Сохраняю результат...")
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print("Готово.")
    print(f"Файл сохранён: {OUTPUT_EXCEL_PATH}")


if __name__ == "__main__":
    main()