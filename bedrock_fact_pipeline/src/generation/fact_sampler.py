import json
import os
import random
from typing import Any, Dict, List, Optional

from src.data_report.schema import (
    adapt_biome,
    adapt_entity,
    adapt_loot_table,
    adapt_recipe,
    adapt_spawn_rule,
    adapt_trading,
)

_CATEGORY_DIRS = {
    "recipes": "recipe",
    "loot_tables": "loot_table",
    "entities": "entity",
    "spawn_rules": "spawn_rule",
    "trading": "trading",
    "biomes": "biome",
}

_ADAPTER_BY_CATEGORY = {
    "recipe": adapt_recipe,
    "loot_table": adapt_loot_table,
    "entity": adapt_entity,
    "spawn_rule": adapt_spawn_rule,
    "trading": adapt_trading,
    "biome": adapt_biome,
}


def category_for_path(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/")
    for dir_name, category in _CATEGORY_DIRS.items():
        if f"/behavior_pack/{dir_name}/" in normalized:
            return category
    return None


def _behavior_pack_relative_path(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/")
    marker = "/behavior_pack/"
    if marker not in normalized:
        return None
    return normalized.split(marker, 1)[1]


def _subject_id(path: str) -> str:
    normalized = path.replace("\\", "/")
    rel = None
    for dir_name in _CATEGORY_DIRS:
        marker = f"/behavior_pack/{dir_name}/"
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            break
    if rel is None:
        rel = os.path.basename(normalized)
    if rel.endswith(".json"):
        rel = rel[: -len(".json")]
    if rel.endswith(".biome"):
        rel = rel[: -len(".biome")]
    return f"minecraft:{rel}"


def _strip_json_comments(text: str) -> str:
    """Strip `//` line comments from JSON text, leaving string literals untouched.

    Real Mojang-shipped bedrock-samples files use JSONC-style `//` comments,
    which strict `json.loads` rejects. Scans char-by-char tracking string
    state so a value like the recipe pattern row "///" is never mistaken
    for a comment.
    """
    result = []
    in_string = False
    escape = False
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < length and text[i + 1] == "/":
            while i < length and text[i] not in "\r\n":
                i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _load_json_tolerant(file_path: str) -> Any:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(_strip_json_comments(text))


def _try_load_json_tolerant(file_path: str) -> Optional[Any]:
    try:
        return _load_json_tolerant(file_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Skipping unparseable fact-report file {file_path}: {exc}")
        return None


def discover_fact_records(data_report_dir: str) -> List[Dict[str, Any]]:
    # Loot tables can reference other loot table files by behavior_pack-
    # relative path (e.g. gameplay/fishing.json -> fishing/junk.json).
    # Pre-load every loot table so adapt_loot_table can resolve those
    # references regardless of os.walk order.
    loot_table_cache: Dict[str, Any] = {}
    for root, _dirs, files in os.walk(data_report_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(root, filename)
            if category_for_path(file_path) != "loot_table":
                continue
            rel_path = _behavior_pack_relative_path(file_path)
            if rel_path is None:
                continue
            parsed = _try_load_json_tolerant(file_path)
            if parsed is not None:
                loot_table_cache[rel_path] = parsed

    records: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(data_report_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(root, filename)
            category = category_for_path(file_path)
            if category is None:
                continue
            subject_id = _subject_id(file_path)
            if category == "loot_table":
                rel_path = _behavior_pack_relative_path(file_path)
                raw_json = loot_table_cache.get(rel_path) if rel_path else None
                if raw_json is None:
                    continue
                records.extend(adapt_loot_table(subject_id, raw_json, resolve=loot_table_cache.get))
                continue
            raw_json = _try_load_json_tolerant(file_path)
            if raw_json is None:
                continue
            records.extend(_ADAPTER_BY_CATEGORY[category](subject_id, raw_json))
    return records


def sample_fact_records(
    records: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pool = [r for r in records if category is None or r["category"] == category]
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, k=n)
