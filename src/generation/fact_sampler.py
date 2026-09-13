import json
import os
import random
from typing import Any, Dict, List, Optional

from src.data_report.schema import (
    adapt_advancement,
    adapt_enchantment,
    adapt_jukebox_song,
    adapt_loot_table,
    adapt_painting_variant,
    adapt_recipe,
    adapt_tag,
    adapt_villager_trade,
    adapt_worldgen_structure,
)

_ADAPTER_BY_CATEGORY = {
    "recipe": adapt_recipe,
    "loot_table": adapt_loot_table,
    "advancement": adapt_advancement,
    "tag": adapt_tag,
    "worldgen_structure": adapt_worldgen_structure,
    "enchantment": adapt_enchantment,
    "villager_trade": adapt_villager_trade,
    "jukebox_song": adapt_jukebox_song,
    "painting_variant": adapt_painting_variant,
}


def category_for_path(path: str) -> Optional[str]:
    normalized = path.replace("\\", "/")
    if "/datapacks/" in normalized:
        # Optional/experimental vanilla datapacks (trade_rebalance, etc.)
        # nest a full second "data/<namespace>/<registry>/..." tree under
        # datapacks/<pack_name>/, which _subject_id's single-level parsing
        # can't represent. Out of scope for now rather than producing a
        # garbled compound identifier.
        return None
    if "/recipe/" in normalized or "/recipes/" in normalized:
        return "recipe"
    if "/loot_table" in normalized:
        return "loot_table"
    if "/advancement" in normalized:
        return "advancement"
    if "/tags/" in normalized:
        return "tag"
    if "/worldgen/structure/" in normalized:
        return "worldgen_structure"
    # the enchantment *definition* registry (not /tags/enchantment/, handled
    # above, nor /enchantment_provider/, which lacks the trailing slash match)
    if "/enchantment/" in normalized:
        return "enchantment"
    if "/villager_trade/" in normalized:
        return "villager_trade"
    if "/jukebox_song/" in normalized:
        return "jukebox_song"
    if "/painting_variant/" in normalized:
        return "painting_variant"
    return None


def _subject_id(data_report_dir: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, data_report_dir)
    rel = rel[: -len(".json")] if rel.endswith(".json") else rel
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "data":
        namespace = parts[1]
        registry_len = 2 if parts[2] == "worldgen" else 1
        name_start = 2 + registry_len
        name = "/".join(parts[name_start:]) if len(parts) > name_start else parts[2]
        return f"{namespace}:{name}"
    return rel


def discover_fact_records(data_report_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(data_report_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(root, filename)
            category = category_for_path(file_path)
            if category is None:
                continue
            adapter = _ADAPTER_BY_CATEGORY[category]
            with open(file_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            subject_id = _subject_id(data_report_dir, file_path)
            records.extend(adapter(subject_id, raw_json))
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
