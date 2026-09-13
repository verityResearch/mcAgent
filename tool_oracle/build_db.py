"""Build the offline SQLite fact oracle from the pinned data report.

This is the inference-time TOOL for the tool-augmented model (see
docs/reports/2026-08-16-minecraft-domain-case-study.md). The same report that
feeds the verifier becomes a queryable DB, so tool + verifier + training data
all derive from one source: offline, deterministic, and regenerable per game
version without retraining.

Tables:
  breeding_food(animal, food_item)      -- *_food item tags, nested #refs RESOLVED
  tag(tag, member)                      -- all item tags, nested #refs resolved
  recipe(result_item, ingredient)       -- from the recipe adapter, flattened (all slots,
                                         -- OR-alternatives merged) -- membership/content-
                                         -- coverage use only, NOT AND/OR-structure-preserving.
  recipe_slot(result_item, method, slot, ingredient)
                                         -- the same recipe adapter, UNflattened: one row per
                                         -- (recipe file, AND-slot, OR-alternative). "method" is
                                         -- the source recipe file id (a result_item can have
                                         -- multiple recipe files -- e.g. crafting + stonecutting
                                         -- -- each its own independent method); "slot" is the
                                         -- 0-based AND-slot index within that one file. Use this,
                                         -- not the flat table, when the AND/OR shape matters.
  enchantment(name, max_level, slots)   -- from the enchantment adapter
  villager_trade(trade, wants_item, wants_count, gives_item, gives_count)
  loot(source, drop_item)               -- block/entity drops
  item_use(item, nutrition, saturation, max_damage, attack_damage, equip_slot)
                                         -- from reports/minecraft/components/item/*.json:
                                         -- what an item IS FOR (eating/durability/combat/
                                         -- wearing), not its tags/recipe/drops. This is a
                                         -- DIFFERENT source tree (reports/, not data/) --
                                         -- per-item default component dumps, not registry
                                         -- json files, so it has its own extraction pass.
  ore_depth(block, placement, dist_type, min_y, max_y, count)
                                         -- from data/minecraft/worldgen/{configured_feature,
                                         -- placed_feature}/*.json (type=minecraft:ore only):
                                         -- real vanilla ore Y-level generation data, one row
                                         -- per (ore block, placement rule). "count" is that
                                         -- placement's per-chunk attempt count -- a real,
                                         -- comparable-across-placements frequency proxy, not a
                                         -- true placement probability (biome/discard-chance
                                         -- effects aren't modeled). min_y/max_y are absolute
                                         -- world Y, resolved from the height_range's own
                                         -- absolute/above_bottom/below_top form using the real
                                         -- dimension_type/overworld.json min_y/height.

Recursive tag resolution is load-bearing: a tag value like "#minecraft:meat"
is expanded to its concrete members (beef, chicken, ...) so a lookup returns
real items, not tag references.

Run from the repo root so `from src...` resolves:
    PYTHONPATH=. python3 tool_oracle/build_db.py <report_dir> minecraft.db
"""
import glob
import json
import os
import sqlite3
import sys
from typing import Callable, Dict, List, Set, Tuple

from src.generation.fact_sampler import discover_fact_records

_TagKey = Tuple[str, str]  # (registry, tag_id)


def _load_all_tags(data_report: str) -> Dict[_TagKey, List]:
    """Map (registry, tag_id) -> raw value list, across every tag registry."""
    tags: Dict[_TagKey, List] = {}
    root = os.path.join(data_report, "data", "minecraft", "tags")
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        parts = os.path.relpath(path, root)[: -len(".json")].split(os.sep)
        registry, name = parts[0], "/".join(parts[1:])
        try:
            tags[(registry, f"minecraft:{name}")] = json.load(open(path)).get("values", [])
        except (OSError, json.JSONDecodeError):
            pass
    return tags


def _make_resolver(tags: Dict[_TagKey, List]) -> Callable[[str, str], Set[str]]:
    """resolve(registry, tag_id) -> set of concrete item ids, expanding nested
    ``#tag`` references recursively (cycle-safe, memoized)."""
    cache: Dict[_TagKey, Set[str]] = {}

    def resolve(registry: str, tag_id: str, seen: Set[_TagKey] = None) -> Set[str]:
        key = (registry, tag_id)
        if key in cache:
            return cache[key]
        seen = seen or set()
        if key in seen:
            return set()
        seen.add(key)
        out: Set[str] = set()
        for v in tags.get(key, []):
            if isinstance(v, dict):
                v = v.get("id")
            if not isinstance(v, str):
                continue
            if v.startswith("#"):
                out |= resolve(registry, v[1:], seen)
            else:
                out.add(v)
        cache[key] = out
        return out

    return resolve


_SCHEMA = (
    "CREATE TABLE breeding_food(animal TEXT, food_item TEXT)",
    "CREATE TABLE tag(tag TEXT, member TEXT)",
    "CREATE TABLE recipe(result_item TEXT, ingredient TEXT)",
    "CREATE TABLE recipe_slot(result_item TEXT, method TEXT, slot INT, ingredient TEXT)",
    "CREATE TABLE enchantment(name TEXT, max_level INT, slots TEXT)",
    "CREATE TABLE villager_trade(trade TEXT, wants_item TEXT, wants_count INT, "
    "gives_item TEXT, gives_count INT)",
    "CREATE TABLE loot(source TEXT, drop_item TEXT)",
    "CREATE TABLE jukebox_song(song TEXT, length_seconds INT)",
    "CREATE TABLE painting(painting TEXT, width INT, height INT)",
    "CREATE TABLE item_use(item TEXT, nutrition INT, saturation REAL, "
    "max_damage INT, attack_damage REAL, equip_slot TEXT)",
    "CREATE TABLE ore_depth(block TEXT, placement TEXT, dist_type TEXT, "
    "min_y INT, max_y INT, count INT)",
)
_INDEXES = (
    "CREATE INDEX i_food ON breeding_food(animal)",
    "CREATE INDEX i_tag ON tag(tag)",
    "CREATE INDEX i_recipe ON recipe(result_item)",
    "CREATE INDEX i_recipe_slot ON recipe_slot(result_item)",
    "CREATE INDEX i_ench ON enchantment(name)",
    "CREATE INDEX i_loot ON loot(source)",
    "CREATE INDEX i_item_use ON item_use(item)",
    "CREATE INDEX i_ore_depth ON ore_depth(block)",
)


def _load_dimension_bounds(data_report: str):
    """(min_y, max_y) for the overworld, from the real dimension_type report --
    needed to resolve height_range's above_bottom/below_top forms to absolute
    Y. Falls back to None if the report doesn't have this file (older/partial
    reports); callers must treat that as "can't resolve relative heights"."""
    path = os.path.join(data_report, "data", "minecraft", "dimension_type", "overworld.json")
    try:
        d = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None
    min_y = d.get("min_y")
    height = d.get("height")
    if min_y is None or height is None:
        return None
    return min_y, min_y + height - 1


def _load_ore_configured_features(data_report: str) -> Dict[str, List[str]]:
    """configured_feature_id -> [target block ids], for type=minecraft:ore
    configured features only (skips trees/plants/etc -- anything else isn't
    a minable ore, and its height data wouldn't mean "where do I mine this")."""
    root = os.path.join(data_report, "data", "minecraft", "worldgen", "configured_feature")
    out: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(root, "*.json")):
        cf_id = f"minecraft:{os.path.basename(path)[: -len('.json')]}"
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("type") != "minecraft:ore":
            continue
        blocks = []
        for target in (d.get("config") or {}).get("targets") or []:
            name = (target.get("state") or {}).get("Name")
            if name:
                blocks.append(name)
        if blocks:
            out[cf_id] = blocks
    return out


def _resolve_height(value, min_y: int, max_y: int):
    """A height_range endpoint ({"absolute": N} / {"above_bottom": N} /
    {"below_top": N}) -> the real absolute Y it names."""
    if "absolute" in value:
        return value["absolute"]
    if "above_bottom" in value:
        return min_y + value["above_bottom"]
    if "below_top" in value:
        return max_y - value["below_top"]
    return None


def _resolve_count(value) -> int:
    """The minecraft:count placement modifier's count field is usually a bare
    int, but can be its own value-provider object (a uniform/weighted_list/
    clamped range instead of a fixed number). Resolved to a single
    representative int -- already documented as a rough frequency proxy, not
    an exact probability, so an average here is consistent with that."""
    if isinstance(value, int):
        return value
    if not isinstance(value, dict):
        return 1
    t = value.get("type", "")
    if t == "minecraft:constant":
        return round(value.get("value", 1))
    if "min_inclusive" in value and "max_inclusive" in value:
        return round((value["min_inclusive"] + value["max_inclusive"]) / 2)
    if t == "minecraft:weighted_list":
        dist = value.get("distribution") or []
        total_w = sum(e.get("weight", 1) for e in dist)
        if total_w:
            return round(sum(e.get("data", 0) * e.get("weight", 1) for e in dist) / total_w)
    if t == "minecraft:clamped" and value.get("source"):
        return _resolve_count(value["source"])
    return 1


def _load_ore_placements(data_report: str):
    """Yields (block, placed_feature_id, dist_type, min_y, max_y, count) for
    every placed_feature whose feature resolves to an ore configured_feature.
    Skipped entirely if dimension bounds or the placed_feature directory are
    unavailable -- a missing/partial report means "no ore-depth data", not a
    crash."""
    bounds = _load_dimension_bounds(data_report)
    if bounds is None:
        return
    min_y, max_y = bounds
    configured = _load_ore_configured_features(data_report)
    root = os.path.join(data_report, "data", "minecraft", "worldgen", "placed_feature")
    for path in glob.glob(os.path.join(root, "*.json")):
        pf_id = f"minecraft:{os.path.basename(path)[: -len('.json')]}"
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        blocks = configured.get(d.get("feature"))
        if not blocks:
            continue
        height = None
        count = 1
        for pl in d.get("placement") or []:
            if pl.get("type") == "minecraft:height_range":
                height = pl.get("height")
            elif pl.get("type") == "minecraft:count":
                count = _resolve_count(pl.get("count", 1))
        if not height:
            continue
        lo = _resolve_height(height.get("min_inclusive") or {}, min_y, max_y)
        hi = _resolve_height(height.get("max_inclusive") or {}, min_y, max_y)
        if lo is None or hi is None:
            continue
        # Some vanilla placements define a nominal range that extends past the
        # real playable world (e.g. diamond's -144, well below the actual -64
        # floor) -- an artifact of how the height-provider math is authored,
        # not a claim that ore generates somewhere unreachable. Clamped so
        # reported Y-levels are always real, in-world coordinates.
        lo, hi = max(lo, min_y), min(hi, max_y)
        dist_type = (height.get("type") or "").split(":", 1)[-1]
        for block in blocks:
            yield (block, pf_id, dist_type, lo, hi, count)


def _load_item_use(data_report: str):
    """Walk reports/minecraft/components/item/*.json (per-item default
    component dumps -- a different source tree than data/, which holds
    registry json files) and yield (item_id, nutrition, saturation,
    max_damage, attack_damage, equip_slot) for items with >=1 usage field.
    Skipped entirely if the report wasn't generated (older/partial reports)."""
    root = os.path.join(data_report, "reports", "minecraft", "components", "item")
    if not os.path.isdir(root):
        return
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        item_id = f"minecraft:{os.path.basename(path)[: -len('.json')]}"
        try:
            comps = json.load(open(path)).get("components", {})
        except (OSError, json.JSONDecodeError):
            continue
        food = comps.get("minecraft:food") or {}
        nutrition = food.get("nutrition")
        saturation = food.get("saturation")
        max_damage = comps.get("minecraft:max_damage")
        attack_damage = None
        for mod in comps.get("minecraft:attribute_modifiers") or []:
            if mod.get("type") == "minecraft:attack_damage":
                attack_damage = mod.get("amount")
                break
        equip = comps.get("minecraft:equippable") or {}
        equip_slot = equip.get("slot")
        if any(v is not None for v in (nutrition, saturation, max_damage, attack_damage, equip_slot)):
            yield (item_id, nutrition, saturation, max_damage, attack_damage, equip_slot)


def build(data_report: str, db_path: str) -> Dict[str, int]:
    """Build the DB and return a table -> row-count summary."""
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    c = con.cursor()
    for ddl in _SCHEMA:
        c.execute(ddl)

    tags = _load_all_tags(data_report)
    resolve = _make_resolver(tags)
    counts = {t: 0 for t in ("breeding_food", "tag", "recipe", "enchantment", "villager_trade",
                             "loot", "jukebox_song", "painting", "item_use", "ore_depth")}

    for (registry, tag_id) in tags:
        if registry != "item":
            continue
        members = sorted(resolve(registry, tag_id))
        for m in members:
            c.execute("INSERT INTO tag VALUES(?,?)", (tag_id, m))
            counts["tag"] += 1
        if tag_id.endswith("_food"):
            animal = tag_id[len("minecraft:"): -len("_food")]
            for m in members:
                c.execute("INSERT INTO breeding_food VALUES(?,?)", (animal, m))
                counts["breeding_food"] += 1

    for f in discover_fact_records(data_report):
        cat, sid, fld = f["category"], f["subject_id"], f["fields"]
        if cat == "recipe":
            # Key by the actual crafted item, not the recipe filename (sid). The
            # filename can be a variant id like "<item>_from_<x>_stonecutting";
            # fld["result_item"] is the true output. Multiple recipe files for the
            # same item (crafting + stonecutting) merge under one key.
            result_item = fld.get("result_item") or sid
            for ing in fld.get("ingredients", []):
                c.execute("INSERT INTO recipe VALUES(?,?)", (result_item, ing)); counts["recipe"] += 1
            for slot, group in enumerate(fld.get("ingredient_groups", [])):
                for ing in group:
                    c.execute("INSERT INTO recipe_slot VALUES(?,?,?,?)", (result_item, sid, slot, ing))
        elif cat == "enchantment":
            c.execute("INSERT INTO enchantment VALUES(?,?,?)",
                      (sid, fld.get("max_level"), json.dumps(fld.get("slots")))); counts["enchantment"] += 1
        elif cat == "villager_trade":
            c.execute("INSERT INTO villager_trade VALUES(?,?,?,?,?)",
                      (sid, fld.get("wants_item"), fld.get("wants_count"),
                       fld.get("gives_item"), fld.get("gives_count"))); counts["villager_trade"] += 1
        elif cat == "loot_table":
            for d in fld.get("drops", []):
                c.execute("INSERT INTO loot VALUES(?,?)", (sid, d)); counts["loot"] += 1
        elif cat == "jukebox_song":
            c.execute("INSERT INTO jukebox_song VALUES(?,?)",
                      (sid, fld.get("length_in_seconds"))); counts["jukebox_song"] += 1
        elif cat == "painting_variant":
            c.execute("INSERT INTO painting VALUES(?,?,?)",
                      (sid, fld.get("width"), fld.get("height"))); counts["painting"] += 1

    for row in _load_item_use(data_report):
        c.execute("INSERT INTO item_use VALUES(?,?,?,?,?,?)", row)
        counts["item_use"] += 1

    for row in _load_ore_placements(data_report):
        c.execute("INSERT INTO ore_depth VALUES(?,?,?,?,?,?)", row)
        counts["ore_depth"] += 1

    for idx in _INDEXES:
        c.execute(idx)
    con.commit()
    con.close()
    return counts


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: build_db.py <data_report_dir> <db_path>")
    counts = build(sys.argv[1], sys.argv[2])
    kb = os.path.getsize(sys.argv[2]) / 1024
    print(f"built {sys.argv[2]} ({kb:.0f} KB)")
    for table, n in counts.items():
        print(f"  {table:16} {n}")


if __name__ == "__main__":
    main()
