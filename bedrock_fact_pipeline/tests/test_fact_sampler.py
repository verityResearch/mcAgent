import json
import random

from src.generation.fact_sampler import (
    _strip_json_comments,
    category_for_path,
    discover_fact_records,
    sample_fact_records,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_raw(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_category_for_path():
    assert category_for_path(".../behavior_pack/recipes/acacia_boat.json") == "recipe"
    assert category_for_path(".../behavior_pack/loot_tables/chests/village/x.json") == "loot_table"
    assert category_for_path(".../behavior_pack/entities/bee.json") == "entity"
    assert category_for_path(".../behavior_pack/spawn_rules/bee.json") == "spawn_rule"
    assert category_for_path(".../behavior_pack/trading/armorer_trades.json") == "trading"
    assert category_for_path(".../behavior_pack/biomes/beach.biome.json") == "biome"
    assert category_for_path(".../behavior_pack/shapes/x.json") is None


def test_discover_fact_records(tmp_path):
    _write(tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/recipes/acacia_boat.json", {
        "minecraft:recipe_shaped": {
            "key": {"#": {"item": "minecraft:acacia_planks"}},
            "result": {"item": "minecraft:acacia_boat"},
        }
    })
    _write(tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/biomes/beach.biome.json", {
        "minecraft:biome": {"components": {"minecraft:climate": {"temperature": 0.8, "downfall": 0.4}}}
    })
    _write(tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/shapes/camera.json", {"unused": True})

    records = discover_fact_records(str(tmp_path))
    categories = sorted(r["category"] for r in records)
    assert categories == ["biome", "recipe"]
    recipe = next(r for r in records if r["category"] == "recipe")
    assert recipe["subject_id"] == "minecraft:acacia_boat"
    biome = next(r for r in records if r["category"] == "biome")
    assert biome["subject_id"] == "minecraft:beach"


def test_sample_fact_records_deterministic_and_bounded():
    records = [
        {"category": "recipe", "subject_id": f"minecraft:item{i}", "fields": {}}
        for i in range(5)
    ]
    picked = sample_fact_records(records, n=2, rng=random.Random(0))
    assert len(picked) == 2
    assert sample_fact_records(records, n=10, rng=random.Random(0)) == records
    only_recipes = sample_fact_records(records, n=2, rng=random.Random(0), category="recipe")
    assert all(r["category"] == "recipe" for r in only_recipes)


def test_strip_json_comments_removes_line_comments():
    text = (
        "{\n"
        "  // Every four ticks, the sensor scans for threats.\n"
        '  "minecraft:entity_sensor": {}\n'
        "}\n"
    )
    stripped = _strip_json_comments(text)
    assert "//" not in stripped
    assert json.loads(stripped) == {"minecraft:entity_sensor": {}}


def test_strip_json_comments_preserves_slashes_inside_string_values():
    # Real bedrock-samples recipe pattern row uses "/" as a key symbol,
    # producing a literal "///" string value that must survive untouched.
    text = '{\n  "pattern": ["///", " / ", "/_/"],\n  "url": "http://example.com"\n}\n'
    stripped = _strip_json_comments(text)
    assert json.loads(stripped) == {
        "pattern": ["///", " / ", "/_/"],
        "url": "http://example.com",
    }


def test_discover_fact_records_tolerates_line_comments(tmp_path):
    _write_raw(
        tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/entities/armadillo.json",
        "{\n"
        '  "minecraft:entity": {\n'
        '    "description": {"identifier": "minecraft:armadillo", "is_spawnable": true,\n'
        '                     "is_summonable": true, "spawn_category": "creature"},\n'
        "    // this comment used to crash discovery entirely\n"
        '    "component_groups": {}\n'
        "  }\n"
        "}\n",
    )

    records = discover_fact_records(str(tmp_path))

    assert len(records) == 1
    assert records[0]["subject_id"] == "minecraft:armadillo"


def test_discover_fact_records_resolves_cross_file_loot_table_references(tmp_path):
    _write(
        tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/loot_tables/gameplay/fishing.json",
        {"pools": [{"rolls": 1, "entries": [
            {"type": "loot_table", "name": "loot_tables/gameplay/fishing/junk.json", "weight": 10},
        ]}]},
    )
    _write(
        tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/loot_tables/gameplay/fishing/junk.json",
        {"pools": [{"rolls": 1, "entries": [{"type": "item", "name": "minecraft:leather_boots"}]}]},
    )

    records = discover_fact_records(str(tmp_path))
    by_subject = {r["subject_id"]: r for r in records}

    assert by_subject["minecraft:gameplay/fishing"]["fields"]["drops"] == ["minecraft:leather_boots"]
    assert by_subject["minecraft:gameplay/fishing/junk"]["fields"]["drops"] == ["minecraft:leather_boots"]


def test_discover_fact_records_skips_unparseable_file(tmp_path):
    _write(tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/recipes/acacia_boat.json", {
        "minecraft:recipe_shaped": {
            "key": {"#": {"item": "minecraft:acacia_planks"}},
            "result": {"item": "minecraft:acacia_boat"},
        }
    })
    _write_raw(
        tmp_path / "bedrock-samples-1.26.30.5/behavior_pack/recipes/broken.json",
        "{ this is not valid json at all",
    )

    records = discover_fact_records(str(tmp_path))

    assert len(records) == 1
    assert records[0]["subject_id"] == "minecraft:acacia_boat"
