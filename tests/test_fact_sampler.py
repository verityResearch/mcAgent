import json
import random

from src.generation.fact_sampler import category_for_path, discover_fact_records, sample_fact_records


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_category_for_path():
    assert category_for_path("data/minecraft/recipe/piston.json") == "recipe"
    assert category_for_path("data/minecraft/loot_table/blocks/grass.json") == "loot_table"
    assert category_for_path("data/minecraft/tags/blocks/logs.json") == "tag"
    assert category_for_path("data/minecraft/worldgen/structure/village_plains.json") == "worldgen_structure"
    assert category_for_path("data/minecraft/some_other_thing/x.json") is None


def test_discover_fact_records(tmp_path):
    _write(tmp_path / "data/minecraft/recipe/piston.json", {
        "type": "minecraft:crafting_shaped",
        "pattern": ["###", "#X#", "#Y#"],
        "key": {
            "#": {"item": "minecraft:oak_planks"},
            "X": {"item": "minecraft:cobblestone"},
            "Y": {"item": "minecraft:iron_ingot"},
        },
        "result": {"id": "minecraft:piston", "count": 1},
    })
    _write(tmp_path / "data/minecraft/tags/blocks/logs.json", {
        "values": ["minecraft:oak_log", "minecraft:spruce_log"],
    })
    _write(tmp_path / "data/minecraft/some_other_thing/x.json", {"unused": True})

    records = discover_fact_records(str(tmp_path))
    categories = sorted(r["category"] for r in records)
    assert categories == ["recipe", "tag"]
    recipe = next(r for r in records if r["category"] == "recipe")
    assert recipe["subject_id"] == "minecraft:piston"
    tag = next(r for r in records if r["category"] == "tag")
    assert tag["subject_id"] == "minecraft:blocks/logs"


def test_category_for_path_excludes_nested_experimental_datapacks():
    # Real 26.2 data report: optional/experimental vanilla datapacks
    # (trade_rebalance, minecart_improvements, redstone_experiments) each
    # bundle a full second "data/<namespace>/<registry>/..." tree nested
    # under datapacks/<pack_name>/. _subject_id's single-level parsing can't
    # represent that nesting, so these are out of scope for now rather than
    # producing a garbled compound identifier.
    path = "data/minecraft/datapacks/trade_rebalance/data/minecraft/tags/villager_trade/librarian/level_2.json"
    assert category_for_path(path) is None


def test_discover_fact_records_skips_nested_experimental_datapack_content(tmp_path):
    _write(
        tmp_path / "data/minecraft/datapacks/trade_rebalance/data/minecraft/tags/villager_trade/librarian/level_2.json",
        {"values": ["minecraft:emerald"]},
    )
    _write(tmp_path / "data/minecraft/tags/blocks/logs.json", {"values": ["minecraft:oak_log"]})

    records = discover_fact_records(str(tmp_path))

    assert len(records) == 1
    assert records[0]["subject_id"] == "minecraft:blocks/logs"


def test_discover_fact_records_worldgen_structure(tmp_path):
    _write(tmp_path / "data/minecraft/worldgen/structure/village_plains.json", {
        "type": "minecraft:jigsaw",
        "biomes": "#minecraft:has_structure/village_plains",
        "step": "surface_structures",
    })
    records = discover_fact_records(str(tmp_path))
    assert len(records) == 1
    assert records[0]["category"] == "worldgen_structure"
    assert records[0]["subject_id"] == "minecraft:village_plains"


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


def test_sample_fact_records_category_filter_matches_factrecord_vocabulary():
    records = [
        {"category": "tag", "subject_id": "minecraft:logs", "fields": {}},
        {"category": "worldgen_structure", "subject_id": "minecraft:village_plains", "fields": {}},
        {"category": "recipe", "subject_id": "minecraft:piston", "fields": {}},
    ]
    only_tags = sample_fact_records(records, n=10, rng=random.Random(0), category="tag")
    assert only_tags == [records[0]]
    only_structures = sample_fact_records(records, n=10, rng=random.Random(0), category="worldgen_structure")
    assert only_structures == [records[1]]
