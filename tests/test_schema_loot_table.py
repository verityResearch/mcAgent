from src.data_report.schema import adapt_loot_table


GRASS_LOOT_TABLE = {
    "type": "minecraft:block",
    "pools": [
        {
            "rolls": 1,
            "entries": [{"type": "minecraft:item", "name": "minecraft:wheat_seeds"}],
            "conditions": [{"condition": "minecraft:survives_explosion"}],
        }
    ],
}


def test_adapt_loot_table_collects_drops():
    records = adapt_loot_table("minecraft:blocks/grass", GRASS_LOOT_TABLE)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "loot_table"
    assert record["subject_id"] == "minecraft:blocks/grass"
    assert record["fields"]["drops"] == ["minecraft:wheat_seeds"]


def test_adapt_loot_table_empty_pools_returns_empty():
    assert adapt_loot_table("minecraft:empty", {"type": "minecraft:empty", "pools": []}) == []


DEEPSLATE_IRON_LOOT_TABLE = {
    "type": "minecraft:block",
    "pools": [
        {
            "rolls": 1,
            "entries": [
                {
                    "type": "minecraft:alternatives",
                    "children": [
                        {"type": "minecraft:item", "name": "minecraft:deepslate_iron_ore"},
                        {"type": "minecraft:item", "name": "minecraft:raw_iron"},
                    ],
                }
            ],
        }
    ],
}


def test_adapt_loot_table_recurses_into_alternatives_children():
    records = adapt_loot_table("minecraft:blocks/deepslate_iron_ore", DEEPSLATE_IRON_LOOT_TABLE)
    assert len(records) == 1
    assert records[0]["fields"]["drops"] == ["minecraft:deepslate_iron_ore", "minecraft:raw_iron"]


TAG_LOOT_TABLE = {
    "type": "minecraft:block",
    "pools": [
        {
            "rolls": 1,
            "entries": [{"type": "minecraft:tag", "name": "minecraft:logs"}],
        }
    ],
}


def test_adapt_loot_table_tag_entry_becomes_hash_reference():
    records = adapt_loot_table("minecraft:blocks/tagged", TAG_LOOT_TABLE)
    assert len(records) == 1
    assert records[0]["fields"]["drops"] == ["#minecraft:logs"]


MIXED_LOOT_TABLE = {
    "type": "minecraft:block",
    "pools": [
        {
            "rolls": 1,
            "entries": [{"type": "minecraft:item", "name": "minecraft:apple"}],
        },
        {
            "rolls": 1,
            "entries": [
                {
                    "type": "minecraft:alternatives",
                    "children": [
                        {"type": "minecraft:item", "name": "minecraft:stick"},
                        {"type": "minecraft:item", "name": "minecraft:oak_sapling"},
                    ],
                }
            ],
        },
    ],
}


def test_adapt_loot_table_mixed_pools_collect_all_drops():
    records = adapt_loot_table("minecraft:blocks/mixed", MIXED_LOOT_TABLE)
    assert len(records) == 1
    assert records[0]["fields"]["drops"] == [
        "minecraft:apple",
        "minecraft:oak_sapling",
        "minecraft:stick",
    ]
