from src.data_report.schema import adapt_loot_table


VILLAGE_HOUSE_LOOT = {
    "pools": [
        {
            "rolls": {"min": 3, "max": 8},
            "entries": [
                {"type": "item", "weight": 2, "name": "minecraft:dandelion"},
                {"type": "item", "name": "minecraft:poppy"},
                {"type": "loot_table", "name": "loot_tables/chests/village/village_bundle.json"},
            ],
        }
    ]
}


def test_adapt_loot_table_collects_item_entries_only():
    records = adapt_loot_table("minecraft:chests/village/village_plains_house", VILLAGE_HOUSE_LOOT)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "loot_table"
    assert record["fields"]["drops"] == ["minecraft:dandelion", "minecraft:poppy"]


def test_adapt_loot_table_no_item_entries_returns_empty():
    only_reference = {"pools": [{"rolls": 1, "entries": [{"type": "loot_table", "name": "x.json"}]}]}
    assert adapt_loot_table("minecraft:x", only_reference) == []


# Real bedrock-samples shape: gameplay/fishing.json references
# gameplay/fishing/junk.json by behavior_pack-relative path.
FISHING_LOOT = {
    "pools": [
        {
            "rolls": 1,
            "entries": [
                {"type": "loot_table", "name": "loot_tables/gameplay/fishing/junk.json", "weight": 10},
            ],
        }
    ]
}
JUNK_LOOT = {
    "pools": [{"rolls": 1, "entries": [{"type": "item", "name": "minecraft:leather_boots", "weight": 10}]}]
}


def test_adapt_loot_table_resolves_cross_file_reference_when_resolver_given():
    lookup = {"loot_tables/gameplay/fishing/junk.json": JUNK_LOOT}
    records = adapt_loot_table("minecraft:gameplay/fishing", FISHING_LOOT, resolve=lookup.get)
    assert len(records) == 1
    assert records[0]["fields"]["drops"] == ["minecraft:leather_boots"]


def test_adapt_loot_table_unresolvable_reference_is_ignored_not_an_error():
    records = adapt_loot_table("minecraft:gameplay/fishing", FISHING_LOOT, resolve=lambda name: None)
    assert records == []


def test_adapt_loot_table_cross_file_reference_cycle_does_not_hang():
    a = {"pools": [{"rolls": 1, "entries": [
        {"type": "item", "name": "minecraft:a_item"},
        {"type": "loot_table", "name": "b.json"},
    ]}]}
    b = {"pools": [{"rolls": 1, "entries": [
        {"type": "item", "name": "minecraft:b_item"},
        {"type": "loot_table", "name": "a.json"},
    ]}]}
    lookup = {"a.json": a, "b.json": b}
    records = adapt_loot_table("minecraft:a", a, resolve=lookup.get)
    assert len(records) == 1
    assert records[0]["fields"]["drops"] == ["minecraft:a_item", "minecraft:b_item"]
