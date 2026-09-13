from src.data_report.schema import adapt_biome


BEACH_BIOME = {
    "minecraft:biome": {
        "description": {"identifier": "minecraft:beach"},
        "components": {
            "minecraft:climate": {"downfall": 0.4, "temperature": 0.8},
            "minecraft:tags": {"tags": ["beach", "monster", "overworld", "warm"]},
        },
    }
}


def test_adapt_biome_extracts_climate_and_tags():
    records = adapt_biome("minecraft:beach", BEACH_BIOME)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "biome"
    assert record["fields"] == {
        "temperature": 0.8,
        "downfall": 0.4,
        "tags": ["beach", "monster", "overworld", "warm"],
    }


def test_adapt_biome_missing_climate_keeps_tags():
    no_climate = {"minecraft:biome": {"components": {"minecraft:tags": {"tags": ["nether"]}}}}
    records = adapt_biome("minecraft:x", no_climate)
    assert len(records) == 1
    assert records[0]["fields"] == {"tags": ["nether"]}


def test_adapt_biome_no_components_returns_empty():
    assert adapt_biome("minecraft:x", {"minecraft:biome": {}}) == []
