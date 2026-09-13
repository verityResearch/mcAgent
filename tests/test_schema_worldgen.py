from src.data_report.schema import adapt_worldgen_structure


VILLAGE_PLAINS = {
    "type": "minecraft:jigsaw",
    "biomes": "#minecraft:has_structure/village_plains",
    "step": "surface_structures",
}


def test_adapt_worldgen_structure():
    records = adapt_worldgen_structure("minecraft:village_plains", VILLAGE_PLAINS)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "worldgen_structure"
    assert record["subject_id"] == "minecraft:village_plains"
    assert record["fields"] == {
        "structure_type": "jigsaw",
        "biomes": "#minecraft:has_structure/village_plains",
        "step": "surface_structures",
    }


def test_adapt_worldgen_structure_missing_type_returns_empty():
    assert adapt_worldgen_structure("minecraft:x", {"biomes": "minecraft:plains"}) == []
