from src.data_report.schema import adapt_entity


BEE_ENTITY = {
    "minecraft:entity": {
        "description": {
            "identifier": "minecraft:bee",
            "spawn_category": "creature",
            "is_spawnable": True,
            "is_summonable": True,
        },
        "component_groups": {"bee_baby": {}},
    }
}


def test_adapt_entity_extracts_description_flags():
    records = adapt_entity("minecraft:bee", BEE_ENTITY)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "entity"
    assert record["fields"] == {
        "spawn_category": "creature",
        "is_spawnable": True,
        "is_summonable": True,
    }


def test_adapt_entity_missing_description_returns_empty():
    assert adapt_entity("minecraft:x", {"minecraft:entity": {}}) == []


# Real bedrock-samples villager.json: is_spawnable/is_summonable present,
# but no spawn_category — a real, common mob the adapter must not drop.
VILLAGER_ENTITY = {
    "minecraft:entity": {
        "description": {
            "identifier": "minecraft:villager",
            "is_summonable": True,
            "is_spawnable": True,
        },
    }
}


def test_adapt_entity_without_spawn_category_still_extracts():
    records = adapt_entity("minecraft:villager", VILLAGER_ENTITY)
    assert len(records) == 1
    assert records[0]["fields"] == {
        "spawn_category": None,
        "is_spawnable": True,
        "is_summonable": True,
    }
