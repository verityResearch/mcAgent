from src.data_report.schema import adapt_advancement


MINE_STONE = {
    "display": {"title": {"translate": "advancements.story.mine_stone.title"}},
    "criteria": {
        "get_stone": {
            "trigger": "minecraft:inventory_changed",
            "conditions": {"items": [{"items": ["minecraft:stone"]}]},
        }
    },
}


def test_adapt_advancement_collects_criteria():
    records = adapt_advancement("minecraft:story/mine_stone", MINE_STONE)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "advancement"
    assert record["subject_id"] == "minecraft:story/mine_stone"
    assert record["fields"]["criteria_ids"] == ["get_stone"]


def test_adapt_advancement_no_criteria_returns_empty():
    assert adapt_advancement("minecraft:x", {"display": {}, "criteria": {}}) == []
