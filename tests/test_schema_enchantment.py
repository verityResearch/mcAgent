from src.data_report.schema import adapt_enchantment


SHARPNESS = {
    "anvil_cost": 1,
    "max_level": 5,
    "weight": 10,
    "slots": ["mainhand"],
    "supported_items": "#minecraft:enchantable/sharp_weapon",
    "effects": {"minecraft:damage": [{"...": "nested curve, deliberately ignored"}]},
}


def test_adapt_enchantment_extracts_flat_fields():
    records = adapt_enchantment("minecraft:sharpness", SHARPNESS)
    assert len(records) == 1
    r = records[0]
    assert r["category"] == "enchantment"
    assert r["subject_id"] == "minecraft:sharpness"
    assert r["fields"] == {
        "max_level": 5, "weight": 10, "anvil_cost": 1, "slots": ["mainhand"],
    }
    # nested curve fields must not leak in
    assert "effects" not in r["fields"] and "max_cost" not in r["fields"]


def test_adapt_enchantment_sorts_slots():
    records = adapt_enchantment("minecraft:x", {"max_level": 1, "slots": ["offhand", "mainhand"]})
    assert records[0]["fields"]["slots"] == ["mainhand", "offhand"]


def test_adapt_enchantment_requires_max_level():
    assert adapt_enchantment("minecraft:x", {"weight": 2}) == []
    assert adapt_enchantment("minecraft:x", {"max_level": "five"}) == []
