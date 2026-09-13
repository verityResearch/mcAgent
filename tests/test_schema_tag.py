from src.data_report.schema import adapt_tag


LOGS_TAG = {"values": ["minecraft:oak_log", "minecraft:spruce_log"]}


def test_adapt_tag_collects_members():
    records = adapt_tag("minecraft:logs", LOGS_TAG)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "tag"
    assert record["subject_id"] == "minecraft:logs"
    assert record["fields"]["members"] == ["minecraft:oak_log", "minecraft:spruce_log"]


def test_adapt_tag_empty_values_returns_empty():
    assert adapt_tag("minecraft:empty", {"values": []}) == []


def test_adapt_tag_object_form_values_resolve():
    raw = {"values": [{"id": "minecraft:oak_log", "required": False}]}
    records = adapt_tag("minecraft:logs", raw)
    assert len(records) == 1
    assert records[0]["fields"]["members"] == ["minecraft:oak_log"]


def test_adapt_tag_mixed_string_and_object_values_resolve():
    raw = {
        "values": [
            "minecraft:oak_log",
            {"id": "minecraft:spruce_log", "required": False},
        ]
    }
    records = adapt_tag("minecraft:logs", raw)
    assert len(records) == 1
    assert records[0]["fields"]["members"] == ["minecraft:oak_log", "minecraft:spruce_log"]


def test_adapt_tag_object_form_missing_id_is_skipped():
    raw = {
        "values": [
            {"required": False},
            {"id": 123},
            {"id": "minecraft:oak_log"},
        ]
    }
    records = adapt_tag("minecraft:logs", raw)
    assert len(records) == 1
    assert records[0]["fields"]["members"] == ["minecraft:oak_log"]
