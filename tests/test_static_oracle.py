from src.verification.static_oracle import check_fact_seeded


FACT = {
    "category": "recipe",
    "subject_id": "minecraft:piston",
    "fields": {"result_item": "minecraft:piston", "ingredients": ["minecraft:oak_planks", "minecraft:iron_ingot"]},
}


def test_check_fact_seeded_admits_exact_match():
    candidate = {"answer": "You need oak planks and an iron ingot.",
                 "structured_claim": {"result_item": "minecraft:piston",
                                       "ingredients": ["minecraft:iron_ingot", "minecraft:oak_planks"]}}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is True
    assert result["errors"] == []


def test_check_fact_seeded_rejects_field_mismatch():
    candidate = {"answer": "You need oak planks and an iron ingot.",
                 "structured_claim": {"result_item": "minecraft:sticky_piston",
                                       "ingredients": ["minecraft:iron_ingot", "minecraft:oak_planks"]}}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert any("result_item" in e for e in result["errors"])
    assert result["failure_kind"] == "fact_mismatch"


def test_check_fact_seeded_rejects_missing_structured_claim():
    result = check_fact_seeded({"answer": "some answer"}, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_claim"


def test_check_fact_seeded_works_for_any_category():
    tag_fact = {"category": "tag", "subject_id": "minecraft:logs",
                "fields": {"members": ["minecraft:oak_log", "minecraft:spruce_log"]}}
    candidate = {"answer": "Oak log and spruce log.",
                 "structured_claim": {"members": ["minecraft:spruce_log", "minecraft:oak_log"]}}
    result = check_fact_seeded(candidate, tag_fact)
    assert result["success"] is True


def test_check_fact_seeded_rejects_noncomparable_list_without_crashing():
    fact = {"category": "recipe", "subject_id": "x",
            "fields": {"ingredients": ["minecraft:oak_planks", "minecraft:iron_ingot"]}}
    candidate = {"answer": "some answer",
                 "structured_claim": {"ingredients": [{"item": "minecraft:oak_planks"}, "minecraft:iron_ingot"]}}
    result = check_fact_seeded(candidate, fact)
    assert result["success"] is False
    assert result["failure_kind"] == "fact_mismatch"


def test_check_fact_seeded_admits_across_all_data_report_categories():
    cases = [
        {"category": "recipe", "subject_id": "x", "fields": {"result_item": "minecraft:piston"}},
        {"category": "loot_table", "subject_id": "x", "fields": {"drops": ["minecraft:wheat_seeds"]}},
        {"category": "advancement", "subject_id": "x", "fields": {"criteria_ids": ["get_stone"]}},
        {"category": "worldgen_structure", "subject_id": "x", "fields": {"structure_type": "jigsaw"}},
    ]
    for fact in cases:
        candidate = {"answer": "some answer", "structured_claim": dict(fact["fields"])}
        result = check_fact_seeded(candidate, fact)
        assert result["success"] is True, fact["category"]


def test_check_fact_seeded_rejects_missing_answer():
    candidate = {"structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"


def test_check_fact_seeded_rejects_non_string_answer():
    """The regression this check exists for: a model returning a raw JSON
    array (identical to a structured_claim list field) instead of a
    natural-language sentence -- found via a live base-vs-fine-tuned model
    comparison, where structured_claim matching alone let it through."""
    candidate = {"answer": ["minecraft:oak_planks", "minecraft:iron_ingot"],
                 "structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"


def test_check_fact_seeded_rejects_empty_answer():
    candidate = {"answer": "   ", "structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"
