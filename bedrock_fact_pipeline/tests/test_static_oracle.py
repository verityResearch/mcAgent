from src.verification.static_oracle import check_fact_seeded


FACT = {
    "category": "recipe",
    "subject_id": "minecraft:acacia_boat",
    "fields": {"result_item": "minecraft:acacia_boat", "ingredients": ["minecraft:acacia_planks"]},
}


def test_check_fact_seeded_admits_exact_match():
    candidate = {"answer": "You need acacia planks.",
                 "structured_claim": {"result_item": "minecraft:acacia_boat",
                                       "ingredients": ["minecraft:acacia_planks"]}}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is True
    assert result["errors"] == []


def test_check_fact_seeded_rejects_field_mismatch():
    candidate = {"answer": "You need acacia planks.",
                 "structured_claim": {"result_item": "minecraft:oak_boat",
                                       "ingredients": ["minecraft:acacia_planks"]}}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert any("result_item" in e for e in result["errors"])
    assert result["failure_kind"] == "fact_mismatch"


def test_check_fact_seeded_rejects_missing_structured_claim():
    result = check_fact_seeded({"answer": "some answer"}, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_claim"


def test_check_fact_seeded_works_for_any_category():
    biome_fact = {"category": "biome", "subject_id": "minecraft:beach",
                  "fields": {"tags": ["beach", "warm"]}}
    candidate = {"answer": "Beach and warm.", "structured_claim": {"tags": ["warm", "beach"]}}
    result = check_fact_seeded(candidate, biome_fact)
    assert result["success"] is True


def test_check_fact_seeded_rejects_noncomparable_list_without_crashing():
    fact = {"category": "recipe", "subject_id": "x",
            "fields": {"ingredients": ["minecraft:acacia_planks", "minecraft:iron_ingot"]}}
    candidate = {"answer": "some answer",
                 "structured_claim": {"ingredients": [{"item": "minecraft:acacia_planks"}, "minecraft:iron_ingot"]}}
    result = check_fact_seeded(candidate, fact)
    assert result["success"] is False
    assert result["failure_kind"] == "fact_mismatch"


def test_check_fact_seeded_rejects_missing_answer():
    candidate = {"structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"


def test_check_fact_seeded_rejects_non_string_answer():
    """The regression this check exists for: a model returning a raw JSON
    array (identical to a structured_claim list field) instead of a
    natural-language sentence -- found via a live base-vs-fine-tuned model
    comparison on the sibling fact_pipeline, where structured_claim matching
    alone let it through."""
    candidate = {"answer": ["minecraft:acacia_planks"], "structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"


def test_check_fact_seeded_rejects_empty_answer():
    candidate = {"answer": "   ", "structured_claim": dict(FACT["fields"])}
    result = check_fact_seeded(candidate, FACT)
    assert result["success"] is False
    assert result["failure_kind"] == "malformed_answer"
