from src.verification.static_oracle import _is_refusal, check_fact_seeded


FACT = {"category": "recipe", "subject_id": "minecraft:piston",
        "fields": {"result_item": "minecraft:piston"}}


def test_catches_real_refusals():
    assert _is_refusal("The fact does not provide information about what crafting minecraft:leather produces.")
    assert _is_refusal("The fact does not specify what minecraft:waxed_copper_block produces.")
    assert _is_refusal("The given fact does not provide information about this recipe.")
    assert _is_refusal("There is no information about the result.")
    assert _is_refusal("I cannot determine what this produces.")


def test_does_not_flag_ordinary_prose():
    # the exact real false positive that motivated tightening the regex
    assert not _is_refusal("Wild wolves, on the other hand, do not provide these benefits and will not help you.")
    assert not _is_refusal("Beds do not provide protection from creepers.")
    assert not _is_refusal("minecraft:diorite")
    assert not _is_refusal("You craft it from iron and planks.")
    assert not _is_refusal("This recipe gives you a piston.")


def test_check_fact_seeded_rejects_refusal_even_when_claim_matches():
    # the exact bug: structured_claim is correct, so the field diff would pass,
    # but the answer text is a refusal -> must be rejected as refusal_answer.
    cand = {"answer": "The fact does not provide information about what this produces.",
            "structured_claim": {"result_item": "minecraft:piston"}}
    res = check_fact_seeded(cand, FACT)
    assert res["success"] is False
    assert res["failure_kind"] == "refusal_answer"


def test_check_fact_seeded_still_admits_real_answer():
    cand = {"answer": "It produces a piston.", "structured_claim": {"result_item": "minecraft:piston"}}
    res = check_fact_seeded(cand, FACT)
    assert res["success"] is True
