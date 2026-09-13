import random

from src.generation.topic_bank import (
    ARCHETYPES,
    FREE_RECALL_TOPICS,
    pick_archetype,
    pick_free_recall_topic,
)

_CATEGORIES = ["recipe", "loot_table", "advancement", "tag", "worldgen_structure"]


def test_every_category_has_at_least_one_archetype():
    for category in _CATEGORIES:
        assert ARCHETYPES.get(category), category


def test_pick_archetype_formats_with_fact_record():
    fact = {
        "category": "recipe",
        "subject_id": "minecraft:piston",
        "fields": {"ingredients": ["minecraft:oak_planks", "minecraft:cobblestone", "minecraft:iron_ingot"]},
    }
    archetype = pick_archetype("recipe", random.Random(0))
    question = archetype(fact)
    assert isinstance(question, str)
    assert "piston" in question or "oak_planks" in question


def test_pick_archetype_unknown_category_raises():
    try:
        pick_archetype("nonexistent", random.Random(0))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_free_recall_topics_nonempty_and_pickable():
    assert FREE_RECALL_TOPICS
    topic = pick_free_recall_topic(random.Random(0))
    assert topic in FREE_RECALL_TOPICS


def test_tag_archetype_has_positive_and_negative_variants():
    assert len(ARCHETYPES["tag"]) == 2


def test_tag_negative_archetype_picks_a_non_member():
    fact = {"category": "tag", "subject_id": "minecraft:logs", "fields": {"members": ["minecraft:oak_log"]}}
    negative_archetype = ARCHETYPES["tag"][1]
    question = negative_archetype(fact)
    assert question.startswith("Is minecraft:") and question.endswith("a member of the tag minecraft:logs?")
    mentioned = question[len("Is "):question.index(" a member")]
    assert mentioned not in fact["fields"]["members"]


def test_tag_negative_archetype_falls_back_when_pool_exhausted():
    # every distractor-pool candidate is itself a true member here; the
    # function must still return a question, not crash or loop forever
    from src.generation.topic_bank import _TAG_DISTRACTOR_POOL

    fact = {"category": "tag", "subject_id": "minecraft:everything", "fields": {"members": _TAG_DISTRACTOR_POOL}}
    negative_archetype = ARCHETYPES["tag"][1]
    question = negative_archetype(fact)
    assert question.startswith("Is minecraft:")


def test_tag_negative_archetype_is_deterministic_per_fact():
    fact = {"category": "tag", "subject_id": "minecraft:logs", "fields": {"members": ["minecraft:oak_log"]}}
    negative_archetype = ARCHETYPES["tag"][1]
    assert negative_archetype(fact) == negative_archetype(fact)
