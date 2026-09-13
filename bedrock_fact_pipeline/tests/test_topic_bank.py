import random

from src.generation.topic_bank import (
    ARCHETYPES,
    FREE_RECALL_TOPICS,
    pick_archetype,
    pick_free_recall_topic,
)

_CATEGORIES = ["recipe", "loot_table", "entity", "spawn_rule", "trading", "biome"]


def test_every_category_has_at_least_one_archetype():
    for category in _CATEGORIES:
        assert ARCHETYPES.get(category), category


def test_pick_archetype_formats_with_fact_record():
    fact = {
        "category": "recipe",
        "subject_id": "minecraft:acacia_boat",
        "fields": {"ingredients": ["minecraft:acacia_planks"], "result_item": "minecraft:acacia_boat"},
    }
    archetype = pick_archetype("recipe", random.Random(0))
    question = archetype(fact)
    assert isinstance(question, str)
    assert "acacia_boat" in question or "acacia_planks" in question


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
