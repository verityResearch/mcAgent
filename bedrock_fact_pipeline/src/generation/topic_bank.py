from typing import Any, Callable, Dict, List

ARCHETYPES: Dict[str, List[Callable[[Dict[str, Any]], str]]] = {
    "recipe": [
        lambda f: f"What items do you need to craft {f['subject_id']}?",
        lambda f: f"What does crafting {', '.join(f['fields']['ingredients'])} produce?",
    ],
    "loot_table": [
        lambda f: f"What can {f['subject_id']} drop?",
    ],
    "entity": [
        lambda f: f"Can {f['subject_id']} be spawned with a spawn egg?",
        lambda f: f"Can {f['subject_id']} be summoned with the /summon command?",
    ],
    "spawn_rule": [
        lambda f: f"What biomes does {f['subject_id']} naturally spawn in?",
    ],
    "trading": [
        lambda f: f"What does a villager give for {', '.join(f['fields']['wants'])}?",
    ],
    "biome": [
        lambda f: f"What is the temperature of the {f['subject_id']} biome?",
        lambda f: f"What tags does the {f['subject_id']} biome have?",
    ],
}

FREE_RECALL_TOPICS: List[str] = [
    "Early-game strategy: fastest route to obsidian in Bedrock Edition",
    "Efficient early-game food farming in Bedrock Edition",
    "Redstone: building a basic automatic door in Bedrock Edition",
    "Nether survival: navigating without dying to lava in Bedrock Edition",
    "End-game preparation: gearing up for the Ender Dragon in Bedrock Edition",
    "Villager trading: getting the best early trades in Bedrock Edition",
    "Base defense against nighttime mobs in Bedrock Edition",
    "Efficient strip-mining for diamonds in Bedrock Edition",
]


def archetypes_for(category: str) -> List[Callable[[Dict[str, Any]], str]]:
    return ARCHETYPES.get(category, [])


def pick_archetype(category: str, rng) -> Callable[[Dict[str, Any]], str]:
    options = archetypes_for(category)
    if not options:
        raise ValueError(f"No archetypes registered for category {category!r}")
    return rng.choice(options)


def pick_free_recall_topic(rng) -> str:
    return rng.choice(FREE_RECALL_TOPICS)
