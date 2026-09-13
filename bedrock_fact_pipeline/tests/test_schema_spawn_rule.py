from src.data_report.schema import adapt_spawn_rule


BEE_SPAWN_RULE = {
    "minecraft:spawn_rules": {
        "description": {"identifier": "minecraft:bee"},
        "conditions": [
            {
                "minecraft:brightness_filter": {"min": 7, "max": 15},
                "minecraft:biome_filter": [
                    {"test": "has_biome_tag", "operator": "==", "value": "plains"},
                    {"test": "has_biome_tag", "operator": "==", "value": "sunflower_plains"},
                ],
            }
        ],
    }
}


def test_adapt_spawn_rule_collects_biome_tags():
    records = adapt_spawn_rule("minecraft:bee", BEE_SPAWN_RULE)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "spawn_rule"
    assert record["fields"]["biome_tags"] == ["plains", "sunflower_plains"]


def test_adapt_spawn_rule_no_biome_filter_returns_empty():
    no_filter = {"minecraft:spawn_rules": {"conditions": [{"minecraft:weight": {"default": 10}}]}}
    assert adapt_spawn_rule("minecraft:x", no_filter) == []


# Real bedrock-samples axolotl.json: biome_filter is a bare dict, not a list.
AXOLOTL_SPAWN_RULE = {
    "minecraft:spawn_rules": {
        "description": {"identifier": "minecraft:axolotl"},
        "conditions": [
            {"minecraft:biome_filter": {"test": "has_biome_tag", "operator": "==", "value": "lush_caves"}}
        ],
    }
}


def test_adapt_spawn_rule_bare_dict_filter():
    records = adapt_spawn_rule("minecraft:axolotl", AXOLOTL_SPAWN_RULE)
    assert records[0]["fields"]["biome_tags"] == ["lush_caves"]


# Real bedrock-samples armadillo.json: all_of containing a negated ("not") leaf,
# which must NOT be reported as a positive spawn biome.
ARMADILLO_SPAWN_RULE = {
    "minecraft:spawn_rules": {
        "description": {"identifier": "minecraft:armadillo"},
        "conditions": [
            {
                "minecraft:biome_filter": {
                    "all_of": [
                        {"test": "has_biome_tag", "value": "mesa"},
                        {"test": "has_biome_tag", "operator": "not", "value": "plateau"},
                    ]
                }
            }
        ],
    }
}


def test_adapt_spawn_rule_all_of_skips_negated_leaf():
    records = adapt_spawn_rule("minecraft:armadillo", ARMADILLO_SPAWN_RULE)
    assert records[0]["fields"]["biome_tags"] == ["mesa"]


# Real bedrock-samples bat.json: any_of, both branches positive.
BAT_SPAWN_RULE = {
    "minecraft:spawn_rules": {
        "description": {"identifier": "minecraft:bat"},
        "conditions": [
            {
                "minecraft:biome_filter": {
                    "any_of": [
                        {"test": "has_biome_tag", "operator": "==", "value": "caves"},
                        {"test": "has_biome_tag", "operator": "==", "value": "animal"},
                    ]
                }
            }
        ],
    }
}


def test_adapt_spawn_rule_any_of_collects_all_branches():
    records = adapt_spawn_rule("minecraft:bat", BAT_SPAWN_RULE)
    assert records[0]["fields"]["biome_tags"] == ["animal", "caves"]


# Real bedrock-samples nautilus.json: a list mixing a leaf with a nested
# none_of; the entire none_of subtree must be excluded, not just its negated leaf.
NAUTILUS_SPAWN_RULE = {
    "minecraft:spawn_rules": {
        "description": {"identifier": "minecraft:nautilus"},
        "conditions": [
            {
                "minecraft:biome_filter": [
                    {"test": "has_biome_tag", "operator": "==", "value": "ocean"},
                    {
                        "none_of": [
                            {"test": "has_biome_tag", "operator": "==", "value": "frozen"},
                            {
                                "all_of": [
                                    {"test": "has_biome_tag", "operator": "==", "value": "cold"},
                                    {"test": "has_biome_tag", "operator": "!=", "value": "deep"},
                                ]
                            },
                        ]
                    },
                ]
            }
        ],
    }
}


def test_adapt_spawn_rule_none_of_subtree_excluded_entirely():
    records = adapt_spawn_rule("minecraft:nautilus", NAUTILUS_SPAWN_RULE)
    assert records[0]["fields"]["biome_tags"] == ["ocean"]


def test_adapt_spawn_rule_nested_any_of_all_of_skips_negated_leaf():
    nested = {
        "minecraft:spawn_rules": {
            "description": {"identifier": "minecraft:nautilus"},
            "conditions": [
                {
                    "minecraft:biome_filter": [
                        {"test": "has_biome_tag", "operator": "==", "value": "ocean"},
                        {
                            "any_of": [
                                {"test": "has_biome_tag", "operator": "==", "value": "frozen"},
                                {
                                    "all_of": [
                                        {"test": "has_biome_tag", "operator": "==", "value": "cold"},
                                        {"test": "has_biome_tag", "operator": "!=", "value": "deep"},
                                    ]
                                },
                            ]
                        },
                    ]
                }
            ],
        }
    }
    records = adapt_spawn_rule("minecraft:nautilus", nested)
    assert records[0]["fields"]["biome_tags"] == ["cold", "frozen", "ocean"]
