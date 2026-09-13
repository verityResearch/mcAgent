from src.data_report.schema import adapt_recipe, make_fact_record


SHAPED_ACACIA_BOAT = {
    "minecraft:recipe_shaped": {
        "description": {"identifier": "minecraft:acacia_boat"},
        "key": {"#": {"item": "minecraft:acacia_planks"}},
        "result": {"item": "minecraft:acacia_boat"},
    }
}

FURNACE_BEEF = {
    "minecraft:recipe_furnace": {
        "description": {"identifier": "minecraft:furnace_beef"},
        "input": "minecraft:beef",
        "output": "minecraft:cooked_beef",
    }
}


def test_make_fact_record_shape():
    record = make_fact_record("recipe", "minecraft:acacia_boat", {"a": 1})
    assert record == {"category": "recipe", "subject_id": "minecraft:acacia_boat", "fields": {"a": 1}}


def test_adapt_shaped_recipe():
    records = adapt_recipe("minecraft:acacia_boat", SHAPED_ACACIA_BOAT)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "recipe"
    assert record["fields"]["recipe_type"] == "shaped"
    assert record["fields"]["result_item"] == "minecraft:acacia_boat"
    assert record["fields"]["ingredients"] == ["minecraft:acacia_planks"]


def test_adapt_furnace_recipe_bare_string_shape():
    records = adapt_recipe("minecraft:furnace_beef", FURNACE_BEEF)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "furnace"
    assert record["fields"]["result_item"] == "minecraft:cooked_beef"
    assert record["fields"]["ingredients"] == ["minecraft:beef"]


def test_adapt_recipe_unknown_type_returns_empty():
    assert adapt_recipe("minecraft:x", {"minecraft:recipe_made_up_type": {}}) == []


# Real bedrock-samples crafting_table.json: the "key" ingredient is a tag
# reference ({"tag": ...}), not an item reference ({"item": ...}).
CRAFTING_TABLE_TAG_KEY = {
    "minecraft:recipe_shaped": {
        "description": {"identifier": "minecraft:WorkBench_recipeId"},
        "pattern": ["AA", "AA"],
        "key": {"A": {"tag": "minecraft:planks"}},
        "result": {"item": "crafting_table"},
    }
}


def test_adapt_shaped_recipe_tag_ingredient():
    records = adapt_recipe("minecraft:crafting_table", CRAFTING_TABLE_TAG_KEY)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["result_item"] == "crafting_table"
    assert record["fields"]["ingredients"] == ["minecraft:planks"]


# Real bedrock-samples furnace_log.json: furnace "input" is a tag reference.
FURNACE_LOG_TAG_INPUT = {
    "minecraft:recipe_furnace": {
        "description": {"identifier": "minecraft:furnace_log"},
        "input": {"tag": "minecraft:logs_that_burn"},
        "output": "minecraft:coal:1",
    }
}


def test_adapt_furnace_recipe_tag_input():
    records = adapt_recipe("minecraft:furnace_log", FURNACE_LOG_TAG_INPUT)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["result_item"] == "minecraft:coal:1"
    assert record["fields"]["ingredients"] == ["minecraft:logs_that_burn"]


# Real bedrock-samples cake.json: "result" is a list of items (the cake plus
# the 3 emptied buckets returned as a byproduct) — the main product is first.
CAKE_MULTI_ITEM_RESULT = {
    "minecraft:recipe_shaped": {
        "description": {"identifier": "minecraft:cake"},
        "pattern": ["AAA", "BEB", "CCC"],
        "key": {
            "A": {"item": "minecraft:bucket", "data": 1},
            "B": {"item": "minecraft:sugar"},
            "C": {"item": "minecraft:wheat"},
            "E": {"tag": "minecraft:egg"},
        },
        "result": [
            {"item": "minecraft:cake"},
            {"item": "minecraft:bucket", "count": 3, "data": 0},
        ],
    }
}


def test_adapt_shaped_recipe_multi_item_result_uses_primary_product():
    records = adapt_recipe("minecraft:cake", CAKE_MULTI_ITEM_RESULT)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["result_item"] == "minecraft:cake"
    assert sorted(record["fields"]["ingredients"]) == [
        "minecraft:bucket", "minecraft:egg", "minecraft:sugar", "minecraft:wheat",
    ]


# Real bedrock-samples brew_awkward_blaze_powder.json.
BREWING_MIX = {
    "minecraft:recipe_brewing_mix": {
        "description": {"identifier": "minecraft:brew_awkward_blaze_powder"},
        "input": "minecraft:potion_type:awkward",
        "reagent": "minecraft:blaze_powder",
        "output": "minecraft:potion_type:strength",
    }
}


def test_adapt_recipe_brewing_mix():
    records = adapt_recipe("minecraft:brew_awkward_blaze_powder", BREWING_MIX)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "brewing_mix"
    assert record["fields"]["result_item"] == "minecraft:potion_type:strength"
    assert record["fields"]["ingredients"] == ["minecraft:blaze_powder", "minecraft:potion_type:awkward"]


# Real bedrock-samples brew_potion_sulphur.json.
BREWING_CONTAINER = {
    "minecraft:recipe_brewing_container": {
        "description": {"identifier": "minecraft:brew_potion_sulphur"},
        "input": "minecraft:potion",
        "reagent": "minecraft:gunpowder",
        "output": "minecraft:splash_potion",
    }
}


def test_adapt_recipe_brewing_container():
    records = adapt_recipe("minecraft:brew_potion_sulphur", BREWING_CONTAINER)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "brewing_container"
    assert record["fields"]["result_item"] == "minecraft:splash_potion"
    assert record["fields"]["ingredients"] == ["minecraft:gunpowder", "minecraft:potion"]


# Real bedrock-samples smithing_netherite_sword.json.
SMITHING_TRANSFORM = {
    "minecraft:recipe_smithing_transform": {
        "description": {"identifier": "minecraft:smithing_netherite_sword"},
        "template": "minecraft:netherite_upgrade_smithing_template",
        "base": "minecraft:diamond_sword",
        "addition": "minecraft:netherite_ingot",
        "result": "minecraft:netherite_sword",
    }
}


def test_adapt_recipe_smithing_transform():
    records = adapt_recipe("minecraft:smithing_netherite_sword", SMITHING_TRANSFORM)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "smithing_transform"
    assert record["fields"]["result_item"] == "minecraft:netherite_sword"
    assert record["fields"]["ingredients"] == [
        "minecraft:diamond_sword", "minecraft:netherite_ingot", "minecraft:netherite_upgrade_smithing_template",
    ]


# Real bedrock-samples smithing_armor_trim.json: tag-based ingredients and
# NO result field at all -- trimming re-decorates the base item in place
# rather than producing a distinct output item.
SMITHING_TRIM = {
    "minecraft:recipe_smithing_trim": {
        "description": {"identifier": "minecraft:smithing_armor_trim"},
        "template": {"tag": "minecraft:trim_templates"},
        "base": {"tag": "minecraft:trimmable_armors"},
        "addition": {"tag": "minecraft:trim_materials"},
    }
}


def test_adapt_recipe_smithing_trim_uses_base_as_result():
    records = adapt_recipe("minecraft:smithing_armor_trim", SMITHING_TRIM)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "smithing_trim"
    assert record["fields"]["result_item"] == "minecraft:trimmable_armors"
    assert record["fields"]["ingredients"] == [
        "minecraft:trim_materials", "minecraft:trim_templates", "minecraft:trimmable_armors",
    ]
