from src.data_report.schema import adapt_recipe, make_fact_record


SHAPED_PISTON_RECIPE = {
    "type": "minecraft:crafting_shaped",
    "pattern": ["###", "#X#", "#Y#"],
    "key": {
        "#": {"item": "minecraft:oak_planks"},
        "X": {"item": "minecraft:cobblestone"},
        "Y": {"item": "minecraft:iron_ingot"},
    },
    "result": {"id": "minecraft:piston", "count": 1},
}


def test_make_fact_record_shape():
    record = make_fact_record("recipe", "minecraft:piston", {"a": 1})
    assert record == {"category": "recipe", "subject_id": "minecraft:piston", "fields": {"a": 1}}


def test_adapt_shaped_recipe():
    records = adapt_recipe("minecraft:piston", SHAPED_PISTON_RECIPE)
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "recipe"
    assert record["subject_id"] == "minecraft:piston"
    assert record["fields"]["recipe_type"] == "crafting_shaped"
    assert record["fields"]["result_item"] == "minecraft:piston"
    assert record["fields"]["result_count"] == 1
    assert set(record["fields"]["ingredients"]) == {
        "minecraft:oak_planks", "minecraft:cobblestone", "minecraft:iron_ingot",
    }


def test_adapt_recipe_unknown_type_returns_empty():
    records = adapt_recipe("minecraft:x", {"type": "minecraft:crafting_special_firework_star"})
    assert records == []


SHAPED_DICT_TAG_RECIPE = {
    "type": "minecraft:crafting_shaped",
    "pattern": ["##", "##"],
    "key": {
        "#": {"tag": "minecraft:planks"},
    },
    "result": {"id": "minecraft:crafting_table", "count": 1},
}


def test_adapt_shaped_recipe_dict_tag_ingredient():
    records = adapt_recipe("minecraft:crafting_table", SHAPED_DICT_TAG_RECIPE)
    assert len(records) == 1
    assert set(records[0]["fields"]["ingredients"]) == {"#minecraft:planks"}


SHAPED_BARE_ITEM_RECIPE = {
    "type": "minecraft:crafting_shaped",
    "pattern": ["#", "#"],
    "key": {
        "#": "minecraft:stick",
    },
    "result": {"id": "minecraft:ladder", "count": 3},
}


def test_adapt_shaped_recipe_bare_string_item_ingredient():
    records = adapt_recipe("minecraft:ladder", SHAPED_BARE_ITEM_RECIPE)
    assert len(records) == 1
    assert set(records[0]["fields"]["ingredients"]) == {"minecraft:stick"}


SHAPED_BARE_TAG_RECIPE = {
    "type": "minecraft:crafting_shaped",
    "pattern": ["##", "##"],
    "key": {
        "#": "#minecraft:planks",
    },
    "result": {"id": "minecraft:crafting_table", "count": 1},
}


def test_adapt_shaped_recipe_bare_string_tag_ingredient():
    records = adapt_recipe("minecraft:crafting_table", SHAPED_BARE_TAG_RECIPE)
    assert len(records) == 1
    assert set(records[0]["fields"]["ingredients"]) == {"#minecraft:planks"}


SHAPED_MIXED_RECIPE = {
    "type": "minecraft:crafting_shaped",
    "pattern": ["ABC"],
    "key": {
        "A": {"item": "minecraft:iron_ingot"},
        "B": {"tag": "minecraft:planks"},
        "C": "minecraft:stick",
    },
    "result": {"id": "minecraft:x", "count": 1},
}


def test_adapt_shaped_recipe_mixed_ingredient_shapes():
    records = adapt_recipe("minecraft:x", SHAPED_MIXED_RECIPE)
    assert len(records) == 1
    assert set(records[0]["fields"]["ingredients"]) == {
        "minecraft:iron_ingot", "#minecraft:planks", "minecraft:stick",
    }


# Real 26.2 data report: deepslate_tile_stairs.json.
STONECUTTING_RECIPE = {
    "type": "minecraft:stonecutting",
    "ingredient": "minecraft:deepslate_bricks",
    "result": {"id": "minecraft:deepslate_tile_stairs"},
}


def test_adapt_stonecutting_recipe():
    records = adapt_recipe("minecraft:deepslate_tile_stairs", STONECUTTING_RECIPE)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "stonecutting"
    assert record["fields"]["result_item"] == "minecraft:deepslate_tile_stairs"
    assert record["fields"]["ingredients"] == ["minecraft:deepslate_bricks"]


# Real 26.2 data report: purple_bundle.json.
TRANSMUTE_RECIPE = {
    "type": "minecraft:crafting_transmute",
    "category": "equipment",
    "group": "bundle_dye",
    "input": "#minecraft:bundles",
    "material": "minecraft:purple_dye",
    "result": {"id": "minecraft:purple_bundle"},
}


def test_adapt_crafting_transmute_recipe():
    records = adapt_recipe("minecraft:purple_bundle", TRANSMUTE_RECIPE)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "crafting_transmute"
    assert record["fields"]["result_item"] == "minecraft:purple_bundle"
    assert set(record["fields"]["ingredients"]) == {"#minecraft:bundles", "minecraft:purple_dye"}


# Real 26.2 data report: netherite_axe smithing recipe.
SMITHING_TRANSFORM_RECIPE = {
    "type": "minecraft:smithing_transform",
    "addition": "#minecraft:netherite_tool_materials",
    "base": "minecraft:diamond_axe",
    "result": {"id": "minecraft:netherite_axe"},
    "template": "minecraft:netherite_upgrade_smithing_template",
}


def test_adapt_smithing_transform_recipe():
    records = adapt_recipe("minecraft:netherite_axe", SMITHING_TRANSFORM_RECIPE)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "smithing_transform"
    assert record["fields"]["result_item"] == "minecraft:netherite_axe"
    assert set(record["fields"]["ingredients"]) == {
        "#minecraft:netherite_tool_materials", "minecraft:diamond_axe",
        "minecraft:netherite_upgrade_smithing_template",
    }


# Real 26.2 data report: rib armor trim smithing recipe -- no "result" field
# at all, since trimming re-decorates the base item in place.
SMITHING_TRIM_RECIPE = {
    "type": "minecraft:smithing_trim",
    "addition": "#minecraft:trim_materials",
    "base": "#minecraft:trimmable_armor",
    "pattern": "minecraft:rib",
    "template": "minecraft:rib_armor_trim_smithing_template",
}


def test_adapt_smithing_trim_recipe_uses_base_as_result():
    records = adapt_recipe("minecraft:rib_armor_trim", SMITHING_TRIM_RECIPE)
    assert len(records) == 1
    record = records[0]
    assert record["fields"]["recipe_type"] == "smithing_trim"
    assert record["fields"]["result_item"] == "#minecraft:trimmable_armor"
    assert set(record["fields"]["ingredients"]) == {
        "#minecraft:trim_materials", "#minecraft:trimmable_armor",
        "minecraft:rib_armor_trim_smithing_template",
    }
