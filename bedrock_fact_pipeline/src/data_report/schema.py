from typing import Any, Callable, Dict, List, Optional


def make_fact_record(category: str, subject_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"category": category, "subject_id": subject_id, "fields": fields}


_RECIPE_TYPE_KEYS = {
    "minecraft:recipe_shaped": "shaped",
    "minecraft:recipe_shapeless": "shapeless",
    "minecraft:recipe_furnace": "furnace",
    "minecraft:recipe_brewing_mix": "brewing_mix",
    "minecraft:recipe_brewing_container": "brewing_container",
    "minecraft:recipe_smithing_transform": "smithing_transform",
    "minecraft:recipe_smithing_trim": "smithing_trim",
}


def _ingredient_items(ingredient: Any) -> List[str]:
    if isinstance(ingredient, str):
        return [ingredient]
    if isinstance(ingredient, dict) and isinstance(ingredient.get("item"), str):
        return [ingredient["item"]]
    if isinstance(ingredient, dict) and isinstance(ingredient.get("tag"), str):
        return [ingredient["tag"]]
    if isinstance(ingredient, list):
        items: List[str] = []
        for entry in ingredient:
            items.extend(_ingredient_items(entry))
        return items
    return []


def adapt_recipe(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    recipe_key = next((k for k in _RECIPE_TYPE_KEYS if k in raw_json), None)
    if recipe_key is None:
        return []
    body = raw_json[recipe_key]

    if "output" in body:
        result_item = body["output"] if isinstance(body["output"], str) else None
    else:
        result = body.get("result")
        if isinstance(result, dict):
            result_item = result.get("item")
        elif isinstance(result, str):
            result_item = result
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            # Multi-item results (e.g. cake returns emptied buckets as a
            # byproduct) list the primary crafted product first.
            result_item = result[0].get("item")
        elif result is None and "base" in body:
            # recipe_smithing_trim has no result field: it re-decorates the
            # base item in place rather than producing a distinct output, so
            # the (possibly tag-based) base item is the closest true result.
            base_items = _ingredient_items(body["base"])
            result_item = base_items[0] if base_items else None
        else:
            result_item = None

    ingredients: List[str] = []
    if "key" in body:
        for ingredient in (body.get("key") or {}).values():
            ingredients.extend(_ingredient_items(ingredient))
    elif "ingredients" in body:
        for ingredient in body.get("ingredients") or []:
            ingredients.extend(_ingredient_items(ingredient))
    elif "template" in body:
        # recipe_smithing_transform / recipe_smithing_trim: three named
        # ingredient slots instead of a pattern key or flat list.
        for slot in ("template", "base", "addition"):
            if slot in body:
                ingredients.extend(_ingredient_items(body[slot]))
    elif "reagent" in body:
        # recipe_brewing_mix / recipe_brewing_container: a base input plus
        # a reagent, both bare item strings.
        for slot in ("input", "reagent"):
            if slot in body:
                ingredients.extend(_ingredient_items(body[slot]))
    elif "input" in body:
        ingredients.extend(_ingredient_items(body.get("input")))

    if not result_item or not ingredients:
        return []

    fields = {
        "recipe_type": _RECIPE_TYPE_KEYS[recipe_key],
        "result_item": result_item,
        "ingredients": sorted(set(ingredients)),
    }
    return [make_fact_record("recipe", subject_id, fields)]


def _loot_table_drops(
    raw_json: Dict[str, Any],
    resolve: Optional[Callable[[str], Optional[Dict[str, Any]]]],
    visited: set,
) -> List[str]:
    drops: List[str] = []
    for pool in raw_json.get("pools") or []:
        for entry in pool.get("entries") or []:
            name = entry.get("name")
            entry_type = entry.get("type")
            if entry_type == "item" and name:
                drops.append(name)
            elif entry_type == "loot_table" and name and resolve and name not in visited:
                referenced = resolve(name)
                if isinstance(referenced, dict):
                    visited.add(name)
                    drops.extend(_loot_table_drops(referenced, resolve, visited))
    return drops


def adapt_loot_table(
    subject_id: str,
    raw_json: Dict[str, Any],
    resolve: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    drops = _loot_table_drops(raw_json, resolve, visited=set())
    if not drops:
        return []
    fields = {"drops": sorted(set(drops))}
    return [make_fact_record("loot_table", subject_id, fields)]


def adapt_entity(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    description = (raw_json.get("minecraft:entity") or {}).get("description") or {}
    if "identifier" not in description:
        return []
    fields = {
        "spawn_category": description.get("spawn_category"),
        "is_spawnable": bool(description.get("is_spawnable", False)),
        "is_summonable": bool(description.get("is_summonable", False)),
    }
    return [make_fact_record("entity", subject_id, fields)]


def _biome_filter_tags(filt: Any) -> List[str]:
    """Recursively collect positively-required has_biome_tag values.

    Real bedrock-samples biome_filter is a small boolean-expression shape:
    a bare leaf dict, a list of leaves/composites, or a composite
    (all_of/any_of/none_of) nesting more filters. Only tags that are true
    positive requirements somewhere in the expression are collected —
    none_of subtrees and negated ("!="/"not") leaves are excluded so a
    biome the subject explicitly does NOT spawn in is never reported as
    one it does.
    """
    if isinstance(filt, list):
        tags: List[str] = []
        for entry in filt:
            tags.extend(_biome_filter_tags(entry))
        return tags
    if not isinstance(filt, dict):
        return []
    if "all_of" in filt:
        return _biome_filter_tags(filt["all_of"])
    if "any_of" in filt:
        return _biome_filter_tags(filt["any_of"])
    if "none_of" in filt:
        return []
    if (filt.get("test") == "has_biome_tag" and filt.get("operator") in (None, "==")
            and isinstance(filt.get("value"), str)):
        return [filt["value"]]
    return []


def adapt_spawn_rule(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = raw_json.get("minecraft:spawn_rules") or {}
    biome_tags: List[str] = []
    for condition in body.get("conditions") or []:
        biome_filter = condition.get("minecraft:biome_filter")
        if biome_filter is not None:
            biome_tags.extend(_biome_filter_tags(biome_filter))
    if not biome_tags:
        return []
    fields = {"biome_tags": sorted(set(biome_tags))}
    return [make_fact_record("spawn_rule", subject_id, fields)]


def _trade_items(entries: Any) -> List[str]:
    items: List[str] = []
    for entry in entries or []:
        if isinstance(entry, dict) and isinstance(entry.get("item"), str):
            items.append(entry["item"])
    return items


def adapt_trading(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    profession = subject_id.split(":", 1)[-1]
    if profession.endswith("_trades"):
        profession = profession[: -len("_trades")]
    records: List[Dict[str, Any]] = []
    for tier_index, tier in enumerate(raw_json.get("tiers") or []):
        for trade_index, trade in enumerate(tier.get("trades") or []):
            wants = _trade_items(trade.get("wants"))
            gives = _trade_items(trade.get("gives"))
            if not wants or not gives:
                continue
            trade_subject_id = f"minecraft:{profession}/tier{tier_index}/trade{trade_index}"
            fields = {"wants": sorted(set(wants)), "gives": sorted(set(gives))}
            records.append(make_fact_record("trading", trade_subject_id, fields))
    return records


def adapt_biome(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    components = (raw_json.get("minecraft:biome") or {}).get("components") or {}
    fields: Dict[str, Any] = {}
    climate = components.get("minecraft:climate")
    if isinstance(climate, dict):
        if "temperature" in climate:
            fields["temperature"] = climate["temperature"]
        if "downfall" in climate:
            fields["downfall"] = climate["downfall"]
    tags_component = components.get("minecraft:tags")
    if isinstance(tags_component, dict) and isinstance(tags_component.get("tags"), list):
        fields["tags"] = tags_component["tags"]
    if not fields:
        return []
    return [make_fact_record("biome", subject_id, fields)]
