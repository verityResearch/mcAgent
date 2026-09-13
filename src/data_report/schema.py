from typing import Any, Dict, List, Optional


def make_fact_record(category: str, subject_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    return {"category": category, "subject_id": subject_id, "fields": fields}


_SHAPED_TYPES = {"minecraft:crafting_shaped"}
_SHAPELESS_TYPES = {"minecraft:crafting_shapeless"}
_SINGLE_INGREDIENT_TYPES = {
    "minecraft:smelting", "minecraft:blasting", "minecraft:smoking", "minecraft:campfire_cooking",
    "minecraft:stonecutting",
}
_TEMPLATE_TYPES = {"minecraft:smithing_transform", "minecraft:smithing_trim"}
_TRANSMUTE_TYPES = {"minecraft:crafting_transmute"}


def _ingredient_items(ingredient: Any) -> List[str]:
    if isinstance(ingredient, str):
        return [ingredient]
    if isinstance(ingredient, dict):
        if "item" in ingredient:
            return [ingredient["item"]]
        if "tag" in ingredient:
            return [f"#{ingredient['tag']}"]
        return []
    if isinstance(ingredient, list):
        items = []
        for entry in ingredient:
            items.extend(_ingredient_items(entry))
        return items
    return []


def adapt_recipe(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    recipe_type = raw_json.get("type")
    result = raw_json.get("result") or {}
    result_item = result.get("id") if isinstance(result, dict) else None
    result_count = result.get("count", 1) if isinstance(result, dict) else 1

    # ingredient_groups: one entry per AND-slot (a crafting-grid position, or a
    # named slot like "addition"), each slot itself a list of OR-alternatives
    # (interchangeable items/tags that satisfy that one slot). This is the raw
    # recipe JSON's own structure (a shaped recipe's "key" maps each pattern
    # symbol to either a single item or a list of alternatives) -- preserved
    # here rather than flattened, so a downstream consumer can tell "need A
    # AND B" apart from "need A OR B" instead of losing that distinction.
    ingredient_groups: List[List[str]] = []
    if recipe_type in _SHAPED_TYPES:
        for ingredient in (raw_json.get("key") or {}).values():
            items = _ingredient_items(ingredient)
            if items:
                ingredient_groups.append(items)
    elif recipe_type in _SHAPELESS_TYPES:
        for ingredient in raw_json.get("ingredients") or []:
            items = _ingredient_items(ingredient)
            if items:
                ingredient_groups.append(items)
    elif recipe_type in _SINGLE_INGREDIENT_TYPES:
        items = _ingredient_items(raw_json.get("ingredient"))
        if items:
            ingredient_groups.append(items)
    elif recipe_type in _TEMPLATE_TYPES:
        for slot in ("template", "base", "addition"):
            items = _ingredient_items(raw_json.get(slot))
            if items:
                ingredient_groups.append(items)
        if not result_item:
            # smithing_trim has no result field: it re-decorates the base
            # item in place rather than producing a distinct output, so the
            # (possibly tag-referenced) base item is the closest true result.
            base_items = _ingredient_items(raw_json.get("base"))
            result_item = base_items[0] if base_items else None
    elif recipe_type in _TRANSMUTE_TYPES:
        for slot in ("input", "material"):
            items = _ingredient_items(raw_json.get(slot))
            if items:
                ingredient_groups.append(items)
    else:
        return []

    if not result_item or not ingredient_groups:
        return []

    fields = {
        "recipe_type": recipe_type.split(":", 1)[-1],
        "result_item": result_item,
        "result_count": result_count,
        "ingredients": sorted({item for group in ingredient_groups for item in group}),
        "ingredient_groups": [sorted(set(group)) for group in ingredient_groups],
    }
    return [make_fact_record("recipe", subject_id, fields)]


def _loot_entry_items(entry: Dict[str, Any]) -> List[str]:
    entry_type = entry.get("type")
    if entry_type == "minecraft:item":
        name = entry.get("name")
        return [name] if name else []
    if entry_type == "minecraft:tag":
        name = entry.get("name")
        return [f"#{name}"] if name else []
    if entry_type in ("minecraft:alternatives", "minecraft:group", "minecraft:sequence"):
        items: List[str] = []
        for child in entry.get("children") or []:
            items.extend(_loot_entry_items(child))
        return items
    return []


def adapt_loot_table(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    drops: List[str] = []
    for pool in raw_json.get("pools") or []:
        for entry in pool.get("entries") or []:
            drops.extend(_loot_entry_items(entry))
    if not drops:
        return []
    fields = {"drops": sorted(set(drops))}
    return [make_fact_record("loot_table", subject_id, fields)]


def adapt_advancement(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    criteria_ids = sorted((raw_json.get("criteria") or {}).keys())
    if not criteria_ids:
        return []
    fields = {"criteria_ids": criteria_ids}
    return [make_fact_record("advancement", subject_id, fields)]


def _tag_member_id(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        return value["id"]
    return None


def adapt_tag(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    members = [m for m in (_tag_member_id(v) for v in (raw_json.get("values") or [])) if m]
    if not members:
        return []
    fields = {"members": members}
    return [make_fact_record("tag", subject_id, fields)]


def adapt_worldgen_structure(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    structure_type = raw_json.get("type")
    if not structure_type:
        return []
    fields = {
        "structure_type": structure_type.split(":", 1)[-1],
        "biomes": raw_json.get("biomes"),
        "step": raw_json.get("step"),
    }
    return [make_fact_record("worldgen_structure", subject_id, fields)]


def adapt_enchantment(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adapt an enchantment definition into a mechanically-checkable fact.

    Extracts only the flat, unambiguous fields (max_level, weight, anvil_cost,
    equipment slots). Deliberately skips ``effects``/``*_cost`` curves: those are
    nested value expressions whose textual restatement can't be diffed
    reliably, so they'd weaken the oracle rather than strengthen it.
    """
    max_level = raw_json.get("max_level")
    if not isinstance(max_level, int):
        return []
    slots = raw_json.get("slots")
    fields = {
        "max_level": max_level,
        "weight": raw_json.get("weight"),
        "anvil_cost": raw_json.get("anvil_cost"),
        "slots": sorted(slots) if isinstance(slots, list) else slots,
    }
    return [make_fact_record("enchantment", subject_id, fields)]


def adapt_villager_trade(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adapt a villager trade into a mechanically-checkable fact.

    Trades are the ``wants`` (input) / ``gives`` (output) item+count pairs plus
    ``max_uses`` and ``xp`` -- all flat and ground-truthable. Ints are coerced
    from the report's floats (e.g. ``max_uses: 16.0``) so a natural-language
    restatement ("16 uses") diffs cleanly.
    """
    wants = raw_json.get("wants")
    gives = raw_json.get("gives")
    if not isinstance(wants, dict) or not isinstance(gives, dict):
        return []
    wants_id = wants.get("id")
    gives_id = gives.get("id")
    if not wants_id or not gives_id:
        return []

    def _count(obj):
        c = obj.get("count", 1)
        return int(c) if isinstance(c, (int, float)) else c

    def _int_or_none(v):
        return int(v) if isinstance(v, (int, float)) else v

    fields = {
        "wants_item": wants_id,
        "wants_count": _count(wants),
        "gives_item": gives_id,
        "gives_count": _count(gives),
        "max_uses": _int_or_none(raw_json.get("max_uses")),
        "xp": _int_or_none(raw_json.get("xp")),
    }
    return [make_fact_record("villager_trade", subject_id, fields)]


def adapt_jukebox_song(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adapt a jukebox song (music disc) into a checkable fact: its play length,
    comparator output signal, and the sound event it triggers.
    """
    length = raw_json.get("length_in_seconds")
    sound = raw_json.get("sound_event")
    if length is None or not sound:
        return []
    fields = {
        "length_in_seconds": int(length) if isinstance(length, float) and length.is_integer() else length,
        "comparator_output": raw_json.get("comparator_output"),
        "sound_event": sound,
    }
    return [make_fact_record("jukebox_song", subject_id, fields)]


def adapt_painting_variant(subject_id: str, raw_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Adapt a painting variant into a checkable fact: its block dimensions."""
    width = raw_json.get("width")
    height = raw_json.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return []
    return [make_fact_record("painting_variant", subject_id, {"width": width, "height": height})]
