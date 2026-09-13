from src.verification.registry_oracle import (
    MinecraftRegistry,
    check_registry,
    find_nonexistent_ids,
    make_layered_fact_check,
)


def _reg():
    return MinecraftRegistry({
        "minecraft:item": {"minecraft:carrot", "minecraft:potato", "minecraft:iron_ingot",
                           "minecraft:oak_planks", "minecraft:redstone"},
        "minecraft:block": {"minecraft:piston", "minecraft:cobblestone"},
        "minecraft:entity_type": {"minecraft:pig", "minecraft:cow"},
    })


def test_real_ids_pass():
    reg = _reg()
    assert find_nonexistent_ids("craft with minecraft:iron_ingot and minecraft:oak_planks", reg) == []
    assert reg.exists("minecraft:carrot")
    assert reg.exists("minecraft:carrot", registry="minecraft:item")


def test_fabricated_id_flagged():
    reg = _reg()
    bad = find_nonexistent_ids("feed the pig minecraft:corn and minecraft:kelp", reg)
    assert bad == ["minecraft:corn", "minecraft:kelp"]


def test_name_resolver_catches_fabricated_noun():
    reg = _reg()
    assert reg.resolve_name("carrot") == "minecraft:carrot"
    assert reg.resolve_name("iron ingot") == "minecraft:iron_ingot"  # underscore->space
    assert reg.resolve_name("corn") is None  # fabricated -> no real object


def test_check_registry_verdict_shape_admit():
    reg = _reg()
    cand = {"structured_claim": {"ingredients": ["minecraft:iron_ingot", "minecraft:oak_planks"]},
            "answer": "You craft it from iron and planks."}
    res = check_registry(cand, reg)
    assert res["success"] is True and res["failure_kind"] is None


def test_check_registry_verdict_shape_reject():
    reg = _reg()
    cand = {"structured_claim": {"ingredients": ["minecraft:corn"]},
            "answer": "Pistons need minecraft:stick and minecraft:corn."}
    res = check_registry(cand, reg)
    assert res["success"] is False
    assert res["failure_kind"] == "nonexistent_id"
    assert "minecraft:corn" in res["errors"][0]
    assert "minecraft:stick" in res["errors"][0]  # stick is not in this fixture's item set


def test_id_regex_does_not_eat_prose():
    reg = _reg()
    # trailing punctuation / sentence text must not be swallowed into the id
    assert find_nonexistent_ids("Use minecraft:carrot, then wait.", reg) == []


def test_layered_check_short_circuits_on_fabricated_id():
    reg = _reg()
    sentinel = {"called": False}

    def structured(candidate, fact):
        sentinel["called"] = True
        return {"success": True, "errors": [], "failure_kind": None}

    layered = make_layered_fact_check(reg, structured)
    # fabricated id -> Layer 0 rejects, Layer 1 never runs
    v = layered({"structured_claim": {"ingredients": ["minecraft:corn"]}}, {"fields": {}})
    assert v["success"] is False and v["failure_kind"] == "nonexistent_id"
    assert sentinel["called"] is False


def test_layered_check_delegates_when_ids_real():
    reg = _reg()

    def structured(candidate, fact):
        return {"success": False, "errors": ["field mismatch"], "failure_kind": "fact_mismatch"}

    layered = make_layered_fact_check(reg, structured)
    # all ids real -> Layer 1 runs and its verdict (here a mismatch) is returned
    v = layered({"structured_claim": {"ingredients": ["minecraft:iron_ingot"]}}, {"fields": {}})
    assert v["failure_kind"] == "fact_mismatch"
