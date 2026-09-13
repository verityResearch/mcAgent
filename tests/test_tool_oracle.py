import os
import sqlite3
import tempfile

import pytest

from tool_oracle.build_db import _load_all_tags, _make_resolver
from tool_oracle.lookup import OracleDB
from tool_oracle.gen_query_traces import generate, verify_trace


def _tags(tmp_path, files):
    root = os.path.join(str(tmp_path), "data", "minecraft", "tags")
    for rel, payload in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        import json
        open(p, "w").write(json.dumps(payload))
    return str(tmp_path)


def _write_tool_manifest(path):
    """Copy the canonical validated manifest into an isolated test path."""
    from tool_oracle.eval_tool_skill import _load_tool_manifest

    path.write_bytes(_load_tool_manifest().text.encode("utf-8"))
    return path


def test_recursive_tag_resolution(tmp_path):
    # wolf_food references #meat, which expands to concrete items
    dr = _tags(tmp_path, {
        "item/meat.json": {"values": ["minecraft:beef", "minecraft:chicken"]},
        "item/wolf_food.json": {"values": ["#minecraft:meat", "minecraft:salmon"]},
    })
    tags = _load_all_tags(dr)
    resolve = _make_resolver(tags)
    members = resolve("item", "minecraft:wolf_food")
    assert members == {"minecraft:beef", "minecraft:chicken", "minecraft:salmon"}
    assert not any(m.startswith("#") for m in members)


def test_resolution_is_cycle_safe(tmp_path):
    dr = _tags(tmp_path, {
        "item/a.json": {"values": ["#minecraft:b", "minecraft:x"]},
        "item/b.json": {"values": ["#minecraft:a", "minecraft:y"]},
    })
    resolve = _make_resolver(_load_all_tags(dr))
    assert resolve("item", "minecraft:a") == {"minecraft:x", "minecraft:y"}


def _fixture_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE breeding_food(animal TEXT, food_item TEXT)")
    con.execute("CREATE TABLE recipe(result_item TEXT, ingredient TEXT)")
    con.execute("CREATE TABLE enchantment(name TEXT, max_level INT, slots TEXT)")
    con.execute("CREATE TABLE loot(source TEXT, drop_item TEXT)")
    con.execute("CREATE TABLE tag(tag TEXT, member TEXT)")
    con.execute("CREATE TABLE villager_trade(trade TEXT, wants_item TEXT, wants_count INT, "
                "gives_item TEXT, gives_count INT)")
    con.execute("CREATE TABLE jukebox_song(song TEXT, length_seconds INT)")
    con.execute("CREATE TABLE painting(painting TEXT, width INT, height INT)")
    con.execute("CREATE TABLE item_use(item TEXT, nutrition INT, saturation REAL, "
                "max_damage INT, attack_damage REAL, equip_slot TEXT)")
    con.execute("CREATE TABLE ore_depth(block TEXT, placement TEXT, dist_type TEXT, "
                "min_y INT, max_y INT, count INT)")
    con.executemany("INSERT INTO breeding_food VALUES(?,?)",
                    [("pig", "minecraft:carrot"), ("pig", "minecraft:potato"), ("pig", "minecraft:beetroot")])
    con.executemany("INSERT INTO recipe VALUES(?,?)",
                    [("minecraft:piston", "minecraft:oak_planks"), ("minecraft:piston", "minecraft:iron_ingot")])
    con.execute("INSERT INTO enchantment VALUES('minecraft:sharpness', 5, '[]')")
    con.executemany("INSERT INTO loot VALUES(?,?)",
                    [("minecraft:blocks/oak_log", "minecraft:oak_log")])
    con.executemany("INSERT INTO tag VALUES(?,?)",
                    [("minecraft:planks", "minecraft:oak_planks"), ("minecraft:planks", "minecraft:birch_planks")])
    con.execute("INSERT INTO villager_trade VALUES('minecraft:butcher/1/x', 'minecraft:rabbit', 4, 'minecraft:emerald', 1)")
    con.execute("INSERT INTO jukebox_song VALUES('minecraft:13', 178)")
    con.execute("INSERT INTO painting VALUES('minecraft:skeleton', 4, 3)")
    con.execute("INSERT INTO item_use VALUES('minecraft:apple', 4, 2.4, NULL, NULL, NULL)")
    con.execute("INSERT INTO item_use VALUES('minecraft:carrot_on_a_stick', NULL, NULL, 25, NULL, NULL)")
    con.commit()
    con.close()
    return path


def test_lookup_and_verify():
    path = _fixture_db()
    db = OracleDB(path)
    try:
        assert db.breeding_foods("pig") == ["minecraft:beetroot", "minecraft:carrot", "minecraft:potato"]
        # fabrication guard
        assert db.is_breeding_food("carrot", "pig") is True
        assert db.is_breeding_food("corn", "pig") is False        # fabricated item
        assert db.is_breeding_food("wheat", "pig") is False        # real item, wrong animal
        assert db.enchantment_max_level("minecraft:sharpness") == 5
    finally:
        db.close()
        os.remove(path)


def test_generated_traces_all_self_verify():
    import copy
    import random
    path = _fixture_db()
    db = OracleDB(path)
    try:
        traces = list(generate(db, random.Random(0)))
        assert traces, "expected some traces"
        assert all(verify_trace(t, db) for t in traces)
        # multi-turn shape: user, assistant(tool_call), user(result), assistant(answer)
        assert [m["role"] for m in traces[0]["messages"]] == ["user", "assistant", "user", "assistant"]
        assert "<tool_call>" in traces[0]["messages"][1]["content"]
        # tamper the RESULT so it no longer matches the DB -> must fail verification
        tampered = copy.deepcopy(traces[0])
        tampered["messages"][2]["content"] = tampered["messages"][2]["content"].replace(
            "<result>", "<result>minecraft:corn, ")
        assert verify_trace(tampered, db) is False
    finally:
        db.close()
        os.remove(path)


def test_multihop_traces_self_verify():
    """Two-hop recipe->recipe traces chain correctly and self-verify; a tampered
    second-hop result must fail verification."""
    import copy
    import random

    from tool_oracle.gen_multihop_traces import generate as gen_mh
    from tool_oracle.gen_multihop_traces import verify_multihop

    path = _fixture_db()
    db = OracleDB(path)
    try:
        # fixture: piston <- oak_planks (both craftable). Add oak_planks recipe.
        db.con.execute("INSERT INTO recipe VALUES('minecraft:oak_planks', 'minecraft:oak_log')")
        db.con.commit()
        traces = list(gen_mh(db, random.Random(0), limit=10))
        assert traces, "expected at least one 2-hop chain (piston <- oak_planks)"
        assert all(verify_multihop(t, db) for t in traces)
        assert [m["role"] for m in traces[0]["messages"]] == \
            ["user", "assistant", "user", "assistant", "user", "assistant"]
        bad = copy.deepcopy(traces[0])
        bad["messages"][4]["content"] = "<result>minecraft:bogus</result>"
        assert verify_multihop(bad, db) is False
    finally:
        db.close()
        os.remove(path)


def test_large_tag_count_and_sample_shape():
    """Tags with >12 members get a count+sample trace (not full enumeration);
    verify_trace accepts count+partial-sample and rejects a wrong count."""
    import copy
    import random

    path = _fixture_db()
    db = OracleDB(path)
    try:
        # give the fixture a tag with 13 members (over the max_tag_members=12 gate)
        db.con.executemany(
            "INSERT INTO tag VALUES('minecraft:big_tag', ?)",
            [(f"minecraft:item_{i}",) for i in range(13)])
        db.con.commit()
        traces = list(generate(db, random.Random(0), max_tag_members=12, large_tag_cap=40))
        big = [t for t in traces if "big_tag" in t["messages"][1]["content"]]
        assert big, "expected a large-tag trace for the 13-member tag"
        assert all(verify_trace(t, db) for t in big)
        assert "13 items" in big[0]["messages"][3]["content"]
        # a wrong count must fail verification
        bad = copy.deepcopy(big[0])
        bad["messages"][3]["content"] = bad["messages"][3]["content"].replace("13 items", "99 items")
        assert verify_trace(bad, db) is False
    finally:
        db.close()
        os.remove(path)


def test_cross_table_multihop_self_verifies():
    """breeding_food -> recipe chains (feed animal X, craft that food) chain
    correctly and self-verify via the generalized verify_multihop."""
    import random

    from tool_oracle.gen_multihop_traces import generate_cross, verify_multihop

    path = _fixture_db()
    db = OracleDB(path)
    try:
        # give the pig a craftable breeding food: golden_apple <- gold_ingot+apple
        db.con.execute("INSERT INTO breeding_food VALUES('pig', 'minecraft:golden_apple')")
        db.con.executemany("INSERT INTO recipe VALUES('minecraft:golden_apple', ?)",
                           [("minecraft:gold_ingot",), ("minecraft:apple",)])
        db.con.commit()
        traces = list(generate_cross(db, random.Random(0)))
        assert traces, "expected a breeding->recipe chain (pig <- golden_apple)"
        assert all(verify_multihop(t, db) for t in traces)
        # first hop must be breeding_food, second must be recipe
        t = traces[0]
        assert "lookup(breeding_food" in t["messages"][1]["content"]
        assert "lookup(recipe" in t["messages"][3]["content"]
    finally:
        db.close()
        os.remove(path)


def test_usage_question_declines_without_calling_a_tool():
    """Usage/function questions ("what is X used for?") on items with NO
    item_use data have no matching table -- the correct behavior is a no-tool
    decline, distinct from the fake-subject decline (which DOES call a tool
    and gets empty rows). Items that DO have item_use data (carrot_on_a_stick,
    in this fixture) must be SKIPPED here, not declined -- generate() handles
    those with a real query+answer instead."""
    import random

    from tool_oracle.gen_query_traces import generate_usage_declines, verify_usage_decline

    path = _fixture_db()
    db = OracleDB(path)
    try:
        traces = list(generate_usage_declines(db, random.Random(0)))
        assert traces, "expected usage-decline traces"
        assert all(verify_usage_decline(t) for t in traces)
        for t in traces:
            assert len(t["messages"]) == 2
            assert "<tool_call>" not in t["messages"][1]["content"]
        # carrot_on_a_stick HAS item_use data in the fixture -- it must NOT appear
        # among the declines (it should be answered with real facts instead)
        questions = " ".join(t["messages"][0]["content"] for t in traces)
        assert "carrot on a stick" not in questions
        assert "compass" in questions  # compass has no item_use row -- still declines
    finally:
        db.close()
        os.remove(path)
    # a trace that fabricates a "use" instead of declining must fail verification
    bad = {"messages": [
        {"role": "user", "content": "What is a carrot on a stick used for?"},
        {"role": "assistant", "content": "It is used to serve carrot soup."},
    ]}
    assert verify_usage_decline(bad) is False
    # a trace that calls a tool (even correctly) must also fail -- no lookup helps here
    tooled = {"messages": [
        {"role": "user", "content": "What is a compass used for?"},
        {"role": "assistant",
         "content": "<tool_call>lookup(recipe, result_item='minecraft:compass')</tool_call>"},
    ]}
    assert verify_usage_decline(tooled) is False


def test_item_use_traces_self_verify():
    """Items WITH item_use data get a real query+answer trace built from the
    non-None component fields, and it self-verifies; a fabricated answer must
    fail verification."""
    import copy
    import random

    path = _fixture_db()
    db = OracleDB(path)
    try:
        traces = list(generate(db, random.Random(0)))
        iu = [t for t in traces if "lookup(item_use" in t["messages"][1]["content"]]
        assert iu, "expected item_use traces (apple, carrot_on_a_stick in the fixture)"
        assert all(verify_trace(t, db) for t in iu)
        apple = [t for t in iu if "apple" in t["messages"][1]["content"]][0]
        assert "4" in apple["messages"][3]["content"]      # nutrition
        assert "2.4" in apple["messages"][3]["content"]    # saturation
        stick = [t for t in iu if "carrot_on_a_stick" in t["messages"][1]["content"]][0]
        assert "25" in stick["messages"][3]["content"]     # max_damage/durability
        bad = copy.deepcopy(apple)
        bad["messages"][3]["content"] = "It is used to serve carrot soup."
        assert verify_trace(bad, db) is False
    finally:
        db.close()
        os.remove(path)


def test_lookups_are_case_and_suffix_tolerant():
    """A minor key slip (capitalized subject, or an '_enchant' suffix from
    phrasing) should still resolve, but fake subjects must still return empty."""
    path = _fixture_db()
    db = OracleDB(path)
    try:
        assert db.enchantment_max_level("minecraft:sharpness") == 5
        assert db.enchantment_max_level("minecraft:Sharpness") == 5        # capitalized
        assert db.enchantment_max_level("Sharpness") == 5                  # no namespace
        assert db.enchantment_max_level("minecraft:sharpness_enchant") == 5  # suffix slip
        assert db.enchantment_max_level("minecraft:excalibur") is None     # fake still declines
        assert db.breeding_foods("Pig") == db.breeding_foods("pig")        # case-insensitive
        assert db.breeding_foods("dragon") == []                           # fake still empty
    finally:
        db.close()
        os.remove(path)


def test_decline_traces_for_unknown_subjects():
    """Fake subjects yield query -> (no rows) -> decline traces that self-verify,
    and a decline trace that fabricates a fact must FAIL verification."""
    import copy
    import random

    path = _fixture_db()
    db = OracleDB(path)
    try:
        traces = list(generate(db, random.Random(0)))
        declines = [t for t in traces if "(no rows)" in t["messages"][2]["content"]]
        assert declines, "expected at least one decline trace for a fake subject"
        assert all(verify_trace(t, db) for t in declines)
        # a decline that instead ASSERTS a fabricated fact must not verify
        bad = copy.deepcopy(declines[0])
        bad["messages"][3]["content"] = "To craft that you need diamonds and sticks."
        assert verify_trace(bad, db) is False
    finally:
        db.close()
        os.remove(path)


def test_probe_loading_and_unknown_scoring(tmp_path):
    import json as _json

    from tool_oracle.eval_tool_skill import _score, _score_unknown, load_probe_items

    path = _fixture_db()
    db = OracleDB(path)
    try:
        probe = tmp_path / "probe.jsonl"
        probe.write_text(
            "// comment line is skipped\n"
            + _json.dumps({"question": "how do i make a piston?",
                           "table": "recipe", "key": "minecraft:piston", "kind": "known"}) + "\n"
            + _json.dumps({"question": "recipe for a dragon sword?",
                           "table": "recipe", "key": "minecraft:dragon_sword", "kind": "unknown"}) + "\n")
        items = load_probe_items(db, str(probe))
        assert len(items) == 2
        known, unknown = items[0], items[1]
        # known item: ground truth derived from the live DB
        assert known["kind"] == "known"
        assert set(known["expected"]) == {"minecraft:oak_planks", "minecraft:iron_ingot"}
        assert _score(known, "to craft piston you need oak planks and an iron ingot") is True
        # The precision-first scorer requires the subject when recipe or loot
        # value sets can be subsets of another item's complete answer.
        assert _score(known, "you need oak planks and an iron ingot") is False
        assert _score(known, "you need oak planks") is False  # missing an ingredient
        # unknown item: no DB row -> empty expected, decline scores as correct
        assert unknown["kind"] == "unknown" and unknown["expected"] == []
        assert _score_unknown("I couldn't find a recipe for that item.") is True
        assert _score_unknown("To craft a dragon sword you need diamonds and fire.") is False
    finally:
        db.close()
        os.remove(path)


def test_score_subject_identity_regression():
    """Reject a scalar-value answer about the wrong subject."""
    from tool_oracle.eval_tool_skill import _score

    mending = {
        "category": "enchantment",
        "table": "enchantment",
        "key": "minecraft:mending",
        "expected": ["max_level=1"],
    }
    channeling = {
        "category": "enchantment",
        "table": "enchantment",
        "key": "minecraft:channeling",
        "expected": ["max_level=1"],
    }
    mending_answer = "Mending has a maximum level of 1."
    assert _score(mending, mending_answer) is True
    assert _score(channeling, mending_answer) is False
    assert _score(mending, "Unmending has a maximum level of 1.") is False
    assert _score(mending, "Mending has a maximum level of 15.") is False


def test_score_cross_subject_collision_regressions():
    """Reject recipe, loot, and trade facts about a different subject."""
    from tool_oracle.eval_tool_skill import _score

    shears = {
        "category": "recipe",
        "table": "recipe",
        "key": "minecraft:shears",
        "expected": ["minecraft:iron_ingot"],
    }
    piston_answer = (
        "To craft piston you need: cobblestone, redstone, planks, and iron ingot."
    )
    assert _score(shears, "To craft shears you need: iron ingot.") is True
    assert _score(shears, piston_answer) is False

    bamboo_sapling = {
        "category": "loot",
        "table": "loot",
        "key": "minecraft:blocks/bamboo_sapling",
        "expected": ["minecraft:bamboo"],
    }
    pressure_plate_answer = (
        "Bamboo pressure plate drops: bamboo pressure plate."
    )
    assert _score(bamboo_sapling, "Bamboo sapling drops: bamboo.") is True
    assert _score(bamboo_sapling, pressure_plate_answer) is False

    terracotta_trade = {
        "category": "villager_trade",
        "table": "villager_trade",
        "key": "minecraft:mason/4/emerald_white_terracotta",
        "expected": [
            "wants=1 minecraft:emerald",
            "gives=1 minecraft:white_terracotta",
        ],
    }
    glazed_terracotta_answer = (
        "The villager wants 1 emerald and gives 1 white glazed terracotta."
    )
    correct_trade_answer = (
        "The villager wants 1 emerald and gives 1 white terracotta."
    )
    assert _score(terracotta_trade, correct_trade_answer) is True
    assert _score(terracotta_trade, glazed_terracotta_answer) is False


def test_score_rejects_truncated_trade_reasoning():
    """Scattered count/item tokens are not a completed trade answer."""
    from tool_oracle.eval_tool_skill import _score

    trade = {
        "category": "villager_trade",
        "table": "villager_trade",
        "key": "minecraft:cleric/4/turtle_scute_emerald",
        "expected": [
            "wants=4 minecraft:turtle_scute",
            "gives=1 minecraft:emerald",
        ],
    }
    truncated = (
        "The Cleric/4/Turtle Scute Emerald trade is a level 4 cleric trade. "
        "I need to determine what the villager wants and gives."
    )
    assert _score(trade, truncated) is False


def test_final_answer_extraction_rejects_unclosed_reasoning():
    """Prompt restatement inside unfinished reasoning is not an answer."""
    from tool_oracle.eval_tool_skill import _extract_final_answer

    text = (
        "<think>Okay, the user asks what acacia wood drops. "
        "Acacia wood might drop"
    )
    assert _extract_final_answer(text) == ("", False)
    assert _extract_final_answer("</think>\nAcacia wood drops acacia wood.") == (
        "",
        False,
    )


def test_final_answer_extraction_keeps_only_post_reasoning_span():
    """Expected tokens in reasoning cannot satisfy the final-answer scorer."""
    from tool_oracle.eval_tool_skill import _extract_final_answer

    text = (
        "<think>The question says acacia wood and drop, but I must use the "
        "result.</think>\nAcacia wood drops: acacia wood.<|im_end|>"
    )
    assert _extract_final_answer(text) == (
        "Acacia wood drops: acacia wood.",
        True,
    )


def test_final_answer_extraction_accepts_direct_answer():
    """Fine-tuned direct-answer generations remain eligible."""
    from tool_oracle.eval_tool_skill import _extract_final_answer

    assert _extract_final_answer("Mending has a maximum level of 1.") == (
        "Mending has a maximum level of 1.",
        True,
    )
    assert _extract_final_answer("<think>done</think>\n") == ("", False)
    assert _extract_final_answer(
        "<tool_call>lookup(recipe, result_item='minecraft:shears')</tool_call>"
    ) == ("", False)


def test_token_completion_classification_fails_closed_at_cap():
    """An EOS at the cap is complete; cap exhaustion without EOS is not."""
    from tool_oracle.eval_tool_skill import _classify_token_completion

    assert _classify_token_completion([7, 8, 99], 3, 99) == (3, "eos", True)
    assert _classify_token_completion([7, 8, 9], 3, 99) == (
        3,
        "max_tokens",
        False,
    )
    assert _classify_token_completion([7, 8], 3, [98, 99]) == (
        2,
        "stopped",
        True,
    )


def test_question_seed_is_stable_and_order_independent():
    from tool_oracle.eval_tool_skill import _question_seed

    first = _question_seed(7, "How do I craft shears?")
    assert first == _question_seed(7, "How do I craft shears?")
    assert first != _question_seed(8, "How do I craft shears?")
    assert first != _question_seed(7, "What does an anvil drop?")
    assert _question_seed(None, "How do I craft shears?") is None


def test_default_tool_manifest_exposes_every_accepted_signature():
    """The prompt surface and executable parser cannot silently diverge."""
    import re

    from tool_oracle.eval_tool_skill import (
        _LOOKUP_KEY_ARGUMENTS,
        _load_tool_manifest,
    )

    signatures = dict(re.findall(
        r"^lookup\(([a-z_]+), ([a-z_]+)='KEY'\)$",
        _load_tool_manifest().text,
        re.MULTILINE,
    ))
    assert signatures == _LOOKUP_KEY_ARGUMENTS


def test_initial_conversation_requires_and_preserves_manifest():
    from tool_oracle.eval_tool_skill import (
        _initial_conversation,
        _load_tool_manifest,
    )

    manifest = _load_tool_manifest().text
    assert _initial_conversation("question", manifest) == [
        {"role": "system", "content": manifest},
        {"role": "user", "content": "question"},
    ]
    with pytest.raises(ValueError, match="tool manifest is empty"):
        _initial_conversation("question", "")


def test_manifest_expected_sha256_is_raw_byte_exact(tmp_path):
    import hashlib

    from tool_oracle.eval_tool_skill import _load_tool_manifest

    canonical = _load_tool_manifest().text
    manifest = tmp_path / "manifest-crlf.txt"
    raw = canonical.replace("\n", "\r\n").encode("utf-8")
    manifest.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()

    loaded = _load_tool_manifest(manifest, expected)
    assert loaded.sha256 == expected
    assert "\r\n" in loaded.text
    with pytest.raises(ValueError, match="does not match"):
        _load_tool_manifest(manifest, "0" * 64)


def test_manifest_contains_no_probe_or_heldout_keys():
    import json
    from pathlib import Path

    from tool_oracle.eval_tool_skill import _load_tool_manifest

    manifest = _load_tool_manifest().text
    probe = Path("tool_oracle/robustness_probe_full.jsonl")
    keys = set()
    for line in probe.read_text().splitlines():
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        keys.add(row.get("key"))
        keys.add(row.get("second_key"))
    assert not {key for key in keys if key and key in manifest}


def test_paired_results_require_item_seed_and_initial_prompt_parity():
    import copy

    from tool_oracle.eval_tool_skill import (
        _build_evidence_payload,
        _validate_paired_results,
    )

    base = [{
        "category": "recipe",
        "table": "recipe",
        "key": "minecraft:shears",
        "question": "How do I craft shears?",
        "expected": ["minecraft:iron_ingot"],
        "sampling_seed": 7,
        "raw_turns": [{"prompt_sha256": "a" * 64}],
    }]
    adapter = copy.deepcopy(base)
    pairing = _validate_paired_results(base, adapter)
    assert pairing["row_order_parity"] is True
    assert pairing["sampling_seed_parity"] is True
    assert pairing["initial_prompt_parity"] is True

    payload = _build_evidence_payload(
        {"backend": "cuda", "requested_arms": ["base", "adapter"]},
        base,
        adapter,
    )
    assert payload["metadata"]["arms"] == ["base", "adapter"]
    assert payload["metadata"]["comparison_mode"] == (
        "within_backend_paired_items"
    )
    assert payload["metadata"]["initial_prompt_parity"] is True
    assert set(payload) == {"metadata", "base", "adapter"}

    changed = copy.deepcopy(adapter)
    changed[0]["question"] = "different order"
    with pytest.raises(ValueError, match="row order"):
        _validate_paired_results(base, changed)
    changed = copy.deepcopy(adapter)
    changed[0]["sampling_seed"] = 8
    with pytest.raises(ValueError, match="sampling seed"):
        _validate_paired_results(base, changed)
    changed = copy.deepcopy(adapter)
    changed[0]["raw_turns"][0]["prompt_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial prompt"):
        _validate_paired_results(base, changed)


def test_requested_adapter_failure_never_builds_evidence_packet():
    from tool_oracle.eval_tool_skill import _build_evidence_payload

    with pytest.raises(ValueError, match="did not complete"):
        _build_evidence_payload(
            {"requested_arms": ["base", "adapter"]},
            [{"question": "q"}],
            None,
        )


def test_adapter_config_rejects_incompatible_declared_base(tmp_path):
    from tool_oracle.eval_tool_skill import _validate_adapter_config_identity

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"wrong/base",'
        '"peft_type":"LORA","task_type":"CAUSAL_LM"}\n'
    )
    with pytest.raises(ValueError, match="does not match"):
        _validate_adapter_config_identity("expected/base", adapter)

    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"expected/base",'
        '"peft_type":"IA3","task_type":"CAUSAL_LM"}\n'
    )
    with pytest.raises(ValueError, match="LORA adapter"):
        _validate_adapter_config_identity("expected/base", adapter)


def test_dependency_injected_pairing_runs_base_before_attach():
    from tool_oracle.eval_tool_skill import _evaluate_base_then_adapter

    events = []
    tokenizer = object()

    class Model:
        def __init__(self, label):
            self.label = label

        def eval(self):
            events.append(("eval", self.label))

    def run_arm(label, model, received_tokenizer):
        assert received_tokenizer is tokenizer
        events.append(("run", label, model.label))
        return [label]

    def attach(base, adapter_path):
        events.append(("attach", base.label, adapter_path))
        return Model("adapter-model")

    base_results, adapter_results = _evaluate_base_then_adapter(
        Model("base-model"),
        tokenizer,
        "adapter-path",
        run_arm,
        attach,
    )
    assert base_results == ["base"]
    assert adapter_results == ["adapter"]
    assert events == [
        ("eval", "base-model"),
        ("run", "base", "base-model"),
        ("attach", "base-model", "adapter-path"),
        ("eval", "adapter-model"),
        ("run", "adapter", "adapter-model"),
    ]


def test_evaluation_metadata_binds_inputs_and_controls(tmp_path):
    import argparse

    from tool_oracle.eval_tool_skill import _evaluation_metadata

    db = tmp_path / "minecraft.db"
    probe = tmp_path / "probe.jsonl"
    manifest = tmp_path / "tool-manifest.txt"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    db.write_bytes(b"db")
    probe.write_text('{"question":"q"}\n')
    _write_tool_manifest(manifest)
    (adapter / "weights.bin").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"model-id",'
        '"peft_type":"LORA","task_type":"CAUSAL_LM"}\n'
    )
    args = argparse.Namespace(
        model="model-id",
        adapter=str(adapter),
        db=str(db),
        probe=str(probe),
        train_jsonl=None,
        tool_manifest=str(manifest),
        max_tokens=1024,
        max_hops=3,
        seed=7,
    )
    items = [{"question": "q", "table": "recipe", "key": "minecraft:x"}]

    metadata = _evaluation_metadata("cuda", args, items, "revision")

    assert metadata["schema"] == "tool-eval-evidence-v6"
    assert metadata["model_revision"] == "revision"
    assert metadata["evaluation_rows"] == 1
    assert metadata["max_tokens"] == 1024
    assert metadata["seed"] == 7
    assert metadata["db_sha256"]
    assert metadata["probe_sha256"]
    assert metadata["adapter_sha256"]
    assert metadata["tool_manifest_sha256"]
    assert metadata["tool_manifest_version"] == "minecraft-lookup-manifest-v1"
    assert metadata["requested_arms"] == ["base", "adapter"]
    assert "arms" not in metadata
    assert "comparison_mode" not in metadata
    assert metadata["oracle_lookup_sha256"]
    assert metadata["trace_helpers_sha256"]


def test_evidence_output_cannot_alias_bound_inputs(tmp_path):
    import argparse

    from tool_oracle.eval_tool_skill import _validate_output_target

    db = tmp_path / "minecraft.db"
    probe = tmp_path / "probe.jsonl"
    manifest = tmp_path / "tool-manifest.txt"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    db.write_bytes(b"db")
    probe.write_text('{"question":"q"}\n')
    _write_tool_manifest(manifest)
    args = argparse.Namespace(
        model="remote/model",
        adapter=str(adapter),
        db=str(db),
        probe=str(probe),
        train_jsonl=None,
        tool_manifest=str(manifest),
        tool_manifest_sha256=None,
        out=str(tmp_path / "result.json"),
    )

    _validate_output_target(args)
    args.out = str(db)
    with pytest.raises(ValueError, match="bound database"):
        _validate_output_target(args)
    args.out = str(adapter / "result.json")
    with pytest.raises(ValueError, match="bound adapter"):
        _validate_output_target(args)
    args.out = str(manifest)
    with pytest.raises(ValueError, match="bound tool manifest"):
        _validate_output_target(args)


def test_evidence_output_requires_bound_remote_model_and_adapter(tmp_path):
    import argparse

    from tool_oracle.eval_tool_skill import _validate_evidence_bindings

    manifest = tmp_path / "tool-manifest.txt"
    _write_tool_manifest(manifest)
    args = argparse.Namespace(
        model="remote/model",
        revision=None,
        adapter=None,
        tool_manifest=str(manifest),
        out=str(tmp_path / "result.json"),
    )
    with pytest.raises(ValueError, match="exact --revision"):
        _validate_evidence_bindings(args)
    args.revision = "main"
    with pytest.raises(ValueError, match="exact --revision"):
        _validate_evidence_bindings(args)

    args.revision = "0123456789abcdef0123456789abcdef01234567"
    with pytest.raises(ValueError, match="predeclared --tool-manifest-sha256"):
        _validate_evidence_bindings(args)
    import hashlib

    args.tool_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        _validate_evidence_bindings(args)
    args.tool_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _validate_evidence_bindings(args)
    args.adapter = "remote/adapter"
    with pytest.raises(ValueError, match="local hashable bytes"):
        _validate_evidence_bindings(args)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"remote/model",'
        '"peft_type":"LORA","task_type":"CAUSAL_LM"}\n'
    )
    args.adapter = str(adapter)
    _validate_evidence_bindings(args)

    manifest.write_bytes(b"")
    args.tool_manifest_sha256 = hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValueError, match="tool manifest is empty"):
        _validate_evidence_bindings(args)


def test_evaluation_metadata_fails_closed_on_mid_run_drift(tmp_path):
    import argparse

    import pytest

    from tool_oracle.eval_tool_skill import (
        _evaluation_metadata,
        _finalize_evaluation_metadata,
    )

    db = tmp_path / "minecraft.db"
    manifest = tmp_path / "tool-manifest.txt"
    db.write_bytes(b"before")
    _write_tool_manifest(manifest)
    args = argparse.Namespace(
        model="model-id",
        adapter=None,
        db=str(db),
        probe=None,
        train_jsonl=None,
        tool_manifest=str(manifest),
        max_tokens=4096,
        max_hops=3,
        seed=0,
    )
    items = [{"question": "q"}]
    before = _evaluation_metadata("cuda", args, items)
    db.write_bytes(b"after")
    after = _evaluation_metadata("cuda", args, items)

    with pytest.raises(RuntimeError, match="inputs changed during run"):
        _finalize_evaluation_metadata(before, after, "revision")

    db.write_bytes(b"before")
    before = _evaluation_metadata("cuda", args, items)
    manifest.write_text(manifest.read_text() + "after\n")
    after = _evaluation_metadata("cuda", args, items)
    with pytest.raises(RuntimeError, match="tool_manifest_sha256"):
        _finalize_evaluation_metadata(before, after, "revision")


def test_evaluation_metadata_rejects_wrong_loaded_revision(tmp_path):
    import argparse

    import pytest

    from tool_oracle.eval_tool_skill import (
        _evaluation_metadata,
        _finalize_evaluation_metadata,
    )

    db = tmp_path / "minecraft.db"
    manifest = tmp_path / "tool-manifest.txt"
    db.write_bytes(b"db")
    _write_tool_manifest(manifest)
    args = argparse.Namespace(
        model="model-id",
        revision="expected-revision",
        local_files_only=True,
        adapter=None,
        db=str(db),
        probe=None,
        train_jsonl=None,
        tool_manifest=str(manifest),
        max_tokens=4096,
        max_hops=3,
        seed=0,
    )
    metadata = _evaluation_metadata("cuda", args, [{"question": "q"}])

    with pytest.raises(RuntimeError, match="does not match requested revision"):
        _finalize_evaluation_metadata(metadata, metadata, "wrong-revision")


def test_run_controls_fail_before_generation():
    import pytest

    from tool_oracle.eval_tool_skill import _validate_run_controls

    _validate_run_controls(1, 1)
    with pytest.raises(ValueError, match="max_tokens"):
        _validate_run_controls(0, 1)
    with pytest.raises(ValueError, match="max_hops"):
        _validate_run_controls(1, 0)


def test_json_evidence_write_is_atomic(tmp_path):
    import json

    import pytest

    from tool_oracle.eval_tool_skill import _write_json_atomic

    output = tmp_path / "result.json"
    output.write_text('{"old": true}\n')
    _write_json_atomic(output, {"new": [1, 2, 3]})

    assert json.loads(output.read_text()) == {"new": [1, 2, 3]}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []

    with pytest.raises(TypeError):
        _write_json_atomic(output, {"bad": object()})
    assert json.loads(output.read_text()) == {"new": [1, 2, 3]}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_eval_run_does_not_score_token_cap_answer(monkeypatch):
    """Expected words in a cap-exhausted generation cannot set answer_ok."""
    import sys
    import types

    import tool_oracle.eval_tool_skill as eval_skill

    mlx_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    mlx_sample_utils.make_sampler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_sample_utils)

    call = eval_skill._CALL_RE.search(
        "lookup(recipe, result_item='minecraft:shears')"
    )
    monkeypatch.setattr(
        eval_skill,
        "answer_question",
        lambda *args, **kwargs: {
            "answer": "To craft shears you need: iron ingot.",
            "raw_answer": "<think>unfinished reasoning about shears and iron ingot",
            "raw_turns": [],
            "answer_span_ok": True,
            "answer_evaluable": False,
            "completion_observed": True,
            "finish_reason": "max_tokens",
            "generated_tokens": 256,
            "tool_call_attempted": True,
            "first_call": call,
            "n_calls": 1,
            "calls": [("recipe", "minecraft:shears")],
        },
    )
    item = {
        "category": "recipe",
        "table": "recipe",
        "key": "minecraft:shears",
        "question": "How do I craft shears?",
        "expected": ["minecraft:iron_ingot"],
    }

    result = eval_skill.run(None, None, None, [item], 256)[0]
    assert result["tool_call_ok"] is True
    assert result["answer_span_ok"] is True
    assert result["answer_evaluable"] is False
    assert result["answer_ok"] is False
    assert result["finish_reason"] == "max_tokens"
    assert result["answer_instrument"] == "mechanical-final-span-v3"


def test_unknown_tool_call_requires_exact_key():
    """The old table-only unknown contract cannot hide a wrong fake key."""
    from tool_oracle.eval_tool_skill import _CALL_RE, _tool_call_verdict

    item = {
        "category": "recipe",
        "table": "recipe",
        "key": "minecraft:lightsaber",
        "kind": "unknown",
    }
    wrong = _CALL_RE.search(
        "lookup(recipe, result_item='minecraft:dragon_sword')"
    )
    right = _CALL_RE.search(
        "lookup(recipe, result_item='minecraft:lightsaber')"
    )
    assert _tool_call_verdict(
        item, wrong, [("recipe", "minecraft:dragon_sword")]
    ) == (False, "exact_first_call")
    assert _tool_call_verdict(
        item, right, [("recipe", "minecraft:lightsaber")]
    ) == (True, "exact_first_call")


def test_tool_call_parser_requires_one_complete_envelope():
    """A lookup mention or repaired truncation is not an executable call."""
    from tool_oracle.eval_tool_skill import _parse_tool_call

    valid = _parse_tool_call(
        "<think>Use the oracle.</think>\n"
        "<tool_call>lookup(recipe, result_item='minecraft:shears')</tool_call>"
    )
    assert valid is not None
    assert valid.groups() == ("recipe", "result_item", "minecraft:shears")
    assert _parse_tool_call(
        "I might use lookup(recipe, result_item='minecraft:shears')."
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(recipe, result_item='minecraft:shears')"
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(recipe, result_item='minecraft:shears')</tool_call>"
        "<tool_call>lookup(recipe, result_item='minecraft:piston')</tool_call>"
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(recipe, result_item='minecraft:shears')</tool_call>"
        "<tool_call>lookup(recipe, result_item='minecraft:piston')"
    ) is None
    assert _parse_tool_call(
        "prefix <tool_call>lookup(recipe, result_item='minecraft:shears')"
        "</tool_call>"
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(recipe, result_item='minecraft:shears')"
        "</tool_call> trailing answer"
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(recipe, wrong='minecraft:shears')</tool_call>"
    ) is None
    assert _parse_tool_call(
        "<tool_call>lookup(not_a_table, key='minecraft:shears')</tool_call>"
    ) is None


def test_tool_call_parser_enforces_each_registered_keyword():
    from tool_oracle.eval_tool_skill import _LOOKUP_KEY_ARGUMENTS, _parse_tool_call

    for table, argument in _LOOKUP_KEY_ARGUMENTS.items():
        valid = f"<tool_call>lookup({table}, {argument}='minecraft:x')</tool_call>"
        wrong = f"<tool_call>lookup({table}, wrong='minecraft:x')</tool_call>"
        assert _parse_tool_call(valid) is not None
        assert _parse_tool_call(wrong) is None


def test_usage_no_call_rejects_malformed_call_attempt():
    """A malformed call is not equivalent to choosing not to call a tool."""
    from tool_oracle.eval_tool_skill import _tool_call_verdict

    item = {"category": "usage", "kind": "usage"}
    assert _tool_call_verdict(item, None, [], False) == (True, "no_call")
    assert _tool_call_verdict(item, None, [], True) == (False, "no_call")


def test_malformed_bare_lookup_is_still_a_call_attempt():
    from tool_oracle.eval_tool_skill import _tool_call_attempted

    assert _tool_call_attempted("lookup(recipe, wrong='minecraft:shears')")
    assert _tool_call_attempted("lookup(recipe")
    assert not _tool_call_attempted("I cannot look up that information.")


def test_answer_question_retains_each_raw_turn(monkeypatch):
    """The audit packet preserves tool and answer generations verbatim."""
    import sys
    import types

    import tool_oracle.eval_tool_skill as eval_skill

    steps = iter([
        "<think>query it</think>\n"
        "<tool_call>lookup(recipe, result_item='minecraft:shears')</tool_call>"
        "<|im_end|>",
        "<think>use the returned row</think>\n"
        "To craft shears you need iron ingot.<|im_end|>",
    ])
    mlx = types.ModuleType("mlx_lm")
    mlx.generate = lambda *args, **kwargs: next(steps)
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(
        eval_skill,
        "_execute",
        lambda db, table, key: ["minecraft:iron_ingot"],
    )

    class Tokenizer:
        conversations = []

        @classmethod
        def apply_chat_template(cls, convo, **kwargs):
            cls.conversations.append([dict(message) for message in convo])
            return repr(convo)

    manifest = eval_skill._load_tool_manifest()
    result = eval_skill.answer_question(
        None,
        Tokenizer(),
        None,
        "How do I craft shears?",
        max_tokens=64,
        tool_manifest=manifest,
    )

    assert result["calls"] == [("recipe", "minecraft:shears")]
    assert result["answer"] == "To craft shears you need iron ingot."
    assert result["answer_evaluable"] is True
    assert result["completion_observed"] is True
    assert len(result["raw_turns"]) == 2
    assert result["raw_turns"][0]["parsed_call"] == {
        "table": "recipe",
        "argument": "result_item",
        "key": "minecraft:shears",
    }
    assert result["call_records"] == [result["raw_turns"][0]["parsed_call"]]
    assert result["raw_turns"][1]["parsed_call"] is None
    assert Tokenizer.conversations[0][:2] == [
        {"role": "system", "content": manifest.text},
        {"role": "user", "content": "How do I craft shears?"},
    ]
    assert Tokenizer.conversations[1][0] == Tokenizer.conversations[0][0]
    assert [message["role"] for message in Tokenizer.conversations[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    import hashlib

    assert result["raw_turns"][0]["prompt_sha256"] == hashlib.sha256(
        repr(Tokenizer.conversations[0]).encode("utf-8")
    ).hexdigest()


def test_answer_question_records_malformed_call_as_attempt(monkeypatch):
    """A truncated envelope is retained but neither executed nor scored."""
    import sys
    import types

    import tool_oracle.eval_tool_skill as eval_skill

    mlx = types.ModuleType("mlx_lm")
    mlx.generate = lambda *args, **kwargs: (
        "<tool_call>lookup(recipe, result_item='minecraft:shears')"
    )
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    class Tokenizer:
        @staticmethod
        def apply_chat_template(convo, **kwargs):
            return repr(convo)

    result = eval_skill.answer_question(
        None,
        Tokenizer(),
        None,
        "How do I craft shears?",
        max_tokens=64,
        tool_manifest=eval_skill._load_tool_manifest(),
    )

    assert result["calls"] == []
    assert result["tool_call_attempted"] is True
    assert result["answer_span_ok"] is False
    assert result["answer_evaluable"] is False
    assert result["raw_turns"][0]["tool_call_attempted"] is True


def test_multihop_tool_call_requires_exact_second_hop():
    """Call count alone does not authenticate the second lookup target."""
    from tool_oracle.eval_tool_skill import _CALL_RE, _tool_call_verdict

    item = {
        "category": "recipe",
        "table": "recipe",
        "key": "minecraft:piston",
        "kind": "multihop",
        "second_table": "recipe",
        "second_key": "minecraft:iron_ingot",
    }
    first = _CALL_RE.search(
        "lookup(recipe, result_item='minecraft:piston')"
    )
    wrong_calls = [
        ("recipe", "minecraft:piston"),
        ("recipe", "minecraft:stick"),
    ]
    right_calls = [
        ("recipe", "minecraft:piston"),
        ("recipe", "minecraft:iron_ingot"),
    ]
    assert _tool_call_verdict(item, first, wrong_calls) == (
        False,
        "exact_first_two_calls",
    )
    assert _tool_call_verdict(item, first, right_calls) == (
        True,
        "exact_first_two_calls",
    )


def test_cuda_freezes_eos_pad_and_device_across_peft_attach(tmp_path):
    pytest.importorskip("torch", reason="CUDA eval module imports torch")
    """The CWD-independent CUDA entry point retains frozen base controls."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    probe = textwrap.dedent(
        """
        import importlib.util
        import inspect
        import os
        import sys
        import types

        peft = types.ModuleType("peft")
        peft.PeftModel = object
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = object
        transformers.AutoTokenizer = object
        sys.modules["peft"] = peft
        sys.modules["transformers"] = transformers

        spec = importlib.util.spec_from_file_location(
            "cuda_eval_under_test", os.environ["MCAGENT_CUDA_EVAL_PATH"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Parameter:
            device = "cuda:0"

        class Model:
            generation_config = types.SimpleNamespace(eos_token_id=[2, 3])
            def parameters(self):
                return iter([Parameter()])

        tokenizer = types.SimpleNamespace(pad_token_id=7, eos_token_id=9)
        controls = module._freeze_generation_controls(Model(), tokenizer)
        answer = module.make_answer_question_cuda(Model(), tokenizer, controls)
        nonlocals = inspect.getclosurevars(answer).nonlocals

        tokenizer.pad_token_id = 70
        tokenizer.eos_token_id = 90
        assert controls == {
            "pad_token_id": 7,
            "eos_token_ids": [2, 3],
            "input_device": "cuda:0",
        }
        assert nonlocals["pad_token_id"] == 7
        assert nonlocals["eos_token_ids"] == [2, 3]
        assert nonlocals["input_device"] == "cuda:0"
        """
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MCAGENT_CUDA_EVAL_PATH"] = str(
        Path("tool_oracle_cuda/eval_tool_skill_cuda.py").resolve()
    )
    subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        cwd=tmp_path,
        env=env,
    )
