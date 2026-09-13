"""Generate VERIFIED two-hop query-traces for the tool-augmented model.

A single-hop trace teaches "route one question to one lookup". A multi-hop trace
teaches CHAINING: answer a question that needs the result of the first lookup to
decide the second. The cleanest verifiable chain is recipe -> recipe: craft X,
one of whose ingredients Y is itself craftable, so "how do I make X and its
ingredient Y?" requires looking up X, reading that it needs Y, then looking up Y.

Each trace is a 6-message conversation (two tool calls, each followed by the REAL
tool return):

    user      : <question about X and its craftable ingredient Y>
    assistant : <think>get X's recipe first</think><tool_call>lookup(recipe, X)</tool_call>  [STOP]
    user      : <result>ingredients of X</result>
    assistant : <think>X needs Y; now look up Y</think><tool_call>lookup(recipe, Y)</tool_call>  [STOP]
    user      : <result>ingredients of Y</result>
    assistant : <answer built from BOTH results>

Correct by construction (both lookups come from the DB, Y is verified to be an
ingredient of X) and re-checked by verify_multihop().

    PYTHONPATH=. python3 tool_oracle/gen_multihop_traces.py minecraft.db out.jsonl [--seed N]
"""
import collections
import json
import random
import sys
from typing import Dict, Iterator, List

from tool_oracle.gen_query_traces import (
    _CALL_RE, _RESULT_RE, _bare, _english, _english_recipe,
    _english_depth, _english_sources, _depth_row_str, _source_label,
)
from tool_oracle.lookup import OracleDB

_PHRASINGS = [
    "I want to craft {x} from scratch. What are its ingredients, and how do I make the {y} it needs?",
    "How do I craft {x}, and how do I make its {y} component?",
    "To build a {x} I need {y} among other things - what goes into both?",
    "Walk me through crafting {x}, including making the {y} for it.",
    "What do I need for a {x}, and what does its {y} take to craft?",
]

# Cross-table chain: breeding_food -> recipe ("what animal X eats, and craft that food").
# The table is the animal-FOOD tag (what it eats), not breeding-specific -- so
# phrasings/answers say "eats/feed", never "breeds with".
_CROSS_PHRASINGS = [
    "What does a {a} eat, and how do I craft that {y}?",
    "What do you feed a {a}, and how is the {y} made?",
    "I want to feed my {a}s - what do they eat, and how do I craft the {y}?",
    "A {a} eats {y}; what else does it eat and how do I make the {y}?",
]


def _multihop_trace(question, x, groups_x, y, groups_y) -> Dict:
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant",
         "content": f"<think>Answer needs two steps: first get {_bare(x)}'s recipe, "
                    f"then look up its ingredient {_bare(y)}.</think>\n"
                    f"<tool_call>lookup(recipe, result_item='{x}')</tool_call>"},
        {"role": "user", "content": f"<result>{', '.join(groups_x)}</result>"},
        {"role": "assistant",
         "content": f"<think>{_bare(x)} needs {_bare(y)}; now look up how to craft "
                    f"{_bare(y)}.</think>\n"
                    f"<tool_call>lookup(recipe, result_item='{y}')</tool_call>"},
        {"role": "user", "content": f"<result>{', '.join(groups_y)}</result>"},
        {"role": "assistant",
         "content": f"To craft {_bare(x)} you need: {_english_recipe(groups_x)}. "
                    f"To make the {_bare(y)}, you need: {_english_recipe(groups_y)}."},
    ]}


def generate(db: OracleDB, rng: random.Random, limit: int = 250, max_per_sub: int = 6) -> Iterator[Dict]:
    """Two-hop recipe->recipe traces, capped per sub-ingredient for variety."""
    items = [r[0] for r in db.con.execute("SELECT DISTINCT result_item FROM recipe")]
    items = [x for x in items if "_from_" not in x]
    craftable = set(items)
    per_sub: collections.Counter = collections.Counter()
    emitted = 0
    for x in items:
        if emitted >= limit:
            break
        # Select from X's CANONICAL (grouped, single-method) ingredients, not
        # the flat all-methods-merged list -- a sub-ingredient that only shows
        # up in a different recipe file (e.g. black_carpet's dye-recolor
        # variant lists other-color carpets as ingredients; the canonical
        # wool-based recipe doesn't) would never appear in the <result> this
        # trace actually shows, breaking the chain.
        groups_x = db.recipe_ingredient_groups(x)
        flat_x = {sub for slot in groups_x for sub in slot.split("/")}
        subs = [y for y in flat_x if not y.startswith("#") and y in craftable and y != x]
        # prefer a sub-ingredient we have not over-used yet
        subs.sort(key=lambda y: per_sub[y])
        if not subs:
            continue
        y = subs[0]
        if per_sub[y] >= max_per_sub:
            continue
        groups_y = db.recipe_ingredient_groups(y)
        if not groups_y:
            continue
        per_sub[y] += 1
        emitted += 1
        q = rng.choice(_PHRASINGS).format(x=_bare(x), y=_bare(y))
        yield _multihop_trace(q, x, groups_x, y, groups_y)


def generate_cross(db: OracleDB, rng: random.Random, limit: int = 60) -> Iterator[Dict]:
    """Cross-table two-hop traces: breeding_food -> recipe. Feed animal A whose
    breeding food Y is itself craftable; the chain is get A's foods, see Y is
    craftable, look up Y's recipe."""
    craftable = set(x for (x,) in db.con.execute("SELECT DISTINCT result_item FROM recipe")
                    if "_from_" not in x)
    emitted = 0
    for (animal,) in db.con.execute("SELECT DISTINCT animal FROM breeding_food"):
        if emitted >= limit:
            break
        foods = db.breeding_foods(animal)
        craft_foods = [f for f in foods if f in craftable]
        if not craft_foods:
            continue
        y = craft_foods[0]
        ings_y = db.recipe_ingredients(y)
        if not ings_y:
            continue
        groups_y = db.recipe_ingredient_groups(y)
        emitted += 1
        q = rng.choice(_CROSS_PHRASINGS).format(a=animal, y=_bare(y))
        yield {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant",
             "content": f"<think>Two steps: first get what {animal}s eat, then look "
                        f"up how to craft it.</think>\n"
                        f"<tool_call>lookup(breeding_food, animal='{animal}')</tool_call>"},
            {"role": "user", "content": f"<result>{', '.join(foods)}</result>"},
            {"role": "assistant",
             "content": f"<think>{animal} eats {_bare(y)}, which is craftable; now look up "
                        f"its recipe.</think>\n"
                        f"<tool_call>lookup(recipe, result_item='{y}')</tool_call>"},
            {"role": "user", "content": f"<result>{', '.join(groups_y)}</result>"},
            {"role": "assistant",
             "content": f"{animal.capitalize()}s eat: {_english(foods)}. "
                        f"To craft the {_bare(y)}, you need: {_english_recipe(groups_y)}."},
        ]}


def verify_multihop(trace: Dict, db: OracleDB) -> bool:
    """Re-execute both hops and confirm the chain is real. Handles recipe->recipe
    (Y is an ingredient of X) and breeding_food->recipe (Y is a breeding food of
    A). In both cases each stated <result> must equal the DB and the final answer
    must cover the second hop's ingredients."""
    m = trace["messages"]
    if len(m) != 6:
        return False
    c1, c2 = _CALL_RE.search(m[1]["content"]), _CALL_RE.search(m[3]["content"])
    if not c1 or not c2 or c2.group(1) != "recipe":
        return False
    t1, k1, y = c1.group(1), c1.group(2), c2.group(2)
    stated_1 = set(_RESULT_RE.search(m[2]["content"]).group(1).split(", "))
    stated_y = set(_RESULT_RE.search(m[4]["content"]).group(1).split(", "))
    groups_y = db.recipe_ingredient_groups(y)
    answer = m[5]["content"].lower()
    if t1 == "recipe":
        hop1 = db.recipe_ingredient_groups(k1)  # grouped: <result> for a recipe hop is grouped
    elif t1 == "breeding_food":
        hop1 = db.breeding_foods(k1)
    else:
        return False
    if stated_1 != set(hop1) or stated_y != set(groups_y):
        return False
    # y must really be one of X's ingredients -- for a grouped hop1, y may sit
    # inside a "/"-joined OR-slot rather than being its own exact entry.
    hop1_items = {sub for slot in hop1 for sub in slot.split("/")} if t1 == "recipe" else set(hop1)
    if y not in hop1_items or not groups_y:       # the chain must be real
        return False
    # Coverage vs groups_y, not the flat (all-methods-merged) ingredient list:
    # the answer names ONE recipe method's ingredients (matching _english_recipe),
    # and a multi-method item (e.g. stick: planks OR bamboo, two separate recipe
    # files) can have a flat ingredient set wider than any single method's.
    return all(_bare(sub) in answer for slot in groups_y for sub in slot.split("/"))


_SOURCE_DEPTH_PHRASINGS = [
    "Where do I find {n}, and at what Y level?",
    "Where can I get {n}? What depth should I mine at?",
    "How do I get {n}, and where's the best place to look for it?",
    "I need {n} - where do I mine it, and how deep?",
]


def _find_ore_block(db: OracleDB, sources: List[str]):
    """The first block source (of a loot_sources() result) that's a real
    minable ore with depth data, or None."""
    for s in sources:
        if s.startswith("minecraft:blocks/"):
            cand = f"minecraft:{s[len('minecraft:blocks/'):]}"
            if db.ore_depths(cand):
                return cand
    return None


def _build_source_depth_trace(db: OracleDB, question: str, drop_item: str, think1=None):
    """Build one loot_source->ore_depth trace for a specific (question,
    drop_item), or None if drop_item doesn't resolve to a real depth-linked
    ore. Shared by the main sweep and the alias normalize-cases below."""
    sources = db.loot_sources(drop_item)
    if not sources:
        return None
    block = _find_ore_block(db, sources)
    if not block:
        return None
    depth_rows = db.ore_depths(block)
    n = _bare(drop_item)
    block_label = _source_label(f"minecraft:blocks/{block.split(':', 1)[-1]}")
    think1 = think1 or (f"Two steps: first find what drops {n}, then check the Y level "
                         f"for whichever source is a minable ore.")
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant",
         "content": f"<think>{think1}</think>\n"
                    f"<tool_call>lookup(loot_source, drop_item='{drop_item}')</tool_call>"},
        {"role": "user", "content": f"<result>{', '.join(sources)}</result>"},
        {"role": "assistant",
         "content": f"<think>{n} comes from {block_label}, which is an ore block; look up "
                    f"its Y-level range.</think>\n"
                    f"<tool_call>lookup(ore_depth, block='{block}')</tool_call>"},
        {"role": "user",
         "content": f"<result>{', '.join(_depth_row_str(r) for r in depth_rows)}</result>"},
        {"role": "assistant",
         "content": f"You can get {n} by {_english_sources(sources)}. "
                    f"{block_label.capitalize()} generates {_english_depth(depth_rows)}."},
    ]}


def generate_source_depth(db: OracleDB, rng: random.Random, limit: int = 100) -> Iterator[Dict]:
    """Two-hop: loot_source(drop_item) -> a block source that's a real minable
    ore -> ore_depth(block). Combines HOW to get an item (mine/kill, from
    loot_source) with WHERE (Y-level, from ore_depth) into one answer -- the
    actual "where do I find iron" question, which needs both: loot_source
    alone answers "mine iron ore" but says nothing about depth."""
    drop_items = [r[0] for r in db.con.execute("SELECT DISTINCT drop_item FROM loot")]
    rng.shuffle(drop_items)
    emitted = 0
    for drop_item in drop_items:
        if emitted >= limit:
            break
        n = _bare(drop_item)
        q = rng.choice(_SOURCE_DEPTH_PHRASINGS).format(n=n)
        trace = _build_source_depth_trace(db, q, drop_item)
        if trace is None:
            continue
        yield trace
        emitted += 1


# Alias cases: "iron"/"gold"/"copper" alone most naturally means the raw
# resource you mine, not the ingot or the ore block -- a real play-test
# question ("Where do I find Iron?") a plain drop_item sweep never phrases
# this way (it only ever asks about "raw iron", the canonical bare name).
# Several paraphrasings per alias: one example wasn't enough weight for the
# model to override its default "use the noun literally as the item id"
# behavior (confirmed live: v18 queried the literal, nonexistent
# minecraft:iron and honestly declined -- safe, but not useful).
_SOURCE_DEPTH_NORMALIZE_CASES = [
    ("Where do I find Iron?", "minecraft:raw_iron"),
    ("Where can I get some iron?", "minecraft:raw_iron"),
    ("How do I get iron in Minecraft?", "minecraft:raw_iron"),
    ("I need iron - where do I mine it?", "minecraft:raw_iron"),
    ("Where does iron come from?", "minecraft:raw_iron"),
    ("Where can I get gold?", "minecraft:raw_gold"),
    ("How do I find gold in Minecraft?", "minecraft:raw_gold"),
    ("I need some gold - where do I get it?", "minecraft:raw_gold"),
    ("Where can I get copper?", "minecraft:raw_copper"),
    ("How do I find copper?", "minecraft:raw_copper"),
    ("Where do diamonds come from?", "minecraft:diamond"),
]


def generate_source_depth_normalize(db: OracleDB) -> Iterator[Dict]:
    for question, drop_item in _SOURCE_DEPTH_NORMALIZE_CASES:
        n = _bare(drop_item)
        think1 = (f"This asks where to get {n} loosely (\"iron\"/\"gold\"/\"copper\" means the "
                   f"raw resource you mine). First find what drops {n}, then check its Y level.")
        trace = _build_source_depth_trace(db, question, drop_item, think1=think1)
        if trace is not None:
            yield trace


def verify_source_depth(trace: Dict, db: OracleDB) -> bool:
    """Re-execute both hops of a loot_source->ore_depth chain and confirm each
    stated <result> equals the DB, the queried block really is one of the
    resolved sources, and the answer covers both the source names and the
    primary depth band's real Y numbers."""
    m = trace["messages"]
    if len(m) != 6:
        return False
    c1, c2 = _CALL_RE.search(m[1]["content"]), _CALL_RE.search(m[3]["content"])
    if not c1 or not c2 or c1.group(1) != "loot_source" or c2.group(1) != "ore_depth":
        return False
    drop_item, block = c1.group(2), c2.group(2)
    stated_1 = set(_RESULT_RE.search(m[2]["content"]).group(1).split(", "))
    stated_2 = set(_RESULT_RE.search(m[4]["content"]).group(1).split(", "))
    sources = db.loot_sources(drop_item)
    depth_rows = db.ore_depths(block)
    depth_strs = [_depth_row_str(r) for r in depth_rows]
    if stated_1 != set(sources) or stated_2 != set(depth_strs):
        return False
    block_bare = block.split(":", 1)[-1]
    if f"minecraft:blocks/{block_bare}" not in sources or not depth_rows:
        return False
    answer = m[5]["content"].lower()
    if not all(_source_label(s) in answer for s in sources):
        return False
    return str(depth_rows[0][2]) in answer and str(depth_rows[0][3]) in answer


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: gen_multihop_traces.py <db_path> [out.jsonl] [--seed N]")
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "multihop_traces.jsonl"
    db = OracleDB(sys.argv[1])
    rng = random.Random(seed)
    traces = list(generate(db, rng))
    ok = sum(verify_multihop(t, db) for t in traces)
    with open(out, "w") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")
    print(f"wrote {len(traces)} two-hop traces to {out}")
    print(f"self-verified {ok}/{len(traces)} traces")


if __name__ == "__main__":
    main()
