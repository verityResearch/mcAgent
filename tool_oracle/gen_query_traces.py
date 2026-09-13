"""Generate VERIFIED multi-turn query-traces for the tool-augmented model.

Each trace is a 4-message conversation that teaches the SKILL:

    user      : <question>
    assistant : <think>…</think>\n<tool_call>lookup(table, key='…')</tool_call>   [STOPS]
    user      : <result>…</result>                                     (the REAL tool return)
    assistant : <answer composed from ONLY that result>

Multi-turn is load-bearing: the assistant emits the tool call and STOPS, so at
inference the result comes from the actual tool, not a hallucinated one. The
model learns two things -- (1) route a question to the right lookup, (2) compose
an answer from a returned result -- neither of which is memorizing the fact.

Correct by construction (the lookup comes from the template, the <result> is the
DB's actual return, the answer is built from it) and re-checked by
verify_trace(): result equals the DB, answer covers exactly the returned rows.
Phrasing is varied per item so the skill generalizes past one template.

    PYTHONPATH=. python3 tool_oracle/gen_query_traces.py minecraft.db traces.jsonl [--seed N]
"""
import json
import random
import re
import sys
from typing import Dict, Iterator, List

from tool_oracle.lookup import OracleDB


def _bare(i: str) -> str:
    return i.split(":", 1)[-1].replace("_", " ")


def _english(items: List[str]) -> str:
    names = [_bare(i) for i in items]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f", or {names[-1]}"


def _english_recipe(groups: List[str]) -> str:
    """Recipe-specific phrasing: each entry is one AND-slot, "/"-joined when it
    holds multiple OR-alternatives (see OracleDB.recipe_ingredient_groups).
    Slots are joined with "and" (all required); alternatives within a slot
    with "or", parenthesized when there's more than one so "stick and (coal
    or charcoal)" can't misparse as three flat options the way plain "stick,
    coal, or charcoal" did (the actual bug this function replaces)."""
    def _one(token: str) -> str:
        names = [_bare(t) for t in token.split("/")]
        if len(names) == 1:
            return names[0]
        alt = names[0] + f" or {names[1]}" if len(names) == 2 \
            else ", ".join(names[:-1]) + f", or {names[-1]}"
        return f"({alt})"
    parts = [_one(g) for g in groups]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _source_label(src: str) -> str:
    """A loot source id ('minecraft:blocks/iron_ore', 'minecraft:entities/husk')
    -> its player-facing name ('iron ore', 'husk'). Unlike _bare, this strips
    the blocks//entities/ path segment too, not just the minecraft: namespace.

    entities/ sources can carry a color/variant segment ('entities/sheep/pink'
    -- a pink sheep specifically, not a generic entity called "pink"); a naive
    last-segment split would read that variant alone and lose "sheep"
    entirely. Reversing the sub-segments reads naturally: variant before mob."""
    parts = src.split(":", 1)[-1].split("/")
    kind, rest = parts[0], parts[1:]
    if kind == "entities" and len(rest) > 1:
        return " ".join(reversed(rest)).replace("_", " ")
    return (rest[-1] if rest else src).replace("_", " ")


def _english_sources(sources: List[str]) -> str:
    """Phrase loot_sources's block/entity-only rows as "mine X" / "kill Y".
    When both kinds are present, joined with "; or" -- a plain "or" would read
    as one more item in the immediately-preceding list ("...redstone wire or
    killing witch") instead of a separate mine-vs-kill clause."""
    def _join(names: List[str]) -> str:
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f", or {names[-1]}"
    blocks = sorted({_source_label(s) for s in sources if s.startswith("minecraft:blocks/")})
    mobs = sorted({_source_label(s) for s in sources if s.startswith("minecraft:entities/")})
    parts = []
    if blocks:
        parts.append(f"mining {_join(blocks)}")
    if mobs:
        parts.append(f"killing {_join(mobs)}")
    return "; or ".join(parts)


def _depth_row_str(row) -> str:
    """One ore_depth row -> its <result> token. Compact and parseable, not
    meant to be player-facing (that's _english_depth's job)."""
    _, dist_type, lo, hi, count = row
    return f"Y{lo}to{hi}({dist_type},count{count})"


def _english_depth(rows) -> str:
    """Phrase ore_depth rows (highest count first) as real Y-level guidance.
    "count" is the per-chunk attempt count for that placement -- a genuine,
    comparable-across-placements frequency signal (see build_db.py), so
    leading with the highest-count band is a defensible "most likely" claim,
    not a fabricated one. trapezoid's peak is its real midpoint (that's the
    actual shape of a trapezoid height distribution); uniform has no peak, so
    it's described as an even spread instead of inventing one."""
    def band(row):
        _, dist_type, lo, hi, count = row
        rng = f"from Y {lo} to Y {hi}"
        if dist_type == "trapezoid":
            peak = round((lo + hi) / 2)
            return f"{rng} (most common around Y {peak})"
        if dist_type == "uniform":
            return f"{rng} (evenly spread)"
        return rng
    if not rows:
        return ""
    parts = [band(rows[0])]
    if len(rows) > 1 and (rows[1][2], rows[1][3]) != (rows[0][2], rows[0][3]):
        parts.append(band(rows[1]))
    text = "; it also generates ".join(parts)
    if len(rows) > 2:
        text += ", and in smaller amounts elsewhere"
    return text


# Many phrasings per category -- terse AND conversational/first-person/
# problem-framed -- so the routing skill fires on natural wording, not just the
# template shape. (Round-3 fix: the robustness probe showed the skill was
# brittle to phrasing it hadn't seen; these span that distribution.)
_PHRASINGS = {
    # NOTE: this table is built from minecraft:<animal>_food item tags, which are
    # what the animal EATS (temptation/feeding) -- NOT the breeding-specific item.
    # For some animals (e.g. horses) food is broader than what actually breeds
    # them (golden apple/carrot only), so phrasings/answers say "eats/feed", never
    # "breeds with", to avoid asserting a falsehood.
    "breeding_food": [
        "What do {n}s eat in Minecraft?",
        "What can you feed a {n}?",
        "What food does a {n} eat?",
        "Which items will a {n} eat?",
        "My {n} is hurt - what can I feed it?",
        "I'm trying to tempt a {n} - what does it eat?",
        "What do you hold to lead a {n} around?",
        "What food do {n}s accept?",
        "Starting a {n} farm - what do they eat?",
        "Can you tell me what a {n} will eat?",
    ],
    "recipe": [
        "What items do you need to craft {n}?",
        "How do you make {n} in Minecraft?",
        "What's the recipe for {n}?",
        "What do I craft {n} from?",
        "I need a {n} - what do I put together to make one?",
        "Walk me through crafting a {n}.",
        "What goes into making a {n}?",
        "How is a {n} crafted?",
        "Remind me of the crafting combo for {n}?",
        "I want to build a {n}. What materials do I need?",
        "What are the ingredients for a {n}?",
    ],
    "enchantment": [
        "What is the maximum level of {n}?",
        "How high does {n} go?",
        "What's the max level for the {n} enchantment?",
        "Is there a level cap on {n}?",
        "How high can I push {n}?",
        "What's the ceiling on the {n} enchant?",
        "Max level for {n}, do you know?",
        "Up to what level does {n} go?",
    ],
    "loot": [
        "What does {n} drop?",
        "What can you get from {n}?",
        "What are the drops for {n}?",
        "If I kill a {n}, what drops out of it?",
        "Loot-wise, what comes off a {n}?",
        "When I break a {n}, what do I get back?",
        "What items does a {n} leave behind?",
        "What's the loot from a {n}?",
    ],
    # The REVERSE of loot: given an item, where does it come from. Restricted to
    # blocks/entities sources (see OracleDB.loot_sources) so the answer is
    # "mine X" / "kill Y", not a raw loot-table id.
    "loot_source": [
        "Where do I find {n}?",
        "Where can I get {n}?",
        "How do I obtain {n}?",
        "Where does {n} come from?",
        "What do I need to mine or kill to get {n}?",
        "How do I get some {n}?",
        "Where would I look for {n}?",
    ],
    # Depth data is keyed by BLOCK (the ore, not the item it drops) -- these
    # phrasings ask about the block directly. See the loot_source->ore_depth
    # multihop in gen_multihop_traces.py for the drop-item-phrased version
    # ("where do I find iron") that chains through loot_source first.
    "ore_depth": [
        "At what Y level does {n} generate?",
        "How deep should I mine to find {n}?",
        "What's the best Y level for {n}?",
        "Where in the world does {n} spawn?",
        "What height range does {n} generate at?",
    ],
    "villager_trade": [
        "In the villager trade {n}, what does the villager want and give?",
        "What are the terms of the {n} villager trade?",
        "For the {n} trade, what does the villager take and hand over?",
    ],
    "tag": [
        "What items belong to the tag {n}?",
        "Which items are in {n}?",
        "What's in the {n} tag?",
    ],
    # For LARGE tags (>12 members) enumerating everything doesn't scale; the
    # answer shape is count + a sample, so the phrasing leans that way too.
    "tag_large": [
        "How many items are tagged {n}, and can you name a few?",
        "Roughly how many items are in the {n} tag? Give some examples.",
        "What are some items in the {n} tag, and how many are there total?",
    ],
    "jukebox_song": [
        "How many seconds long is the music disc {n}?",
        "What's the length of the {n} music disc?",
        "How long does the {n} disc play for?",
    ],
    "painting": [
        "What are the width and height in blocks of the {n} painting?",
        "How big is the {n} painting?",
        "What are the dimensions of the {n} painting?",
    ],
    # Same phrasing family as a plain "what is X used for" question -- the
    # routing decision (query item_use vs. decline) depends on whether the item
    # HAS component data, not on how the question is worded.
    "item_use": [
        "What is {art} {n} used for?",
        "What do you use {art} {n} for?",
        "What's the purpose of {art} {n}?",
        "How do you use {art} {n}?",
        "What does {art} {n} do?",
    ],
}

# Obviously-fake subjects per category, used to teach EMPTY-RESULT handling: the
# model queries, the tool returns (no rows), and the assistant must DECLINE
# rather than fabricate. (Round-3 fix: the probe showed 0/5 declines -- the model
# said "Excalibur has a maximum level of 3" for a nonexistent enchantment.)
_FAKE = {
    "recipe": ["minecraft:dragon_sword", "minecraft:lightsaber", "minecraft:magic_wand",
               "minecraft:plasma_rifle", "minecraft:infinity_gauntlet", "minecraft:cloud_block",
               "minecraft:phoenix_feather", "minecraft:mithril_ingot", "minecraft:laser_drill",
               "minecraft:soul_cannon"],
    "breeding_food": ["phoenix", "unicorn", "griffin", "dragon", "kraken", "basilisk",
                      "chimera", "wyvern"],
    "enchantment": ["minecraft:excalibur", "minecraft:godslayer", "minecraft:omnipotence",
                    "minecraft:soul_reaper", "minecraft:infinity_edge", "minecraft:doombringer"],
    "loot": ["minecraft:entities/chaos_wraith", "minecraft:entities/shadow_demon",
             "minecraft:entities/mega_boss", "minecraft:entities/void_lord",
             "minecraft:entities/frost_titan", "minecraft:entities/storm_elemental"],
    "loot_source": ["minecraft:mithril_ingot", "minecraft:dragon_scale", "minecraft:soul_gem",
                    "minecraft:kryptonite", "minecraft:unobtainium", "minecraft:phoenix_down"],
    "ore_depth": ["minecraft:mithril_ore", "minecraft:adamantite_ore", "minecraft:orichalcum_ore",
                 "minecraft:starlight_ore"],
}

_DECLINE_MARKERS = ("i don't have", "no data", "couldn't find", "isn't in", "not in the",
                    "no information", "doesn't exist", "not a real")

# Usage/function questions ("what is X used for"). Round-10: a live-eval smoke
# test showed the model answering "carrot on a stick is used to serve carrot
# soup" -- a fabrication with NO grounding in any tool result (it queried
# recipe(carrot_on_a_stick), got the real ingredients back, then invented an
# unrelated "use"). Round-11: the item_use table (reports/minecraft/components/
# item/*.json -- food/durability/attack-damage/equip-slot) now answers a real
# SUBSET of usage questions for the ~180 items that have that data; for items
# with NO item_use row this question type still has nothing to query, so the
# no-tool decline from round-10 is kept for exactly that remainder (checked
# against the live DB in generate_usage_declines, not a fixed list -- an item
# that gains item_use data in a future game version stops being a decline case
# automatically).
_USAGE_ITEMS = ["carrot on a stick", "compass", "clock", "spyglass", "lead",
               "name tag", "shears", "shield", "elytra", "totem of undying",
               "trident", "fishing rod", "flint and steel", "map"]


def _item_use_facts(row) -> List[str]:
    """(nutrition, saturation, max_damage, attack_damage, equip_slot) -> the
    non-None fields as 'key=value' strings, in a fixed order. Floats are
    rounded to 2dp -- Minecraft's raw component dump carries float32 noise
    (e.g. saturation 1.8000001 for beef) that would otherwise leak into
    training answers verbatim."""
    nutrition, saturation, max_damage, attack_damage, equip_slot = row
    facts = []
    if nutrition is not None:
        facts.append(f"nutrition={nutrition}")
    if saturation is not None:
        facts.append(f"saturation={round(saturation, 2)}")
    if max_damage is not None:
        facts.append(f"max_damage={max_damage}")
    if attack_damage is not None:
        facts.append(f"attack_damage={round(attack_damage, 2)}")
    if equip_slot is not None:
        facts.append(f"equip_slot={equip_slot}")
    return facts


def _article(n: str) -> str:
    return "an" if n[:1].lower() in "aeiou" else "a"


def _item_use_answer(n: str, facts: List[str]) -> str:
    """Build a natural-language answer from the non-None item_use facts."""
    parts = []
    by_key = dict(f.split("=", 1) for f in facts)
    if "nutrition" in by_key or "saturation" in by_key:
        bits = []
        if "nutrition" in by_key:
            bits.append(f"{by_key['nutrition']} hunger")
        if "saturation" in by_key:
            bits.append(f"{by_key['saturation']} saturation")
        parts.append(f"eating {_article(n)} {n} restores {' and '.join(bits)}")
    if "attack_damage" in by_key:
        parts.append(f"it deals {by_key['attack_damage']} extra attack damage as a weapon")
    if "equip_slot" in by_key:
        parts.append(f"it's worn in the {by_key['equip_slot']} slot")
    if "max_damage" in by_key:
        parts.append(f"it has {by_key['max_damage']} uses before breaking")
    return f"{_article(n).capitalize()} {n}: " + "; ".join(parts) + "."

# Key-normalization cases: natural wording -> canonical DB key. The round-3 probe
# showed the model builds wrong keys from conversational phrasing ("a pair of
# shears" -> pair_of_shears; "calf" -> recipe/calf; a bare mob name -> missing the
# entities/ loot path). Each (surface_question, table, canonical_key) teaches the
# extraction explicitly in the think step. Skipped if the key has no DB row.
_NORMALIZE_CASES = [
    ("What two things go into a pair of shears?", "recipe", "minecraft:shears"),
    ("How do I make a set of iron leggings?", "recipe", "minecraft:iron_leggings"),
    ("What do I need for a stack of torches?", "recipe", "minecraft:torch"),
    ("What do my cows eat?", "breeding_food", "cow"),
    ("What food do baby chickens - I mean chickens - eat?", "breeding_food", "chicken"),
    ("What do I feed my sheep?", "breeding_food", "sheep"),
    ("If I kill a creeper, what drops out of it?", "loot", "minecraft:entities/creeper"),
    ("What comes off a zombie when it dies?", "loot", "minecraft:entities/zombie"),
    ("What does a skeleton drop when killed?", "loot", "minecraft:entities/skeleton"),
    ("What loot does an enderman leave?", "loot", "minecraft:entities/enderman"),
    # NOTE: "iron"/"gold"/"copper"/"diamond" alias examples used to live here
    # (as plain loot_source cases), but those items now all resolve to real
    # ore_depth data, so they're taught as loot_source->ore_depth CHAINS
    # instead -- see _SOURCE_DEPTH_NORMALIZE_CASES in gen_multihop_traces.py.
    # Keeping both would teach two different answer shapes for the same
    # question (a shallow "mine X" and a depth-enriched one).
]

# Implicit-goal recipe cases: a goal clause that does NOT name the item, followed
# by a second clause that does. The round-7 probe showed the model blends goal
# vocabulary into a fabricated key ("smelt ore faster" + "blast furnace" ->
# queried "blaze ore"/"blaze rod", neither real) -- no training phrasing had this
# two-clause goal-then-subject shape; every recipe phrasing named the item
# directly. Each think step explicitly names the item and calls out that the
# goal clause is not the subject.
_IMPLICIT_GOAL_CASES = [
    ("I want to smelt ore faster. How is a blast furnace put together?",
     "recipe", "minecraft:blast_furnace",
     "The goal (smelt ore faster) doesn't name an item; the actual subject is "
     "\"blast furnace\" in the second clause."),
    ("I keep getting lost at night. What do I need to build a compass?",
     "recipe", "minecraft:compass",
     "The goal (getting lost) doesn't name an item; the subject is \"compass\"."),
    ("My tools keep breaking too fast. How do I make a smithing table?",
     "recipe", "minecraft:smithing_table",
     "The goal (tools breaking) doesn't name an item; the subject is "
     "\"smithing table\"."),
    ("I want to store more items safely. How is a shulker box crafted?",
     "recipe", "minecraft:shulker_box",
     "The goal (storing items) doesn't name an item; the subject is "
     "\"shulker box\"."),
    ("I need to see underwater better. What goes into a spyglass?",
     "recipe", "minecraft:spyglass",
     "The goal (seeing underwater) doesn't name an item; the subject is "
     "\"spyglass\"."),
    ("I want to brew potions. How do I craft a brewing stand?",
     "recipe", "minecraft:brewing_stand",
     "The goal (brewing potions) doesn't name the item; the subject is "
     "\"brewing stand\"."),
]


def _trace(question, think, tool_call, result, answer) -> Dict:
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"<think>{think}</think>\n<tool_call>{tool_call}</tool_call>"},
        {"role": "user", "content": f"<result>{', '.join(result) if result else '(no rows)'}</result>"},
        {"role": "assistant", "content": answer},
    ]}


def generate(db: OracleDB, rng: random.Random, recipe_limit: int = 1200, loot_limit: int = 500,
             tag_limit: int = 150, max_tag_members: int = 12, large_tag_cap: int = 40,
             loot_source_limit: int = 400) -> Iterator[Dict]:
    for animal in [r[0] for r in db.con.execute("SELECT DISTINCT animal FROM breeding_food")]:
        foods = db.breeding_foods(animal)
        if foods:
            q = rng.choice(_PHRASINGS["breeding_food"]).format(n=animal)
            yield _trace(q, f"This asks what {animal}s eat; query the animal-food table.",
                         f"lookup(breeding_food, animal='{animal}')", foods,
                         f"{animal.capitalize()}s eat: {_english(foods)}.")
    # Skip variant recipe-file ids ("<item>_from_<source>_stonecutting", etc.):
    # those are the recipe FILENAME, not the crafted item, and make nonsense
    # questions. The real item (e.g. minecraft:smooth_quartz_slab) is a separate
    # row and is kept. (build_db now keys by the extracted result_item, so fresh
    # DBs won't have these; this also cleans the currently-built DB.)
    recipe_items = [r[0] for r in db.con.execute("SELECT DISTINCT result_item FROM recipe")]
    recipe_items = [x for x in recipe_items if "_from_" not in x][:recipe_limit]
    for item in recipe_items:
        ings = db.recipe_ingredient_groups(item)
        if ings:
            q = rng.choice(_PHRASINGS["recipe"]).format(n=_bare(item))
            yield _trace(q, f"A recipe question for {_bare(item)}; query the recipe table.",
                         f"lookup(recipe, result_item='{item}')", ings,
                         f"To craft {_bare(item)} you need: {_english_recipe(ings)}.")
    for name, ml in db.con.execute("SELECT name, max_level FROM enchantment WHERE max_level IS NOT NULL"):
        q = rng.choice(_PHRASINGS["enchantment"]).format(n=_bare(name))
        yield _trace(q, f"Query the enchantment table for {_bare(name)}'s max level.",
                     f"lookup(enchantment, name='{name}')", [f"max_level={ml}"],
                     f"{_bare(name).capitalize()} has a maximum level of {ml}.")
    for (block,) in db.con.execute("SELECT DISTINCT block FROM ore_depth"):
        rows = db.ore_depths(block)
        if not rows:
            continue
        n = _bare(block)
        q = rng.choice(_PHRASINGS["ore_depth"]).format(n=n)
        yield _trace(q, f"A Y-level question for {n}; query the ore_depth table.",
                     f"lookup(ore_depth, block='{block}')",
                     [_depth_row_str(r) for r in rows],
                     f"{n.capitalize()} generates {_english_depth(rows)}.")
    # Stratified loot sampling: a plain LIMIT grabbed the first N sources in
    # insertion order, which were ALL minecraft:blocks/* -- the model never saw a
    # single entity/mob loot table (round-3 probe: "if I kill a creeper" failed).
    # Include every non-block source (entities/chests/gameplay/shearing/...) and
    # fill the remaining budget with a shuffled sample of block sources.
    all_src = [r[0] for r in db.con.execute("SELECT DISTINCT source FROM loot")]
    non_block = [s for s in all_src if not s.startswith("minecraft:blocks/")]
    block = [s for s in all_src if s.startswith("minecraft:blocks/")]
    rng.shuffle(block)
    loot_srcs = non_block + block[: max(0, loot_limit - len(non_block))]
    for src in loot_srcs:
        drops = db.loot_drops(src)
        if not drops:
            continue
        label = src.split("/")[-1].replace("_", " ")
        # Mobs (entities/) get several phrasings and an explicit "don't guess"
        # think step: the base model has strong priors for common mobs ("zombie
        # head", "creeper head") that override the tool unless routing is
        # reinforced (round-4 probe: creeper/zombie failed while skeleton/spider/
        # enderman passed). Blocks/other sources get one phrasing.
        is_mob = src.startswith("minecraft:entities/")
        n_variants = 3 if is_mob else 1
        think = (f"A drops question for the {label}. Do not answer from memory -- "
                 f"query the loot table and use the returned result." if is_mob
                 else f"A drops question for {label}; query the loot table.")
        for phr in rng.sample(_PHRASINGS["loot"], min(n_variants, len(_PHRASINGS["loot"]))):
            q = phr.format(n=label)
            yield _trace(q, think, f"lookup(loot, source='{src}')", drops,
                         f"{label.capitalize()} drops: {_english(drops)}.")
    # Reverse-loot ("where do I find X") -- only items with a real mine/kill
    # source (see OracleDB.loot_sources); a sampled subset, since ~900 distinct
    # items qualify and this category doesn't need that much corpus weight.
    src_items = [r[0] for r in db.con.execute("SELECT DISTINCT drop_item FROM loot")]
    rng.shuffle(src_items)
    emitted_sources = 0
    for drop_item in src_items:
        if emitted_sources >= loot_source_limit:
            break
        srcs = db.loot_sources(drop_item)
        if not srcs:
            continue
        # Items with a real minable-ore source get the depth-chained treatment
        # instead (gen_multihop_traces.generate_source_depth) -- skipped here
        # so the same item doesn't teach two different answer shapes (a shallow
        # "mine X" and a depth-enriched one) for the same question.
        if any(db.ore_depths(f"minecraft:{s[len('minecraft:blocks/'):]}")
               for s in srcs if s.startswith("minecraft:blocks/")):
            continue
        n = _bare(drop_item)
        q = rng.choice(_PHRASINGS["loot_source"]).format(n=n)
        yield _trace(q, f"A where-to-find question for {n}; query the loot table in reverse "
                     f"(which sources drop it).",
                     f"lookup(loot_source, drop_item='{drop_item}')", srcs,
                     f"You can get {n} by {_english_sources(srcs)}.")
        emitted_sources += 1
    for (trade,) in db.con.execute("SELECT DISTINCT trade FROM villager_trade"):
        t = db.trade(trade)
        if t:
            wi, wc, gi, gc = t
            wc, gc = int(wc or 1), int(gc or 1)
            q = rng.choice(_PHRASINGS["villager_trade"]).format(n=trade)
            yield _trace(q, f"A villager trade question; query villager_trade for {trade}.",
                         f"lookup(villager_trade, trade='{trade}')",
                         [f"wants={wc} {wi}", f"gives={gc} {gi}"],
                         f"The villager wants {wc} {_bare(wi)} and gives {gc} {_bare(gi)}.")
    for song, length in db.con.execute("SELECT song, length_seconds FROM jukebox_song WHERE length_seconds IS NOT NULL"):
        q = rng.choice(_PHRASINGS["jukebox_song"]).format(n=_bare(song))
        yield _trace(q, f"Query jukebox_song for the length of {_bare(song)}.",
                     f"lookup(jukebox_song, song='{song}')", [f"length={length}"],
                     f"The {_bare(song)} music disc is {length} seconds long.")
    for painting, w, h in db.con.execute("SELECT painting, width, height FROM painting"):
        q = rng.choice(_PHRASINGS["painting"]).format(n=_bare(painting))
        yield _trace(q, f"Query the painting table for {_bare(painting)}'s dimensions.",
                     f"lookup(painting, painting='{painting}')", [f"width={w}", f"height={h}"],
                     f"The {_bare(painting)} painting is {w} blocks wide and {h} blocks tall.")
    tag_rows = db.con.execute(
        "SELECT tag, COUNT(*) c FROM tag GROUP BY tag HAVING c BETWEEN 2 AND ? LIMIT ?",
        (max_tag_members, tag_limit)).fetchall()
    for tag, _c in tag_rows:
        members = db.tag_members(tag)
        q = rng.choice(_PHRASINGS["tag"]).format(n=tag)
        yield _trace(q, f"A tag membership question; query the tag table for {tag}.",
                     f"lookup(tag, tag='{tag}')", members,
                     f"The tag {tag} contains: {_english(members)}.")
    # LARGE tags (>max_tag_members, up to a hard cap the tool can still return in
    # one result): enumerating everything doesn't scale as an answer, so the
    # shape is COUNT + a sample, not a full list. Above the hard cap the tool
    # itself would need to paginate/truncate results -- out of scope here (a
    # documented residual limitation), so those tags are skipped entirely.
    large_tag_rows = db.con.execute(
        "SELECT tag, COUNT(*) c FROM tag GROUP BY tag HAVING c > ? AND c <= ?",
        (max_tag_members, large_tag_cap)).fetchall()
    for tag, c in large_tag_rows:
        members = db.tag_members(tag)
        sample = members[:5]
        q = rng.choice(_PHRASINGS["tag_large"]).format(n=tag)
        yield _trace(q, f"A large-tag question ({c} members); query the tag table "
                     f"for {tag} and summarize with a count and a few examples "
                     f"rather than listing all {c}.",
                     f"lookup(tag, tag='{tag}')", members,
                     f"The {tag} tag has {c} items, including {_english(sample)}.")
    # item_use: usage/purpose questions for the ~180 real items with component
    # data (food/durability/attack-damage/equip-slot). This is what fixes the
    # round-10 fabrication for real ("carrot on a stick" HAS a max_damage row).
    for item, in db.con.execute("SELECT DISTINCT item FROM item_use"):
        facts = _item_use_facts(db.item_use(item))
        if not facts:
            continue
        n = _bare(item)
        q = rng.choice(_PHRASINGS["item_use"]).format(n=n, art=_article(n))
        yield _trace(q, f"A usage question for {n}; query item_use for {item}.",
                     f"lookup(item_use, item='{item}')", facts,
                     _item_use_answer(n, facts))
    # EMPTY-RESULT / decline traces: query a fake subject, get (no rows), and
    # DECLINE instead of fabricating. Each fake is checked against the DB so a
    # decline trace is only emitted when the subject truly has no row.
    for fake in _FAKE["recipe"]:
        if db.recipe_ingredients(fake):
            continue
        n = _bare(fake)
        q = rng.choice(_PHRASINGS["recipe"]).format(n=n)
        yield _trace(q, f"A recipe question for {n}; query the recipe table.",
                     f"lookup(recipe, result_item='{fake}')", [],
                     f"I don't have any data on {n} - it isn't in the recipe table, "
                     f"so it doesn't exist as a craftable item in Minecraft.")
    for fake in _FAKE["breeding_food"]:
        if db.breeding_foods(fake):
            continue
        q = rng.choice(_PHRASINGS["breeding_food"]).format(n=fake)
        yield _trace(q, f"This asks what a {fake} eats; query the animal-food table.",
                     f"lookup(breeding_food, animal='{fake}')", [],
                     f"I don't have any data on the {fake} - it isn't in the animal-food "
                     f"table and isn't a real Minecraft animal.")
    for fake in _FAKE["enchantment"]:
        if db.enchantment_max_level(fake) is not None:
            continue
        n = _bare(fake)
        q = rng.choice(_PHRASINGS["enchantment"]).format(n=n)
        yield _trace(q, f"Query the enchantment table for {n}'s max level.",
                     f"lookup(enchantment, name='{fake}')", [],
                     f"I don't have any data on {n} - it isn't in the enchantment table, "
                     f"so it's not a real Minecraft enchantment.")
    for fake in _FAKE["loot"]:
        if db.loot_drops(fake):
            continue
        label = fake.split("/")[-1].replace("_", " ")
        q = rng.choice(_PHRASINGS["loot"]).format(n=label)
        yield _trace(q, f"A drops question for {label}; query the loot table.",
                     f"lookup(loot, source='{fake}')", [],
                     f"I don't have any data on the {label} - it isn't in the loot table "
                     f"and isn't a real Minecraft entity.")
    for fake in _FAKE["loot_source"]:
        if db.loot_sources(fake):
            continue
        n = _bare(fake)
        q = rng.choice(_PHRASINGS["loot_source"]).format(n=n)
        yield _trace(q, f"A where-to-find question for {n}; query the loot table in reverse.",
                     f"lookup(loot_source, drop_item='{fake}')", [],
                     f"I don't have any data on where to find {n} - it isn't a real Minecraft "
                     f"item, or nothing in this world drops it.")
    for fake in _FAKE["ore_depth"]:
        if db.ore_depths(fake):
            continue
        n = _bare(fake)
        q = rng.choice(_PHRASINGS["ore_depth"]).format(n=n)
        yield _trace(q, f"A Y-level question for {n}; query the ore_depth table.",
                     f"lookup(ore_depth, block='{fake}')", [],
                     f"I don't have any data on where {n} generates - it isn't a real "
                     f"Minecraft ore block.")
    # Key-normalization traces: teach natural wording -> canonical key.
    for question, table, key in _NORMALIZE_CASES:
        if table == "recipe":
            rows = db.recipe_ingredient_groups(key)
            if not rows:
                continue
            n = _bare(key)
            yield _trace(question, f"The question is phrased loosely; the item is {n}. "
                         f"Query recipe for {key}.",
                         f"lookup(recipe, result_item='{key}')", rows,
                         f"To craft {n} you need: {_english_recipe(rows)}.")
        elif table == "breeding_food":
            rows = db.breeding_foods(key)
            if not rows:
                continue
            yield _trace(question, f"This asks what {key}s eat. Query the animal-food "
                         f"table for {key}.",
                         f"lookup(breeding_food, animal='{key}')", rows,
                         f"{key.capitalize()}s eat: {_english(rows)}.")
        elif table == "loot":
            rows = db.loot_drops(key)
            if not rows:
                continue
            mob = key.split("/")[-1].replace("_", " ")
            yield _trace(question, f"A mob-drop question about the {mob}; the loot source is "
                         f"{key} (mobs live under entities/). Query loot for {key}.",
                         f"lookup(loot, source='{key}')", rows,
                         f"{mob.capitalize()} drops: {_english(rows)}.")
        elif table == "loot_source":
            rows = db.loot_sources(key)
            if not rows:
                continue
            n = _bare(key)
            yield _trace(question, f"This asks where to get {n}, not what it's used for or how "
                         f"to craft it. Query the loot table in reverse for {key}.",
                         f"lookup(loot_source, drop_item='{key}')", rows,
                         f"You can get {n} by {_english_sources(rows)}.")
    # Implicit-goal traces: teach ignoring goal-clause vocabulary and finding the
    # named subject in the second clause.
    for question, table, key, think in _IMPLICIT_GOAL_CASES:
        rows = db.recipe_ingredient_groups(key)
        if not rows:
            continue
        n = _bare(key)
        yield _trace(question, f"{think} Query recipe for {key}.",
                     f"lookup(recipe, result_item='{key}')", rows,
                     f"To craft {n} you need: {_english_recipe(rows)}.")


_CALL_RE = re.compile(r"lookup\((\w+),\s*\w+='(.*?)'\)")
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)


def verify_trace(trace: Dict, db: OracleDB) -> bool:
    """Re-execute the trace's query and confirm (1) the stated <result> equals
    the DB's real return and (2) the final answer covers exactly those rows."""
    msgs = trace["messages"]
    call = _CALL_RE.search(msgs[1]["content"])
    if not call:
        return False
    stated = set(_RESULT_RE.search(msgs[2]["content"]).group(1).split(", "))
    answer = msgs[3]["content"].lower()
    table, key = call.group(1), call.group(2)
    if table == "breeding_food":
        rows = db.breeding_foods(key)
    elif table == "recipe":
        rows = db.recipe_ingredient_groups(key)
    elif table == "loot":
        rows = db.loot_drops(key)
    elif table == "loot_source":
        rows = db.loot_sources(key)
    elif table == "ore_depth":
        rows = [_depth_row_str(r) for r in db.ore_depths(key)]
    elif table == "tag":
        rows = db.tag_members(key)
    elif table == "enchantment":
        ml = db.enchantment_max_level(key)
        rows = [f"max_level={ml}"] if ml is not None else []
    elif table == "villager_trade":
        t = db.trade(key)
        rows = [f"wants={int(t[1] or 1)} {t[0]}", f"gives={int(t[3] or 1)} {t[2]}"] if t else []
    elif table == "jukebox_song":
        length = db.jukebox_length(key)
        rows = [f"length={length}"] if length is not None else []
    elif table == "painting":
        wh = db.painting_size(key)
        rows = [f"width={wh[0]}", f"height={wh[1]}"] if wh else []
    elif table == "item_use":
        u = db.item_use(key)
        rows = _item_use_facts(u) if u else []
    else:
        return False
    if not rows:
        # decline trace: the DB truly has no row, so the stated result must be
        # (no rows) and the answer must DECLINE rather than assert a fact.
        return stated == {"(no rows)"} and any(m in answer for m in _DECLINE_MARKERS)
    if stated != set(rows):
        return False
    # numeric/scalar categories: the value(s) must appear in the answer text
    if table in ("enchantment", "jukebox_song", "painting", "villager_trade", "item_use"):
        return bool(rows) and all(
            any(tok in answer for tok in r.split("=")[-1].replace("minecraft:", "").replace("_", " ").split())
            for r in rows)
    # Large tags: the answer summarizes with a count + sample, not a full
    # enumeration (unlistable at 13-40+ members). Require the correct count is
    # stated and that at least a few cited items are real members -- not that
    # every member appears (that would be the small-tag rule and would fail
    # here by construction).
    if table == "tag" and len(rows) > 12:
        cited = sum(_bare(r) in answer for r in rows)
        return str(len(rows)) in answer and cited >= 3
    # loot_source rows are "minecraft:blocks/iron_ore"-shaped -- _bare only
    # strips the namespace, leaving "blocks/iron ore"; use the path-aware label
    # (matches _english_sources) instead of the generic bare-name check.
    if table == "loot_source":
        return all(_source_label(r) in answer for r in rows)
    # ore_depth rows are "Y80to319(trapezoid,count90)"-shaped tokens, not
    # player-facing text; check the PRIMARY (highest-count, first-reported)
    # band's real min/max Y numbers appear, matching _english_depth's guarantee
    # that the primary band is always the first one named.
    if table == "ore_depth":
        raw = db.ore_depths(key)
        return bool(raw) and str(raw[0][2]) in answer and str(raw[0][3]) in answer
    # "/"-split is a no-op for every non-recipe category (their rows never
    # contain "/"); for recipe it checks every OR-alternative in every AND-slot
    # is named, matching _english_recipe (which always lists every alternative).
    return all(_bare(sub) in answer for r in rows for sub in r.split("/"))


def generate_usage_declines(db: OracleDB, rng: random.Random) -> Iterator[Dict]:
    """No-tool decline traces for USAGE/FUNCTION questions ("what is X used
    for?") on items with NO item_use row -- for those there is genuinely
    nothing to query, so the correct move is declining WITHOUT calling any
    tool. Checked against the LIVE db (not a fixed assumption): an item that
    gains item_use data in a future game version is skipped here automatically
    and picked up by the item_use generator in generate() instead. Distinct
    from the round-3 fake-subject declines (which DO call a tool and get empty
    rows) -- these are real items, just a fact type the oracle doesn't have
    data for."""
    for item in _USAGE_ITEMS:
        key = f"minecraft:{item.replace(' ', '_')}"
        if db.item_use(key):
            continue  # has real data now -- generate() handles this item instead
        art = _article(item)
        q = rng.choice(_PHRASINGS["item_use"]).format(n=item, art=art)
        answer = (f"I don't have information on what {art} {item} is used for -- I can "
                  f"only look up crafting recipes, breeding food, mob drops, "
                  f"enchantment levels, villager trades, tag membership, jukebox "
                  f"song lengths, painting sizes, and item stats (food value, "
                  f"durability, attack damage, equip slot).")
        yield {"messages": [
            {"role": "user", "content": q},
            {"role": "assistant",
             "content": f"<think>This asks what {art} {item} is used for/does -- that's "
                        f"purpose/mechanics with no data in item_use or any other "
                        f"table. No lookup would help; I should decline "
                        f"directly.</think>\n{answer}"},
        ]}


def verify_usage_decline(trace: Dict) -> bool:
    """A usage-decline trace must call NO tool and must decline."""
    msgs = trace["messages"]
    if len(msgs) != 2:
        return False
    if "<tool_call>" in msgs[1]["content"]:
        return False
    answer = msgs[1]["content"].lower()
    return any(m in answer for m in _DECLINE_MARKERS)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: gen_query_traces.py <db_path> [out.jsonl] [--seed N]")
    seed = 0
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    db = OracleDB(sys.argv[1])
    rng = random.Random(seed)
    traces = list(generate(db, rng))
    bad = [t for t in traces if not verify_trace(t, db)]
    if bad:
        sys.exit(f"ERROR: {len(bad)} traces failed self-verification")
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    if out:
        with open(out, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        print(f"wrote {len(traces)} verified multi-turn query-traces to {out}")
    print(f"self-verified {len(traces)}/{len(traces)} traces")


if __name__ == "__main__":
    main()
