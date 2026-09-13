"""Build a HELD-OUT fact eval set for measuring the Qwen3-1.7B fine-tune.

Samples facts from the pinned data report across the fabrication-prone
categories, EXCLUDING any subject already referenced in the training corpus,
so the eval measures generalization (did the model learn the domain) rather
than memorization of specific training rows. Each item records the exact
ground truth for mechanical scoring -- no LLM judge in the loop, per the
project's "verify against ground truth, not verdicts" standard.

Also injects a few fixed anchor cases the BASE model is known to fabricate
(pig feed -> hallucinates "corn"; piston recipe -> "sticks and dye") so the
before/after fabrication signal is directly visible.

Usage:
    python3 training/build_eval_set.py \
        --data-report <report_dir> \
        --train-jsonl <training corpus jsonl> \
        --out eval_qwen3.json --n 40
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fact_pipeline"))

from src.generation.fact_sampler import discover_fact_records  # noqa: E402

# category -> (question template, ground-truth field, how to score)
# score kinds: "contains_all" (answer must mention every expected value),
#              "contains_any", "int_equals" (a number must appear).
_SPEC = {
    "recipe": ("What items do you need to craft {sid}? List the ingredients.",
               "ingredients", "contains_all"),
    "enchantment": ("What is the maximum level of the enchantment {sid}?",
                    "max_level", "int_equals"),
    "villager_trade": ("In the villager trade {sid}, what item does the villager want and what do they give?",
                       ("wants_item", "gives_item"), "contains_all_pair"),
    "loot_table": ("What can {sid} drop?", "drops", "contains_any"),
    "jukebox_song": ("How many seconds long is the music disc {sid}?",
                     "length_in_seconds", "int_equals"),
    "painting_variant": ("What are the width and height in blocks of the {sid} painting?",
                         ("width", "height"), "contains_all_pair"),
}

# Fixed anchors: things the base model fabricated in the earlier baseline.
_ANCHORS = [
    {"category": "anchor", "subject_id": "pig_feed",
     "question": "What items can you feed a pig in Minecraft to breed it?",
     "expected": ["carrot", "potato", "beetroot"], "score_kind": "contains_any",
     "note": "base fabricated 'corn' (does not exist)"},
    {"category": "anchor", "subject_id": "piston_recipe",
     "question": "What items do you need to craft a piston in Minecraft?",
     "expected": ["planks", "cobblestone", "iron", "redstone"], "score_kind": "contains_all",
     "note": "base fabricated 'sticks and dye'"},
]


def _training_subjects(train_jsonl):
    subs = set()
    if not train_jsonl or not os.path.exists(train_jsonl):
        return subs
    with open(train_jsonl) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("instruction", "") + " " + json.dumps(row.get("fields", ""))
            subs.update(re.findall(r"minecraft:[a-z0-9_/]+", text))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-report", required=True)
    ap.add_argument("--train-jsonl", default=None, help="training corpus subjects (held out, or -- with "
                                                        "--in-distribution -- the ONLY subjects sampled)")
    ap.add_argument("--in-distribution", action="store_true",
                    help="sample facts the model WAS trained on (recall test) instead of held-out ones "
                         "(generalization test). Anchors are omitted -- they only make sense held-out.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40, help="sampled (non-anchor) items")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    facts = discover_fact_records(args.data_report)
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f["category"], []).append(f)
    trained = _training_subjects(args.train_jsonl)
    mode = "in-distribution (recall)" if args.in_distribution else "held-out (generalization)"
    print(f"discovered {len(facts)} facts; {len(trained)} training subjects; mode = {mode}")

    rng = random.Random(args.seed)
    # anchors are fabrication probes -> only meaningful on the held-out set
    items = [] if args.in_distribution else list(_ANCHORS)
    cats = [c for c in _SPEC if c in by_cat]
    per_cat = max(1, args.n // len(cats))
    for cat in cats:
        template, field, kind = _SPEC[cat]
        if args.in_distribution:
            pool = [f for f in by_cat[cat] if f["subject_id"] in trained]
        else:
            pool = [f for f in by_cat[cat] if f["subject_id"] not in trained]
        rng.shuffle(pool)
        picked = 0
        for f in pool:
            if picked >= per_cat:
                break
            fields = f["fields"]
            if isinstance(field, tuple):
                expected = [fields.get(k) for k in field]
                if any(v is None for v in expected):
                    continue
            else:
                expected = fields.get(field)
                if expected is None:
                    continue
            items.append({
                "category": cat, "subject_id": f["subject_id"],
                "question": template.format(sid=f["subject_id"]),
                "expected": expected, "score_kind": kind,
            })
            picked += 1

    with open(args.out, "w") as f:
        json.dump(items, f, indent=2)
    from collections import Counter
    print(f"wrote {len(items)} eval items to {args.out}")
    print("  by category:", dict(Counter(i["category"] for i in items)))


if __name__ == "__main__":
    main()
