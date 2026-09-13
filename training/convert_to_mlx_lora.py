"""Convert fact_pipeline verified samples -> mlx_lm.lora chat format.

Input: one or more fact_pipeline JSONL files, each row
    {"instruction": str, "response": "<thought>...</thought>\n<answer>", "verification": {...}}

Output: a directory with train.jsonl / valid.jsonl in mlx_lm's chat schema:
    {"messages": [{"role":"user","content": instruction},
                  {"role":"assistant","content": "<think>\n...\n</think>\n\n<answer>"}]}

Key methodology choice: the pipeline's `<thought>` block is remapped to Qwen3's
NATIVE `<think>` tag so SFT reinforces the model's built-in thinking mode on
verified Minecraft content, rather than teaching a divergent `<thought>`
convention it would have to learn from scratch. Rows whose response lacks the
`</thought>` delimiter (malformed) are skipped and counted.

Deterministic split (seeded shuffle) so re-runs are reproducible; the split is
by-row, but --split-by-instruction keeps all samples of one instruction on the
same side to avoid train/valid leakage of near-duplicate topics.

Usage:
    python3 training/convert_to_mlx_lora.py \
        --in dataset_per_topic_both_lanes.jsonl \
        --out-dir ~/qwen3-lora/data-mc \
        --val-frac 0.05 --think-tag think
"""
import argparse
import json
import os
import random
import sys

_MARKER = "</thought>\n"


def _registry(data_report):
    """Load the registry existence oracle, or None if unavailable. Imported
    lazily so the converter runs without the fact_pipeline package when no
    --registry-filter is requested."""
    if not data_report:
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fact_pipeline"))
    from src.verification.registry_oracle import load_registry, find_nonexistent_ids
    reg = load_registry(data_report)
    return lambda text: find_nonexistent_ids(text, reg)


def _load(paths, registry_check=None):
    rows = []
    skipped = 0
    fabricated = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                resp = row.get("response", "")
                if _MARKER not in resp:
                    skipped += 1
                    continue
                # Final safety net: drop any sample citing a nonexistent id.
                # The lane oracles apply the registry check to fact-seeded only,
                # so a fabricated id in a free-recall answer can slip past the
                # judge (e.g. "minecraft:village_type"); this catches it before
                # it reaches training.
                if registry_check is not None and registry_check(resp):
                    fabricated += 1
                    continue
                thought, answer = resp.split(_MARKER, 1)
                thought = thought.replace("<thought>", "").strip()
                answer = answer.strip()
                if not answer:
                    skipped += 1
                    continue
                rows.append({
                    "instruction": row["instruction"].strip(),
                    "thought": thought,
                    "answer": answer,
                    "method": row.get("verification", {}).get("method", "?"),
                })
    return rows, skipped, fabricated


def _to_chat(row, think_tag):
    if think_tag:
        content = f"<{think_tag}>\n{row['thought']}\n</{think_tag}>\n\n{row['answer']}"
    else:
        content = row["answer"]
    return {"messages": [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": content},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--think-tag", default="think",
                    help="tag name for the reasoning block; empty string drops the CoT entirely")
    ap.add_argument("--split-by-instruction", action="store_true",
                    help="keep all rows of one instruction on the same split side (no leakage)")
    ap.add_argument("--registry-filter", default=None, metavar="DATA_REPORT_DIR",
                    help="drop any sample citing a minecraft: id not in this report's registry "
                         "(final existence safety net across both lanes)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows, skipped, fabricated = _load(args.inputs, _registry(args.registry_filter))
    if skipped:
        print(f"WARNING: skipped {skipped} malformed row(s) (no '</thought>' or empty answer)")
    if fabricated:
        print(f"FILTERED {fabricated} sample(s) citing a nonexistent minecraft: id")
    rng = random.Random(args.seed)

    if args.split_by_instruction:
        by_instr = {}
        for r in rows:
            by_instr.setdefault(r["instruction"], []).append(r)
        keys = list(by_instr.keys())
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * args.val_frac))
        val_keys = set(keys[:n_val])
        train = [r for k in keys[n_val:] for r in by_instr[k]]
        valid = [r for k in keys[:n_val] for r in by_instr[k]]
    else:
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * args.val_frac))
        valid, train = rows[:n_val], rows[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            for r in split:
                f.write(json.dumps(_to_chat(r, args.think_tag)) + "\n")

    from collections import Counter
    meth = Counter(r["method"] for r in rows)
    print(f"wrote {len(train)} train + {len(valid)} valid to {args.out_dir}")
    print(f"  lane mix: {dict(meth)}  | think-tag: {args.think_tag or '(none/CoT dropped)'}")


if __name__ == "__main__":
    main()
