"""Assemble the tool-augmented training set: query-traces + no-tool negatives.

Two behaviors the model must learn, balanced:
  1. QUERY   -- a factual question -> emit a tool_call, use the returned result
     (multi-turn traces from gen_query_traces.py).
  2. REASON  -- an open strategy question ("how do I survive the first night?")
     -> answer directly, NO tool_call. Without these negatives the model learns
     to always emit a query. Sourced from the round-1 free-recall corpus (each
     already judge-verified prose), reshaped as a plain user->assistant turn
     with the <think> block preserved as Qwen3-native reasoning.

Outputs mlx chat train.jsonl / valid.jsonl. Split is by first-user-message so
near-duplicate phrasings can't leak across the split.

    PYTHONPATH=. python3 tool_oracle/build_training_set.py \
        --db minecraft.db --free-recall <round1_corpus.jsonl> --out-dir data-tool [--val-frac 0.05]
"""
import argparse
import json
import os
import random

from tool_oracle.gen_multihop_traces import generate as generate_multihop
from tool_oracle.gen_multihop_traces import generate_cross, verify_multihop
from tool_oracle.gen_multihop_traces import (
    generate_source_depth, generate_source_depth_normalize, verify_source_depth,
)
from tool_oracle.gen_query_traces import generate, generate_usage_declines, verify_trace, verify_usage_decline
from tool_oracle.lookup import OracleDB

_MARKER = "</thought>\n"


def _no_tool_examples(path):
    """Round-1 free-recall samples -> no-tool reason-directly chat turns."""
    out = []
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        row = json.loads(line)
        if row.get("verification", {}).get("method") != "llm_judge":
            continue
        resp = row.get("response", "")
        if _MARKER not in resp:
            continue
        thought, answer = resp.split(_MARKER, 1)
        thought = thought.replace("<thought>", "").strip()
        answer = answer.strip()
        if not answer:
            continue
        content = f"<think>{thought}</think>\n{answer}"
        out.append({"messages": [
            {"role": "user", "content": row["instruction"].strip()},
            {"role": "assistant", "content": content},
        ]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--free-recall", default=None, help="round-1 corpus jsonl for no-tool negatives")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--multihop", type=int, default=250,
                    help="number of two-hop chain traces to include (0 to disable)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    db = OracleDB(args.db)
    query = list(generate(db, rng))
    bad = [t for t in query if not verify_trace(t, db)]
    if bad:
        raise SystemExit(f"{len(bad)} query-traces failed self-verification")
    multihop = list(generate_multihop(db, rng, limit=args.multihop)) if args.multihop else []
    if args.multihop:
        multihop += list(generate_cross(db, rng))     # breeding_food -> recipe chains
    bad_mh = [t for t in multihop if not verify_multihop(t, db)]
    if bad_mh:
        raise SystemExit(f"{len(bad_mh)} multi-hop traces failed self-verification")
    source_depth = list(generate_source_depth(db, rng)) + list(generate_source_depth_normalize(db))
    bad_sd = [t for t in source_depth if not verify_source_depth(t, db)]
    if bad_sd:
        raise SystemExit(f"{len(bad_sd)} loot_source->ore_depth traces failed self-verification")
    negatives = _no_tool_examples(args.free_recall)
    usage_declines = list(generate_usage_declines(db, rng))
    bad_usage = [t for t in usage_declines if not verify_usage_decline(t)]
    if bad_usage:
        raise SystemExit(f"{len(bad_usage)} usage-decline traces failed self-verification")
    samples = query + multihop + source_depth + negatives + usage_declines
    rng.shuffle(samples)

    # split by first user message to avoid phrasing leakage
    by_key = {}
    for s in samples:
        by_key.setdefault(s["messages"][0]["content"], []).append(s)
    keys = list(by_key)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_frac))
    train = [s for k in keys[n_val:] for s in by_key[k]]
    valid = [s for k in keys[:n_val] for s in by_key[k]]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(os.path.join(args.out_dir, name), "w") as f:
            for s in split:
                f.write(json.dumps(s) + "\n")
    print(f"wrote {len(train)} train + {len(valid)} valid to {args.out_dir}")
    print(f"  query-traces: {len(query)}   multi-hop: {len(multihop)}   "
          f"source->depth: {len(source_depth)}   "
          f"no-tool negatives: {len(negatives)}   usage-declines: {len(usage_declines)}")


if __name__ == "__main__":
    main()
