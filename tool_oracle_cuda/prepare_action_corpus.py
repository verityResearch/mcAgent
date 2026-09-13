#!/usr/bin/env python3
"""Stage 5, the reviewer's step 3: "train and
evaluate" -- prepares round 12's training data by combining round 11's
existing knowledge corpus (train.jsonl/valid.jsonl, 3084/163 lookup()
samples) with the newly-scaled action-trace corpus
(data/verified/mc_bot_action_traces.jsonl, action() samples).

Trains ONE model on both capabilities from base, rather than continuing
from adapter-tool-v11-cuda, matching round 11's own from-base methodology
and avoiding continuation-training plumbing train_lora_cuda.py doesn't
support. This is the direct test of the reviewer's framing: "can a model, freed
of the KNOWLEDGE burden by a verified oracle, also learn to PLAN
multi-step actions" -- both burdens present in one training run, not
isolated.

A GENUINE held-out split, not a random one: DAG-sourced action traces are
split by TASK ITEM (never by trace content) so held-out tasks are items
the trained model has literally never seen a trace for, same discipline
as stage 0's build_eval_items(). Stratified by depthBucket where
population allows more than one sample in that bucket; bucket populations
with only 1 sample stay entirely in train (holding out the only example
of a depth would leave zero training exposure to it, and the held-out
copy would test nothing the model could have learned). hand_authored/
honest_failure/tier23 traces are fixed seed/regression tasks, not part of
the DAG population -- they stay in train unconditionally, matching how
they're used elsewhere in this project (never treated as eval probes).

    python3 prepare_action_corpus.py \
        --action-traces ../data/verified/mc_bot_action_traces.jsonl \
        --base-train train.jsonl --base-valid valid.jsonl \
        --out-train train_v12_action.jsonl --out-valid valid_v12_action.jsonl
"""
import argparse
import json
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def strip_meta(trace):
    """Output must match round 11's own schema exactly -- {"messages": [...]}
    only, no extra keys -- so this new data isn't distinguishable to the
    trainer/tokenizer in a way the existing pipeline wasn't built for."""
    return {"messages": trace["messages"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-traces", required=True)
    ap.add_argument("--base-train", required=True)
    ap.add_argument("--base-valid", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-valid", required=True)
    ap.add_argument("--heldout-per-bucket", type=int, default=3,
                     help="how many DAG items to hold out per depth bucket "
                          "(skipped for buckets with population <= 1)")
    args = ap.parse_args()

    traces = load_jsonl(args.action_traces)
    dag = [t for t in traces if t["meta"]["source"] == "dag"]
    non_dag = [t for t in traces if t["meta"]["source"] != "dag"]

    by_bucket = defaultdict(list)
    for t in dag:
        by_bucket[t["meta"]["depthBucket"]].append(t)

    action_train, action_heldout = [], []
    heldout_by_bucket = {}
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket]
        if len(items) <= 1:
            action_train.extend(items)
            heldout_by_bucket[bucket] = 0
            continue
        n_heldout = min(args.heldout_per_bucket, len(items) - 1)  # never hold out ALL of a bucket
        action_heldout.extend(items[:n_heldout])
        action_train.extend(items[n_heldout:])
        heldout_by_bucket[bucket] = n_heldout

    base_train = load_jsonl(args.base_train)
    base_valid = load_jsonl(args.base_valid)

    out_train = base_train + [strip_meta(t) for t in (non_dag + action_train)]
    out_valid = base_valid + [strip_meta(t) for t in action_heldout]

    write_jsonl(out_train, args.out_train)
    write_jsonl(out_valid, args.out_valid)

    print(f"base: train={len(base_train)} valid={len(base_valid)}")
    print(f"action corpus: {len(traces)} total ({len(dag)} dag, {len(non_dag)} non-dag)")
    print(f"action held-out by bucket: {heldout_by_bucket}")
    print(f"action train contribution: {len(non_dag) + len(action_train)}, "
          f"held-out (eval-only) contribution: {len(action_heldout)}")
    print(f"wrote {args.out_train} ({len(out_train)} rows), {args.out_valid} ({len(out_valid)} rows)")


if __name__ == "__main__":
    main()
