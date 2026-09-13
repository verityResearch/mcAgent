#!/usr/bin/env python3
"""Cheap, zero-live-execution intermediate signal before building the full
live model-driven action-execution eval harness the review asked for: for each of the SAME 15 genuinely
held-out DAG tasks prepare_action_corpus.py carved out, walk the REAL
recorded (gold) trajectory turn by turn, teacher-forced -- at each
assistant-turn position, feed the model the true preceding context (the
real task prompt plus every real prior tool result, exactly as recorded
during live generation) and check whether the model's own generated tool
call matches the gold one.

FIRST VERSION of this script (free-generation from just the task prompt,
no teacher forcing) found something real but not yet meaningful: ALL 15
held-out tasks produced the IDENTICAL first call
(action(goto_block, block='crafting_table', ...)) -- because
craftChain() in gen_action_traces.js ALWAYS emits that exact call first,
unconditionally, for every DAG trace in the training corpus. The model
correctly learned that fixed procedural template, which is a real
positive signal (it learned action() syntax and reproduced a consistent
pattern), but doesn't distinguish "genuine item-specific planning" from
"imitating a template every training trace shares." This teacher-forced,
multi-turn version tests the part that actually varies per item -- which
CRAFT calls happen next, which requires distinguishing between the 15
different held-out targets' real prerequisite chains.

Teacher forcing (not live re-execution) means this still doesn't satisfy
the actual ask -- it tests whether the model can predict the right
NEXT step given TRUE prior context, not whether it can drive a full task
to completion against a live, uncertain world (gate 2). It's a real,
cheap, honest step toward that, not a replacement for it.

    python3 probe_action_first_hop.py \
        --model Qwen/Qwen3-1.7B --adapter adapter-tool-v12-action-cuda \
        --action-traces mc_bot_action_traces.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Same syntax action_tools.js's own header comment describes: "action(tool,
# k='v', k2=42) -- mirrors lookup(table, key='...')'s syntax."
_ACTION_CALL_RE = re.compile(r"action\(\s*(\w+)\s*(?:,\s*(.*?))?\)")
_ITEM_ARG_RE = re.compile(r"item\s*=\s*'([^']*)'")
KNOWN_TIER1_TOOLS = {"goto", "goto_block", "follow", "equip", "drop", "eat", "sleep", "craft", "smelt"}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def held_out_dag_tasks(traces, heldout_per_bucket=3):
    """Mirrors prepare_action_corpus.py's split EXACTLY -- same stratification,
    same ordering (first N of each bucket in file order), so this probes the
    real held-out set the trained model was never shown."""
    dag = [t for t in traces if t["meta"]["source"] == "dag"]
    by_bucket = defaultdict(list)
    for t in dag:
        by_bucket[t["meta"]["depthBucket"]].append(t)
    heldout = []
    for bucket in sorted(by_bucket):
        items = by_bucket[bucket]
        if len(items) <= 1:
            continue
        n = min(heldout_per_bucket, len(items) - 1)
        heldout.extend(items[:n])
    return heldout


def parse_call(text):
    m = _ACTION_CALL_RE.search(text)
    if not m:
        return None
    tool, raw_args = m.group(1), (m.group(2) or "")
    item_m = _ITEM_ARG_RE.search(raw_args)
    return {"tool": tool, "raw_args": raw_args, "item": item_m.group(1) if item_m else None}


def generate_turn(model, tok, convo, max_tokens):
    prompt = tok.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=0.2, top_p=0.9, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--action-traces", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    traces = load_jsonl(args.action_traces)
    heldout = held_out_dag_tasks(traces)
    print(f"{len(heldout)} held-out DAG tasks (should match prepare_action_corpus.py's 15)")

    print(f"loading {args.model} + adapter {args.adapter}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    total_turns = 0
    matched_turns = 0
    total_craft_turns = 0
    matched_craft_turns = 0

    for t in heldout:
        msgs = t["messages"]
        task_text = msgs[0]["content"]
        print(f"\n=== [bucket {t['meta']['depthBucket']}] {task_text} ===")
        convo = [msgs[0]]
        for i in range(1, len(msgs)):
            if msgs[i]["role"] != "assistant":
                convo.append(msgs[i])  # real, gold tool-result turn -- teacher forcing
                continue

            gold_call = parse_call(msgs[i]["content"])
            step = generate_turn(model, tok, convo, args.max_tokens)
            pred_call = parse_call(step)

            total_turns += 1
            if gold_call is None:
                # gold turn is the final answer, not a tool call -- nothing to
                # compare structurally; just note whether the model also
                # stopped calling tools here.
                stopped = pred_call is None
                print(f"  turn {i}: GOLD=final-answer, model-also-stopped={stopped}")
                if stopped:
                    matched_turns += 1
            elif pred_call is None:
                print(f"  turn {i}: GOLD action({gold_call['tool']}, {gold_call['raw_args']}) "
                      f"-> MODEL produced no tool call")
            else:
                tool_match = pred_call["tool"] == gold_call["tool"]
                item_match = pred_call["item"] == gold_call["item"]
                match = tool_match and (gold_call["item"] is None or item_match)
                if gold_call["tool"] == "craft":
                    total_craft_turns += 1
                    if match:
                        matched_craft_turns += 1
                if match:
                    matched_turns += 1
                print(f"  turn {i}: GOLD action({gold_call['tool']}, {gold_call['raw_args']}) "
                      f"-> MODEL action({pred_call['tool']}, {pred_call['raw_args']}) "
                      f"[{'MATCH' if match else 'DIFFERS'}]")

            # Teacher-force with the REAL recorded assistant turn regardless of
            # what the model just predicted -- every subsequent comparison
            # still sees the true trajectory, not the model's own (possibly
            # wrong) continuation compounding errors.
            convo.append(msgs[i])

    print(f"\n=== SUMMARY ===")
    print(f"turns compared: {total_turns}, matched (tool [+item where gold specified one]): "
          f"{matched_turns}/{total_turns}")
    print(f"craft-specifically: {matched_craft_turns}/{total_craft_turns} "
          f"(this is the real per-item planning signal -- goto_block-first turns are "
          f"excluded since every gold trace shares that identical first step)")
    print("NOTE: teacher-forced against REAL recorded ground truth, not live re-execution -- "
          "a real, cheap signal, not a substitute for the reviewer's actual gate-2 outcome-verified ask.")


if __name__ == "__main__":
    main()
