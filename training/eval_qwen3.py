"""Base-vs-LoRA fact eval for the Qwen3-1.7B fine-tune, on an Apple-Silicon Mac (MLX).

Runs the held-out eval set (build_eval_set.py) through base Qwen3-1.7B and the
LoRA-adapted model, scoring each answer MECHANICALLY against ground truth --
no LLM judge. Reports fact-recall accuracy and, via the anchor items, the
before/after on the base model's known fabrications (pig feed, piston recipe).

Scoring is lenient on surface form: ground-truth ids are reduced to their bare
name (minecraft:iron_ingot -> "iron ingot" / "iron_ingot") and matched
case-insensitively as a substring, since a natural-language answer says "iron
ingot", not the namespaced id.

Usage (on an Apple-Silicon Mac):
    python3 eval_qwen3.py --model $HOME/qwen3-lora/base-4bit \
        --eval-set eval_qwen3.json --adapter $HOME/qwen3-lora/adapter-mc
    # omit --adapter to score the base model only
"""
import argparse
import json
import re

import mlx.core as mx  # noqa: F401  (import validates the MLX runtime is present)
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def _bare(value):
    """minecraft:foo/bar -> 'bar'; also the underscore->space variant."""
    s = str(value)
    s = re.sub(r"^#?minecraft:", "", s)
    s = s.split("/")[-1]
    return s.lower()


def _variants(value):
    b = _bare(value)
    return {b, b.replace("_", " ")}


def _answer_of(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def score(item, answer):
    a = answer.lower()
    kind = item["score_kind"]
    exp = item["expected"]
    if kind == "int_equals":
        return str(int(exp)) in re.findall(r"\d+", a)
    exp_list = exp if isinstance(exp, list) else [exp]
    hits = [e for e in exp_list if any(v in a for v in _variants(e))]
    if kind == "contains_any":
        return len(hits) >= 1
    return len(hits) == len(exp_list)  # contains_all / contains_all_pair


def _build_prompt(tok, question, thinking):
    msgs = [{"role": "user", "content": question}]
    # Fact-recall eval defaults to thinking OFF: we want the factual answer, not
    # a reasoning trace that overflows the token budget before reaching one.
    # enable_thinking is Qwen3-specific; fall back gracefully if unsupported.
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                       enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def run_model(model, tok, items, max_new_tokens, thinking):
    sampler = make_sampler(temp=0.2)
    results = []
    for it in items:
        prompt = _build_prompt(tok, it["question"], thinking)
        out = generate(model, tok, prompt=prompt, max_tokens=max_new_tokens, sampler=sampler, verbose=False)
        ans = _answer_of(out)
        results.append({**it, "answer": ans, "correct": score(it, ans)})
    return results


def summarize(label, results):
    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["category"]][0] += int(bool(r["correct"]))
        by_cat[r["category"]][1] += 1
    total_c = sum(int(bool(r["correct"])) for r in results)
    print(f"\n=== {label}: {total_c}/{len(results)} correct ({100 * total_c // max(len(results),1)}%) ===")
    for cat, (c, n) in sorted(by_cat.items()):
        print(f"  {cat:16} {c}/{n}")
    # always surface the anchors verbatim
    for r in results:
        if r["category"] == "anchor":
            print(f"  ANCHOR [{'OK ' if r['correct'] else 'BAD'}] {r['question'][:45]} -> {r['answer'][:70]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 thinking mode (default off; needs a much larger token budget)")
    args = ap.parse_args()

    items = json.load(open(args.eval_set))
    print(f"loaded {len(items)} eval items (thinking={'on' if args.thinking else 'off'})")

    print("loading base model...")
    base, tok = load(args.model)
    base_res = run_model(base, tok, items, args.max_new_tokens, args.thinking)
    summarize("BASE", base_res)

    ft_res = None
    if args.adapter:
        print("\nloading LoRA-adapted model...")
        ft, tok2 = load(args.model, adapter_path=args.adapter)
        ft_res = run_model(ft, tok2, items, args.max_new_tokens, args.thinking)
        summarize("FINE-TUNED", ft_res)

    if args.out:
        json.dump({"base": base_res, "finetuned": ft_res}, open(args.out, "w"), indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
