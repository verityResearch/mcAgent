"""Base-vs-LoRA eval for the GPT2-XL fact_pipeline fine-tune, over eval_set.json.

Prompt format is the load-bearing decision here. finetune_gpt2xl.py trains on
"Question: {instruction}\nAnswer: {answer}" where {instruction} is always the
bare natural-language question -- verified directly against
dataset_gpt2_factseeded_large.jsonl (0/3000 rows leak fact context into
`instruction`). The reasoning trace and the ground-truth fact fields are
never shown to the student model, only to the teacher that generated the
answer. So this is a closed-book memorization test: at eval time the model
must see ONLY the bare question, in the exact trained template, with no fact
context -- unlike TeacherModel.generate_fact_seeded's prompt (used for
generation and for the Mellum2 LoRA evals), which deliberately hands the
model the ground-truth fields so a chat-tuned model can be graded on
faithful restatement. Reusing that prompt shape here would both leak the
answer and put GPT2-XL outside its training distribution (it was never
fine-tuned on a "Fact (...): ...\nQuestion: ..." prefix). Free-recall items
get the same bare-question template for a consistent regression check, but
this run's dataset was 100% fact-seeded (--fact-seeded-ratio 1.0), so the
adapter never saw a free-recall example -- treat that lane as an
out-of-distribution probe, not a pass/fail signal.

Scoring is mechanical, not vibes: tag questions are always phrased against
an actual tag member (topic_bank's archetype always picks members[0]), so
the correct answer is always "yes" -- score is an affirmative-language
match. Every other fact-seeded category (recipe, loot_table, advancement,
worldgen_structure) is scored as the fraction of expected ground-truth
string values found as case-insensitive substrings of the generated answer.

Usage:
    python3 training/eval_gpt2xl.py \
        --eval-set fact_pipeline/eval_set.json \
        --adapter-dir /path/to/gpt2xl_lora_out \
        --output eval_results_gpt2xl.json
"""
import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2-xl"

# Same anti-repetition settings serve_gpt2xl.py adopted after observing base
# GPT2-XL loop on itself under plain greedy decoding -- applied to both base
# and fine-tuned generation here so neither side is disadvantaged by it.
_GEN_KWARGS = dict(do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=3)
_STOP_SEQUENCES = ["\nQuestion:", "\nQ:", "\n\n"]

_FIELDS_BY_CATEGORY = {
    "recipe": "ingredients",
    "loot_table": "drops",
    "advancement": "criteria_ids",
    "worldgen_structure": "biomes",
}


def build_prompt(item):
    question = item["question"] if item["lane"] == "fact_seeded" else item["topic"]
    return f"Question: {question}\nAnswer:"


def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            **_GEN_KWARGS)
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = full_text[len(prompt):].strip()
    for stop in _STOP_SEQUENCES:
        idx = answer.find(stop)
        if idx != -1:
            answer = answer[:idx].strip()
    return answer


def score_fact_seeded(item, answer):
    category = item["category"]
    answer_lower = answer.lower()
    if category == "tag":
        # Archetype always asks about an actual member, so "yes" is correct.
        says_yes = bool(re.search(r"\byes\b", answer_lower))
        says_no = bool(re.search(r"\bno\b", answer_lower))
        return {"kind": "affirmative", "correct": says_yes and not says_no}

    field = _FIELDS_BY_CATEGORY.get(category)
    if field is None or field not in item["fields"]:
        return {"kind": "unscored", "correct": None}
    expected = item["fields"][field]
    if isinstance(expected, str):
        expected = [expected]
    hits = [v for v in expected if str(v).lower() in answer_lower]
    return {
        "kind": "coverage", "correct": len(hits) == len(expected),
        "coverage": len(hits) / len(expected) if expected else None,
        "matched": hits, "missing": [v for v in expected if v not in hits],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    eval_items = json.load(open(args.eval_set))
    print(f"Loaded {len(eval_items)} eval items from {args.eval_set}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda")
    base_model.eval()

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    ft_model.eval()

    results = []
    fact_seeded_scores = {"base": [], "finetuned": []}
    for i, item in enumerate(eval_items):
        prompt = build_prompt(item)
        with ft_model.disable_adapter():
            base_answer = generate(base_model, tokenizer, prompt, args.max_new_tokens)
        ft_answer = generate(ft_model, tokenizer, prompt, args.max_new_tokens)

        row = {"lane": item["lane"], "prompt": prompt, "base_answer": base_answer, "finetuned_answer": ft_answer}
        if item["lane"] == "fact_seeded":
            row["category"] = item["category"]
            row["base_score"] = score_fact_seeded(item, base_answer)
            row["finetuned_score"] = score_fact_seeded(item, ft_answer)
            for side in ("base", "finetuned"):
                correct = row[f"{side}_score"]["correct"]
                if correct is not None:
                    fact_seeded_scores[side].append(correct)
        results.append(row)
        print(f"[{i + 1}/{len(eval_items)}] {item['lane']} "
              f"base_correct={row.get('base_score', {}).get('correct')} "
              f"finetuned_correct={row.get('finetuned_score', {}).get('correct')}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    for side in ("base", "finetuned"):
        scores = fact_seeded_scores[side]
        rate = sum(scores) / len(scores) if scores else float("nan")
        print(f"{side}: {sum(scores)}/{len(scores)} fact_seeded items fully correct ({rate:.0%})")
    print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
