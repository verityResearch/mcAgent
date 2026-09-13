"""Run the same held-out prompts through base vs. fine-tuned Mellum2 and save both sets of answers side by side, using the exact teacher system prompts fact_pipeline used to generate training data (so this tests whether the fine-tune improved on the task it was actually trained for).
"""
import json
import sys
import urllib.request

EVAL_SET_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
BASE_URL = sys.argv[3]  # e.g. http://127.0.0.1:5820/v1
LABEL = sys.argv[4]  # "base" or "finetuned"

FACT_SEEDED_SYSTEM = (
    "You write Minecraft trivia training samples. You are given a GROUND-TRUTH fact and a "
    "question about it. Put that question itself into 'instruction' -- verbatim, or lightly "
    "rephrased into a natural standalone question, but it must still ask specifically about "
    "this fact's subject, not describe the exercise you're doing. Write a natural-language "
    "reasoning trace and answer that use ONLY the given fact -- never invent details. Also "
    "restate the fact's fields exactly as given, under 'structured_claim', so it can be "
    "mechanically checked. "
    'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str, '
    '"structured_claim": object}.'
)

FREE_RECALL_SYSTEM = (
    "You write Minecraft strategy training samples. Given a topic, write a natural-language "
    "question, a reasoning trace, and an answer, drawing on general Minecraft knowledge.\n"
    "Before you write each specific numeric or mechanical claim (a depth, a tool tier, an "
    "enchantment's behavior, a trade ratio, a crafting recipe, and similar details), pause and "
    "honestly ask yourself: do I actually know this precisely, or am I estimating or guessing "
    "and it merely sounds confident? Recalled numbers and mechanics are exactly the kind of "
    "thing that feels certain while being wrong. If there is any real doubt, use the "
    "search_minecraft_wiki tool to check before writing the claim, rather than after. If you "
    "are genuinely certain (e.g. you just verified it, or it's basic/stable game structure), "
    "you do not need to look it up. "
    'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str}.'
)


def call_chat(system, user):
    payload = {
        "model": "eval",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.6,
        "max_tokens": 3072,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"].get("content") or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"_raw": content, "_parse_error": True}


def main():
    eval_set = json.load(open(EVAL_SET_PATH))
    results = []
    for item in eval_set:
        if item["lane"] == "fact_seeded":
            user = f"Fact ({item['category']} {item['subject_id']}): {json.dumps(item['fields'])}\nQuestion: {item['question']}"
            try:
                out = call_chat(FACT_SEEDED_SYSTEM, user)
            except Exception as exc:
                out = {"_error": repr(exc)}
            results.append({"lane": "fact_seeded", "subject_id": item["subject_id"],
                             "question": item["question"], "fields": item["fields"], "response": out})
            print(f"[{LABEL}] fact_seeded {item['subject_id']}: {'OK' if 'answer' in out else out}")
        else:
            user = f"Topic: {item['topic']}"
            try:
                out = call_chat(FREE_RECALL_SYSTEM, user)
            except Exception as exc:
                out = {"_error": repr(exc)}
            results.append({"lane": "free_recall", "topic": item["topic"], "response": out})
            print(f"[{LABEL}] free_recall {item['topic']}: {'OK' if 'answer' in out else out}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
