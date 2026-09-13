"""Like run_eval_compare.py, but uses the real TeacherModel (with wiki tool
access for the free-recall lane, exactly as fact_pipeline uses it in
production) instead of a bare, tool-less chat call.

The original run_eval_compare.py never passed tools/tool_executor to its
chat calls, so it measured pure parametric-memory generation -- not
representative of what the real pipeline gives free-recall generation. This
version closes that gap.

Usage (run from fact_pipeline/, with an SSH tunnel or direct network path to
the target model server):
    PYTHONPATH=. python3 ../training/run_eval_compare_v2.py \
        eval_set.json out.json http://127.0.0.1:PORT/v1 label
"""
import asyncio
import json
import sys

sys.path.insert(0, ".")

from src.generation.teacher import TeacherModel  # noqa: E402

EVAL_SET_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
BASE_URL = sys.argv[3]
LABEL = sys.argv[4]


async def main():
    teacher = TeacherModel(api_key="dummy", model_name="eval", base_url=BASE_URL)
    eval_set = json.load(open(EVAL_SET_PATH))
    results = []
    for item in eval_set:
        if item["lane"] == "fact_seeded":
            try:
                out = await teacher.generate_fact_seeded(
                    {"category": item["category"], "subject_id": item["subject_id"], "fields": item["fields"]},
                    item["question"])
            except Exception as exc:
                out = {"_error": repr(exc)}
            results.append({"lane": "fact_seeded", "subject_id": item["subject_id"],
                             "question": item["question"], "fields": item["fields"], "response": out})
            print(f"[{LABEL}] fact_seeded {item['subject_id']}: {'OK' if 'answer' in out else out}")
        else:
            try:
                out = await teacher.generate_free_recall(item["topic"])
            except Exception as exc:
                out = {"_error": repr(exc)}
            results.append({"lane": "free_recall", "topic": item["topic"], "response": out})
            print(f"[{LABEL}] free_recall {item['topic']}: {'OK' if 'answer' in out else out}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
