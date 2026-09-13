import argparse
import asyncio
import os
import random
import sys

from dotenv import load_dotenv

from src.generation.converter import convert_to_jsonl
from src.generation.fact_sampler import discover_fact_records, sample_fact_records
from src.generation.orchestrator import PipelineOrchestrator
from src.generation.teacher import TeacherModel
from src.generation.topic_bank import pick_archetype, pick_free_recall_topic
from src.verification.judge import Judge
from src.verification.numeric_oracle import _ench_max_levels
from src.verification.registry_oracle import load_registry, make_layered_fact_check
from src.verification.static_oracle import check_fact_seeded

load_dotenv()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Minecraft Fact Pipeline")
    parser.add_argument("--target", type=int, default=10, help="Target number of verified samples")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency limit")
    parser.add_argument("--api-key", type=str,
                        help="Default API key for both teacher and judge (or set OPENAI_API_KEY env)")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1",
                        help="Default API base URL for both teacher and judge")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Default model name for both teacher and judge")
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Override model name for the teacher only (defaults to --model)")
    parser.add_argument("--teacher-base-url", type=str, default=None,
                        help="Override API base URL for the teacher only (defaults to --base-url)")
    parser.add_argument("--teacher-api-key", type=str, default=None,
                        help="Override API key for the teacher only (defaults to --api-key)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override model name for the judge only (defaults to --model)")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="Override API base URL for the judge only (defaults to --base-url)")
    parser.add_argument("--judge-api-key", type=str, default=None,
                        help="Override API key for the judge only (defaults to --api-key)")
    parser.add_argument("--output", type=str, default="data/verified/dataset.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--data-report-dir", type=str, required=True,
                        help="Path to the extracted pinned data report's root directory "
                             "(the parent of its data/ subdirectory -- extract_pinned_version's "
                             "return value)")
    parser.add_argument("--fact-seeded-ratio", type=float, default=0.8,
                        help="Fraction of each batch drawn from the fact-seeded lane; the rest is free-recall")
    parser.add_argument("--max-empty-batches", type=int, default=5,
                        help="Abort after this many consecutive batches that admit zero samples")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting a non-empty --output file")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for fact/topic sampling")
    return parser.parse_args(argv)


def _next_batch_composition(batch_size, fact_seeded_ratio, fact_seeded_dispatched, total_dispatched):
    """Split ``batch_size`` items between the fact-seeded and free-recall lanes.

    Computing ``round(batch_size * fact_seeded_ratio)`` fresh each batch rounds
    the same way every time for a fixed batch_size -- at concurrency=2 and the
    default ratio 0.8, round(2 * 0.8) == round(1.6) == 2 on every single call,
    so free_recall_count is always 0 and the free-recall lane (all 5 judge
    verification layers) never runs at any low concurrency setting. Tracking a
    running total instead lets one batch's rounding error carry over and
    correct itself in a later batch, so the long-run split still converges on
    ``fact_seeded_ratio`` even though no individual small batch can hit it
    exactly. Returns ``(fact_seeded_count, free_recall_count)``.
    """
    target_fact_seeded = round((total_dispatched + batch_size) * fact_seeded_ratio)
    fact_seeded_count = max(0, min(batch_size, target_fact_seeded - fact_seeded_dispatched))
    return fact_seeded_count, batch_size - fact_seeded_count


async def _run(args, teacher_api_key, judge_api_key):
    rng = random.Random(args.seed)
    teacher = TeacherModel(
        api_key=teacher_api_key,
        model_name=args.teacher_model or args.model,
        base_url=args.teacher_base_url or args.base_url,
    )
    judge = Judge(
        api_key=judge_api_key,
        model_name=args.judge_model or args.model,
        base_url=args.judge_base_url or args.base_url,
    )
    # Layer the registry existence oracle (Layer 0) ahead of the structured
    # fact oracle (Layer 1) when the pinned report ships a registries.json;
    # fall back to the structured oracle alone if it doesn't (older reports).
    registries_path = os.path.join(args.data_report_dir, "reports", "registries.json")
    if os.path.exists(registries_path):
        registry = load_registry(args.data_report_dir)
        fact_check = make_layered_fact_check(registry, check_fact_seeded)
        print(f"Loaded registry existence oracle: {len(registry.all_ids)} real ids (Layer 0 enabled).")
    else:
        fact_check = check_fact_seeded
        print("No reports/registries.json found; running structured fact oracle only (Layer 0 disabled).")
    orchestrator = PipelineOrchestrator(teacher, fact_check, judge, concurrency_limit=args.concurrency)

    print(f"Discovering fact records under {args.data_report_dir}...")
    all_facts = discover_fact_records(args.data_report_dir)
    # Give the judge enchantment max-levels as ground truth for its numeric
    # pre-check (the lookup-free tick<->time arithmetic check runs regardless).
    judge.enchantment_max_levels = _ench_max_levels(
        [f for f in all_facts if f["category"] == "enchantment"])
    if not all_facts:
        sys.exit(f"No fact records discovered under {args.data_report_dir}; check the path and data report layout.")
    print(f"Discovered {len(all_facts)} fact records.")

    admitted_results = []
    consecutive_empty = 0
    total_dispatched = 0
    fact_seeded_dispatched = 0
    while len(admitted_results) < args.target:
        needed = args.target - len(admitted_results)
        batch_size = min(max(needed, 1), args.concurrency)
        fact_seeded_count, free_recall_count = _next_batch_composition(
            batch_size, args.fact_seeded_ratio, fact_seeded_dispatched, total_dispatched)

        # sample_fact_records can return fewer than fact_seeded_count when the
        # fact pool itself is smaller than the request -- track what was
        # actually dispatched, not what was asked for, so the accumulator's
        # running ratio can't drift from reality over a long run.
        facts = sample_fact_records(all_facts, fact_seeded_count, rng)
        fact_seeded_dispatched += len(facts)
        total_dispatched += len(facts) + free_recall_count

        fact_seeded_items = [(fact, pick_archetype(fact["category"], rng)(fact)) for fact in facts]
        free_recall_topics = [pick_free_recall_topic(rng) for _ in range(free_recall_count)]

        results = await orchestrator.run_batch(fact_seeded_items, free_recall_topics)
        orchestrator.log_yield(results)
        newly_admitted = [r for r in results if r["verification"]["success"]]
        admitted_results.extend(newly_admitted)

        consecutive_empty = 0 if newly_admitted else consecutive_empty + 1
        if consecutive_empty >= args.max_empty_batches:
            sys.exit(f"Aborting: {consecutive_empty} consecutive batches admitted 0 samples. "
                     f"Saved {len(admitted_results)}/{args.target}.")

        convert_to_jsonl(admitted_results, args.output)
        print(f"Progress: {len(admitted_results)}/{args.target} verified samples saved to {args.output}")

    print(f"Pipeline complete. {len(admitted_results)} samples verified and saved to {args.output}.")


def main(argv=None):
    args = _parse_args(argv)
    shared_api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    teacher_api_key = args.teacher_api_key or shared_api_key
    judge_api_key = args.judge_api_key or shared_api_key
    if not teacher_api_key:
        sys.exit("Error: API key is required for the teacher via --teacher-api-key, --api-key, "
                 "or OPENAI_API_KEY environment variable.")
    if not judge_api_key:
        sys.exit("Error: API key is required for the judge via --judge-api-key, --api-key, "
                 "or OPENAI_API_KEY environment variable.")
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0 and not args.overwrite:
        sys.exit(f"Error: {args.output} already exists and is non-empty. Pass --overwrite to replace it, "
                 f"or choose a different --output path.")
    asyncio.run(_run(args, teacher_api_key, judge_api_key))


if __name__ == "__main__":
    main()
