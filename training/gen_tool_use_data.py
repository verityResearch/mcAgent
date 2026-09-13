"""Generate synthetic tool-use training rows for the GPT2-XL LoRA fine-tune.

Every scenario is correct by construction: the tool's fabricated result and
the final answer are both derived from the same template parameters, so
there's nothing to fact-check against an external oracle (unlike
fact_pipeline's Minecraft data) -- the "ground truth" is whatever the
template says it is.

Two tools only (read_file, run_command), with a short, fixed prompt preamble
-- chosen specifically so the whole thing fits well inside GPT2-XL's
1024-token hard limit (position embeddings are fixed-size; there's no config
flag to raise this). opencode's real tool schema alone runs ~31k characters
(~7-9k tokens) per request, which is why this is a standalone harness rather
than a real opencode provider.

Each scenario produces TWO training rows, reusing finetune_gpt2xl.py's
existing single-span prompt/completion masking unchanged rather than adding
multi-span masking for one combined sequence:
  1. task -> "TOOL_CALL: name(args)"          (decide to call a tool)
  2. task + injected tool result -> "FINAL: answer"   (use the result)

Usage:
    python3 training/gen_tool_use_data.py --count 500 --seed 7 --output tool_use.jsonl
"""
import argparse
import json
import random

_PREAMBLE = (
    "Tools available:\n"
    "- read_file(path): returns the contents of a file\n"
    "- run_command(cmd): runs a shell command and returns its output\n\n"
)

_FILENAMES = [
    "config.yaml", "settings.json", "utils.py", "main.py", "helpers.py",
    "constants.py", "app.py", "server.py", "models.py", "handlers.py",
]
_DIRS = ["src", "lib", "tests", "scripts", "app", "core"]
_CONFIG_KEYS = ["timeout", "retries", "max_connections", "port", "debug", "workers", "cache_size", "batch_size"]
_FUNC_NAMES = ["parse_input", "load_config", "compute_total", "normalize_path", "validate_token", "merge_records"]


def _row(instruction, answer, reasoning):
    return {"instruction": instruction, "response": f"<thought>\n{reasoning}\n</thought>\n{answer}"}


def _tool_call_rows(task, tool_name, args_str, result, final_answer, reasoning_call, reasoning_final):
    call_instruction = f"{_PREAMBLE}Task: {task}"
    call_answer = f'TOOL_CALL: {tool_name}({args_str})'
    final_instruction = f"{_PREAMBLE}Task: {task}\nTool result: {result}"
    final_answer_line = f"FINAL: {final_answer}"
    return [
        _row(call_instruction, call_answer, reasoning_call),
        _row(final_instruction, final_answer_line, reasoning_final),
    ]


def scenario_config_value(rng):
    filename = rng.choice([f for f in _FILENAMES if f.endswith((".yaml", ".json"))])
    key = rng.choice(_CONFIG_KEYS)
    value = rng.choice([str(rng.randint(1, 200)), "true", "false", f'"{rng.choice(["prod", "dev", "staging"])}"'])
    other_keys = rng.sample([k for k in _CONFIG_KEYS if k != key], k=2)
    lines = [f"{key}: {value}"] + [f"{k}: {rng.randint(1, 100)}" for k in other_keys]
    rng.shuffle(lines)
    result = "\\n".join(lines)
    task = f"What is the value of {key} in {filename}?"
    return _tool_call_rows(
        task, "read_file", f'path="{filename}"', result, f"{key} is set to {value}.",
        f"To find {key}, I need to read {filename}.", f"The file shows {key}: {value}, so that's the answer.")


def scenario_function_description(rng):
    filename = rng.choice([f for f in _FILENAMES if f.endswith(".py")])
    func = rng.choice(_FUNC_NAMES)
    arg = rng.choice(["data", "value", "record", "path", "token"])
    result = f"def {func}({arg}):\\n    return {arg}.strip()"
    task = f"What does the function {func} in {filename} do?"
    final = f"{func} takes {arg} and returns it with whitespace stripped."
    return _tool_call_rows(
        task, "read_file", f'path="{filename}"', result, final,
        f"I need to look at {filename} to see what {func} does.",
        f"The function body shows it strips whitespace from {arg} and returns it.")


def scenario_line_count(rng):
    filename = rng.choice(_FILENAMES)
    count = rng.randint(10, 500)
    task = f"How many lines are in {filename}?"
    result = f"{count} {filename}"
    final = f"{filename} has {count} lines."
    return _tool_call_rows(
        task, "run_command", f'cmd="wc -l {filename}"', result, final,
        f"wc -l counts lines in a file, so I'll run it on {filename}.",
        f"The command output was {count} {filename}, so the file has {count} lines.")


def scenario_test_run(rng):
    passed = rng.randint(1, 40)
    failed = rng.choice([0, 0, 0, rng.randint(1, 3)])
    duration = round(rng.uniform(0.05, 12.0), 2)
    task = "Do the tests pass?"
    if failed == 0:
        result = f"{passed} passed in {duration}s"
        final = f"Yes, all {passed} tests passed in {duration} seconds."
    else:
        result = f"{passed} passed, {failed} failed in {duration}s"
        final = f"No, {failed} test(s) failed ({passed} passed) in {duration} seconds."
    return _tool_call_rows(
        task, "run_command", 'cmd="pytest -q"', result, final,
        "Running the test suite will show whether the tests pass.",
        f"The output reports {result}, which answers whether the tests passed.")


def scenario_list_files(rng):
    directory = rng.choice(_DIRS)
    n = rng.randint(2, 6)
    files = rng.sample(_FILENAMES, k=min(n, len(_FILENAMES)))
    task = f"What files are in the {directory} directory?"
    # Real `ls dir/` run non-interactively (as test_tool_harness.py's
    # subprocess.run does) prints one entry per line, not space-separated --
    # confirmed live: training on a double-space join here produced a
    # correct TOOL_CALL that then garbled on FINAL, because the real tool
    # result the harness fed back (newlines escaped to literal "\n", same
    # convention scenario_config_value already uses below) didn't match
    # anything seen in training.
    result = "\\n".join(files)
    final = f"The {directory} directory contains: {', '.join(files)}."
    return _tool_call_rows(
        task, "run_command", f'cmd="ls {directory}/"', result, final,
        f"Listing {directory}/ will show what files it contains.",
        f"The listing shows: {', '.join(files)}.")


_SCENARIOS = [
    scenario_config_value, scenario_function_description,
    scenario_line_count, scenario_test_run, scenario_list_files,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="number of scenarios (2 rows each)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.count):
        scenario = rng.choice(_SCENARIOS)
        rows.extend(scenario(rng))

    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows ({args.count} scenarios) to {args.output}")


if __name__ == "__main__":
    main()
