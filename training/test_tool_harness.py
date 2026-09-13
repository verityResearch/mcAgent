"""Minimal ReAct-loop test harness for the GPT2-XL tool-use LoRA adapter.

Not a general-purpose agent: run_command only executes a small allowlist of
read-only commands (the same shapes gen_tool_use_data.py trained on) rather
than the model's raw generated string -- a 1.5B LoRA fine-tuned on ~500
synthetic scenarios is not a trusted input source for shell execution.
read_file is restricted to paths inside the given project root (resolved
before the containment check, so ../ path traversal can't escape it). The
point of executing real tools rather than mocking them is to test whether
the learned TOOL_CALL/FINAL protocol generalizes to real results, since
training only ever showed it fabricated ones.

Usage:
    python3 training/test_tool_harness.py \
        --adapter-dir /path/to/gpt2xl_lora_combined \
        --project-root /path/to/repo \
        --task "How many lines are in training/eval_gpt2xl.py?"
"""
import argparse
import re
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2-xl"
_PREAMBLE = (
    "Tools available:\n"
    "- read_file(path): returns the contents of a file\n"
    "- run_command(cmd): runs a shell command and returns its output\n\n"
)
_TOOL_CALL_RE = re.compile(r'^TOOL_CALL:\s*(read_file|run_command)\((?:path|cmd)="([^"]*)"\)\s*$')
_ALLOWED_COMMANDS = [
    re.compile(r'^wc -l [\w./_-]+$'),
    re.compile(r'^ls [\w./_-]+/?$'),
    re.compile(r'^pytest -q$'),
]
# Plain greedy decoding, deliberately with NO repetition_penalty or
# no_repeat_ngram_size: this template is short and structurally repetitive
# by design (TOOL_CALL:/FINAL:, matching parens, etc.), and those anti-loop
# settings -- needed for the much longer, free-form free-recall generations
# elsewhere in this project -- were confirmed live to actively corrupt this
# short-form protocol (e.g. "run_command" -> "run_commands", "main.py" ->
# "main. py") that greedy decoding alone reproduces perfectly.
_GEN_KWARGS = dict(do_sample=False)


def build_prompt(instruction):
    # Must match finetune_gpt2xl.py's SFTDataset exactly: it trains on
    # f"Question: {instruction}\nAnswer:" -- any other wrapping is a prompt
    # shape the model has never seen, regardless of how reasonable it looks.
    return f"Question: {instruction}\nAnswer:"


def generate(model, tokenizer, instruction, max_new_tokens=60):
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, **_GEN_KWARGS)
    text = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    for stop in ("\nQuestion:", "\nTask:", "\nTool result:", "\n\n"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()
    return text


def execute_read_file(project_root, path):
    root = project_root.resolve()
    target = (project_root / path).resolve()
    if target != root and root not in target.parents:
        return None, "refused: path escapes project root"
    if not target.is_file():
        return None, f"refused: {path} does not exist"
    return target.read_text(encoding="utf-8", errors="replace")[:2000], None


def execute_run_command(project_root, cmd):
    if not any(pattern.match(cmd) for pattern in _ALLOWED_COMMANDS):
        return None, f"refused: {cmd!r} is not an allowlisted command"
    result = subprocess.run(cmd, shell=True, cwd=project_root, capture_output=True, text=True, timeout=10)
    return (result.stdout or result.stderr).strip()[:2000], None


def run_task(model, tokenizer, project_root, task, max_steps=3):
    print(f"TASK: {task}")
    instruction = f"{_PREAMBLE}Task: {task}"
    for step in range(max_steps):
        completion = generate(model, tokenizer, instruction)
        print(f"  step {step}: {completion!r}")
        if completion.startswith("FINAL:"):
            return completion[len("FINAL:"):].strip()
        match = _TOOL_CALL_RE.match(completion)
        if not match:
            return f"(harness) model did not emit a valid TOOL_CALL or FINAL: {completion!r}"
        tool_name, arg_value = match.groups()
        if tool_name == "read_file":
            result, error = execute_read_file(project_root, arg_value)
        else:
            result, error = execute_run_command(project_root, arg_value)
        if error:
            print(f"  tool error: {error}")
            return f"(harness) tool call refused: {error}"
        result_text = result.replace("\n", "\\n")
        print(f"  tool result: {result_text[:200]}")
        instruction = f"{_PREAMBLE}Task: {task}\nTool result: {result_text}"
    return "(harness) exceeded max_steps without a FINAL answer"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task", action="append", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    project_root = Path(args.project_root)
    for task in args.task:
        answer = run_task(model, tokenizer, project_root, task)
        print(f"ANSWER: {answer}\n{'=' * 70}")


if __name__ == "__main__":
    main()
