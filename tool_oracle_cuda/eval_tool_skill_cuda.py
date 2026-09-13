#!/usr/bin/env python3
"""CUDA/transformers eval harness for the tool-augmented model -- the port of
tool_oracle/eval_tool_skill.py's agentic loop from MLX to PyTorch/peft, for
use on a Linux CUDA host (no MLX support on Linux).

Reuses eval_tool_skill's scoring/probe/summary logic (it's pure Python/sqlite,
made import-safe without mlx_lm earlier this session) by
monkey-patching its MLX-specific answer_question with a CUDA equivalent, then
calling its existing run()/build_eval_items()/load_probe_items()/summarize().
This avoids duplicated, potentially-drifted scoring logic. CUDA additionally
records generated-token count and an observed EOS/cap finish reason so an
unfinished generation cannot be scored as a final answer.

    python3 eval_tool_skill_cuda.py --adapter adapter-tool-v11-cuda \
        --db minecraft.db --probe tool_oracle/robustness_probe_full.jsonl \
        --out probe_v11_cuda.json
    python3 eval_tool_skill_cuda.py --adapter adapter-tool-v11-cuda \
        --db minecraft.db --held-out --train-jsonl train.jsonl --out held_out_v11.json
"""
import argparse
import hashlib
import importlib.metadata
import sys
import types
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# eval_tool_skill.run() does `from mlx_lm.sample_utils import make_sampler`
# internally (not just at module level) -- mlx has NO Linux wheels at all (it
# is Apple-Silicon/Metal only), so that import would hard-fail here. Our
# patched answer_question_cuda ignores the sampler argument entirely, so stub
# both modules before importing eval_tool_skill: the stub just needs to exist,
# not do anything real.
_mlx_lm = types.ModuleType("mlx_lm")
_mlx_lm_sample_utils = types.ModuleType("mlx_lm.sample_utils")
_mlx_lm_sample_utils.make_sampler = lambda *a, **k: None
_mlx_lm.sample_utils = _mlx_lm_sample_utils
_mlx_lm.generate = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("stubbed mlx_lm.generate called -- answer_question was not patched"))
_mlx_lm.load = _mlx_lm.generate
sys.modules["mlx_lm"] = _mlx_lm
sys.modules["mlx_lm.sample_utils"] = _mlx_lm_sample_utils

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tool_oracle import eval_tool_skill as ets  # noqa: E402
from tool_oracle.lookup import OracleDB  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
CUDA_DEVICE = "cuda:0"
ADAPTER_AUTOCAST_DTYPE = True


def _package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_runtime_controls():
    """Record execution controls without claiming bitwise determinism."""
    return {
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "cuda_runtime_version": torch.version.cuda,
        "target_device": CUDA_DEVICE,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _freeze_generation_controls(model, tok):
    """Freeze wrapper-sensitive controls before a PEFT model can mutate base."""
    eos_token_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_token_ids is None:
        eos_token_ids = tok.eos_token_id
    if isinstance(eos_token_ids, (list, tuple, set)):
        eos_token_ids = [int(token_id) for token_id in eos_token_ids]
    elif eos_token_ids is not None:
        eos_token_ids = int(eos_token_ids)
    return {
        "pad_token_id": int(tok.pad_token_id),
        "eos_token_ids": eos_token_ids,
        "input_device": str(next(model.parameters()).device),
    }


def make_answer_question_cuda(model, tok, generation_controls=None):
    controls = generation_controls or _freeze_generation_controls(model, tok)
    pad_token_id = controls["pad_token_id"]
    eos_token_ids = controls["eos_token_ids"]
    input_device = controls["input_device"]

    def answer_question_cuda(
        _model,
        _tok,
        db,
        question,
        max_tokens=ets.DEFAULT_EVAL_MAX_TOKENS,
        max_hops=3,
        sampler=None,
        seed=None,
        tool_manifest=None,
    ):
        ets._validate_run_controls(max_tokens, max_hops)
        sampling_seed = ets._question_seed(seed, question)
        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)
            torch.cuda.manual_seed_all(sampling_seed)
        convo = ets._initial_conversation(question, tool_manifest)
        first_call = None
        calls = []
        call_records = []
        raw_turns = []
        call_attempted = False
        answer = ""
        raw_answer = ""
        answer_span_ok = False
        answer_evaluable = False
        finish_reason = "hop_limit_without_answer"
        generated_tokens = None
        completion_observed = False
        for hop in range(max_hops):
            prompt = tok.apply_chat_template(
                convo, add_generation_prompt=True, tokenize=False
            )
            prompt_sha256 = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            inputs = tok(prompt, return_tensors="pt").to(input_device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=True,
                    temperature=0.2, top_p=0.9, top_k=50,
                    pad_token_id=pad_token_id, eos_token_id=eos_token_ids)
            generated = out[0][inputs["input_ids"].shape[1]:]
            generated_tokens, turn_finish_reason, turn_complete = (
                ets._classify_token_completion(
                    generated.tolist(),
                    max_tokens,
                    eos_token_ids,
                )
            )
            step = tok.decode(generated, skip_special_tokens=False)
            call = ets._parse_tool_call(step)
            attempted = ets._tool_call_attempted(step)
            call_attempted = call_attempted or attempted
            raw_turns.append({
                "text": step,
                "prompt_sha256": prompt_sha256,
                "generated_tokens": generated_tokens,
                "finish_reason": turn_finish_reason,
                "completion_observed": turn_complete,
                "tool_call_attempted": attempted,
                "parsed_call": ets._call_record(call),
            })
            if hop == 0:
                first_call = call
            if not call:
                raw_answer = step
                answer, answer_span_ok = ets._extract_final_answer(step)
                answer_evaluable = answer_span_ok and turn_complete
                completion_observed = turn_complete
                finish_reason = turn_finish_reason
                break
            calls.append((call.group(1), call.group(3)))
            call_records.append(ets._call_record(call))
            rows = ets._execute(db, call.group(1), call.group(3))
            assistant_turn = step.split("</tool_call>")[0] + "</tool_call>"
            convo.append({"role": "assistant", "content": assistant_turn})
            convo.append({"role": "user",
                          "content": f"<result>{', '.join(rows) if rows else '(no rows)'}</result>"})
            raw_answer = step
        return {"answer": answer, "raw_answer": raw_answer,
                "raw_turns": raw_turns,
                "answer_span_ok": answer_span_ok,
                "answer_evaluable": answer_evaluable,
                "completion_observed": completion_observed,
                "finish_reason": finish_reason,
                "generated_tokens": generated_tokens,
                "sampling_seed": sampling_seed,
                "tool_call_attempted": call_attempted,
                "first_call": first_call, "n_calls": len(calls), "calls": calls,
                "call_records": call_records}
    return answer_question_cuda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument(
        "--revision",
        default=None,
        help="exact Hugging Face commit required for remote-model evidence output",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="refuse network fetches for the base model and tokenizer",
    )
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--db", required=True)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--held-out", action="store_true")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--n-per", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--tool-manifest",
        default=str(ets.DEFAULT_TOOL_MANIFEST_PATH),
        help="system context and exact lookup signatures shared by both arms",
    )
    ap.add_argument(
        "--tool-manifest-sha256",
        default=None,
        help="predeclared raw-byte SHA-256; required when --out is used",
    )
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed; each question gets a stable derived sampling seed",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=ets.DEFAULT_EVAL_MAX_TOKENS,
        help=(
            "per-turn generation cap (default: 4096); a final answer that "
            "reaches the cap is recorded as non-evaluable"
        ),
    )
    args = ap.parse_args()

    try:
        ets._validate_run_controls(args.max_tokens, args.max_hops)
        ets._validate_output_target(args, harness_path=__file__)
        tool_manifest = ets._validate_evidence_bindings(args, backend="cuda")
    except ValueError as exc:
        ap.error(str(exc))

    args.runtime_controls = _cuda_runtime_controls()
    args.adapter_loading = {
        "autocast_adapter_dtype": ADAPTER_AUTOCAST_DTYPE,
    }

    db = OracleDB(args.db)
    if args.probe:
        items = ets.load_probe_items(db, args.probe)
        print(f"{len(items)} probe items")
    elif args.held_out:
        items = ets.build_eval_items(db, n_per=args.n_per, train_jsonl=args.train_jsonl)
        print(f"{len(items)} held-out eval items")
    else:
        sys.exit("pass --probe <file> or --held-out")

    metadata_before = (
        ets._evaluation_metadata(
            "cuda",
            args,
            items,
            harness_path=__file__,
            tool_manifest_binding=tool_manifest,
        )
        if args.out
        else None
    )

    print(f"loading base model {args.model}")
    tok = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map={"": CUDA_DEVICE},
    )
    model_revision = getattr(model.config, "_commit_hash", None)
    ets._validate_model_revision(args.revision, model_revision)
    args.loaded_model_name_or_path = getattr(
        model.config, "_name_or_path", None
    )
    if args.adapter:
        adapter_identity = ets._validate_adapter_config_identity(
            args.model, args.adapter
        )
        loaded_aliases = {
            args.model,
            adapter_identity["base_model_name_or_path"],
        }
        if (
            args.loaded_model_name_or_path
            and args.loaded_model_name_or_path not in loaded_aliases
        ):
            raise RuntimeError(
                "loaded model identity conflicts with the adapter base: "
                f"loaded={args.loaded_model_name_or_path!r}, "
                f"accepted={sorted(loaded_aliases)!r}"
            )
    generation_controls = _freeze_generation_controls(model, tok)
    args.frozen_generation_controls = generation_controls
    if metadata_before is not None:
        metadata_before["frozen_generation_controls"] = generation_controls
        metadata_before["loaded_model_name_or_path"] = (
            args.loaded_model_name_or_path
        )

    def run_arm(label, arm_model, shared_tok):
        ets.answer_question = make_answer_question_cuda(
            arm_model, shared_tok, generation_controls
        )
        results = ets.run(
            arm_model,
            shared_tok,
            db,
            items,
            args.max_tokens,
            args.max_hops,
            args.seed,
            tool_manifest,
        )
        ets.summarize(label.upper(), results)
        return results

    def attach_adapter(base_model, adapter_path):
        print(f"attaching adapter {adapter_path} to the evaluated base")
        return PeftModel.from_pretrained(
            base_model,
            adapter_path,
            local_files_only=args.local_files_only,
            autocast_adapter_dtype=ADAPTER_AUTOCAST_DTYPE,
        )

    base_res, ft_res = ets._evaluate_base_then_adapter(
        model,
        tok,
        args.adapter,
        run_arm,
        attach_adapter,
    )
    if args.out:
        manifest_after = ets._load_tool_manifest(
            args.tool_manifest, args.tool_manifest_sha256
        )
        metadata_after = ets._evaluation_metadata(
            "cuda",
            args,
            items,
            model_revision=model_revision,
            harness_path=__file__,
            tool_manifest_binding=manifest_after,
        )
        metadata = ets._finalize_evaluation_metadata(
            metadata_before,
            metadata_after,
            model_revision,
        )
        payload = ets._build_evidence_payload(metadata, base_res, ft_res)
        ets._write_json_atomic(args.out, payload)


if __name__ == "__main__":
    main()
