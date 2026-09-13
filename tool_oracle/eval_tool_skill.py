"""Eval harness for the tool-augmented model, on an Apple-Silicon Mac (MLX).

Runs a real agentic loop against the offline oracle: prompt the model, parse
its <tool_call>, EXECUTE it against minecraft.db, feed the real <result> back,
read the final answer, and score it MECHANICALLY vs ground truth. Base vs LoRA.

Two things are measured (both matter for the skill):
  - tool_call_ok : did the model emit a parseable lookup targeting the right
                   table + key for the question?
  - answer_ok    : does a completed, isolated final-answer span mechanically
                   cover the ground-truth rows? This is not a semantic judge.

Held-out by construction: eval items are built from the DB with
build_eval_items(), sampling subjects NOT in the training traces' set when a
--train-jsonl is given.

    PYTHONPATH=. python3 tool_oracle/eval_tool_skill.py \
        --model <base> --db minecraft.db [--adapter <dir>] [--train-jsonl <t>] [--out r.json]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tool_oracle.gen_query_traces import _item_use_facts, _depth_row_str
from tool_oracle.lookup import OracleDB

# mlx_lm is imported lazily inside run()/main() so the pure scoring/probe
# helpers stay importable on hosts without MLX (e.g. Linux CI/tests).

_CALL_RE = re.compile(
    r"lookup\(\s*(\w+)\s*,\s*(\w+)\s*=\s*'([^']*)'\s*\)"
)
_CALL_TURN_RE = re.compile(
    r"\s*(?:<think>.*?</think>\s*)?"
    r"<tool_call>(.*?)</tool_call>\s*(?:<\|im_end\|>)?\s*",
    re.DOTALL,
)
DEFAULT_EVAL_MAX_TOKENS = 4096
ANSWER_INSTRUMENT_VERSION = "mechanical-final-span-v3"
TOOL_CALL_PARSE_VERSION = "single-complete-signature-turn-v4"
TOOL_MANIFEST_VERSION = "minecraft-lookup-manifest-v1"
EVALUATION_SCHEMA_VERSION = "tool-eval-evidence-v6"
DEFAULT_TOOL_MANIFEST_PATH = Path(__file__).with_name("eval_tool_manifest.txt")
_LOOKUP_KEY_ARGUMENTS = {
    "breeding_food": "animal",
    "enchantment": "name",
    "item_use": "item",
    "jukebox_song": "song",
    "loot": "source",
    "loot_source": "drop_item",
    "ore_depth": "block",
    "painting": "painting",
    "recipe": "result_item",
    "tag": "tag",
    "villager_trade": "trade",
}
_BOUND_HASH_FIELDS = (
    "adapter_sha256",
    "db_sha256",
    "probe_sha256",
    "train_jsonl_sha256",
    "evaluation_items_sha256",
    "model_sha256",
    "evaluator_sha256",
    "backend_harness_sha256",
    "oracle_lookup_sha256",
    "trace_helpers_sha256",
    "tool_manifest_sha256",
)


@dataclass(frozen=True)
class LoadedToolManifest:
    """Exact prompt bytes decoded for inference and bound into evidence."""

    path: str
    text: str
    sha256: str
    version: str


def _parse_tool_call(text):
    """Parse one complete tool-call turn, with no stray text or tags."""
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        return None
    if text.count("<think>") != text.count("</think>"):
        return None
    if text.count("<think>") > 1:
        return None
    turn = _CALL_TURN_RE.fullmatch(text)
    if not turn:
        return None
    call = _CALL_RE.fullmatch(turn.group(1).strip())
    if not call:
        return None
    if _LOOKUP_KEY_ARGUMENTS.get(call.group(1)) != call.group(2):
        return None
    return call


def _call_record(call):
    """Return the complete parsed signature in a JSON-safe shape."""
    if not call:
        return None
    return {
        "table": call.group(1),
        "argument": call.group(2),
        "key": call.group(3),
    }


def _tool_call_attempted(text):
    """Detect malformed or unenveloped call syntax without executing it."""
    return (
        "<tool_call" in text
        or "</tool_call" in text
        or re.search(r"\blookup\s*\(", text) is not None
    )


def _question_seed(seed, question):
    """Derive an order-independent sampling seed for a question."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{int(seed)}\0{question}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _sha256_path(path):
    """Hash one file or a directory manifest including every file's bytes."""
    if not path:
        return None
    root = Path(path)
    if not root.exists():
        return None
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    )
    for candidate in files:
        name = (
            candidate.name
            if root.is_file()
            else candidate.relative_to(root).as_posix()
        )
        digest.update(name.encode())
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _tool_manifest_path(args):
    """Return the explicit prompt-manifest path, including the repo default."""
    return getattr(args, "tool_manifest", None) or str(DEFAULT_TOOL_MANIFEST_PATH)


def _validate_tool_manifest_text(text, source="provided text"):
    """Reject prompt text that drifts from the executable call contract."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"tool manifest is empty: {source}")
    version = re.search(r"^Manifest-Version:\s*(\S+)\s*$", text, re.MULTILINE)
    if not version or version.group(1) != TOOL_MANIFEST_VERSION:
        raise ValueError(
            "tool manifest version must be "
            f"{TOOL_MANIFEST_VERSION}: {source}"
        )
    signature_pairs = re.findall(
        r"^lookup\(([a-z_]+),\s*([a-z_]+)='KEY'\)\s*$",
        text,
        re.MULTILINE,
    )
    if len(signature_pairs) != len(_LOOKUP_KEY_ARGUMENTS) or dict(
        signature_pairs
    ) != _LOOKUP_KEY_ARGUMENTS:
        raise ValueError(
            "tool manifest signatures do not match the executable lookup "
            f"contract: {source}"
        )
    return text


def _load_tool_manifest_binding(path=None, expected_sha256=None):
    """Read, prebind, decode, and validate one exact prompt manifest."""
    manifest_path = Path(path or DEFAULT_TOOL_MANIFEST_PATH).expanduser().resolve()
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read tool manifest {manifest_path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("tool manifest SHA-256 must be exactly 64 hex digits")
        if digest != expected_sha256.lower():
            raise ValueError(
                "tool manifest SHA-256 does not match the predeclared value: "
                f"expected={expected_sha256.lower()}, actual={digest}"
            )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"tool manifest is not valid UTF-8: {manifest_path}") from exc
    _validate_tool_manifest_text(text, manifest_path)
    return LoadedToolManifest(
        path=str(manifest_path),
        text=text,
        sha256=digest,
        version=TOOL_MANIFEST_VERSION,
    )


def _load_tool_manifest(path=None, expected_sha256=None):
    """Load the exact non-empty system context shared by every evaluated arm."""
    return _load_tool_manifest_binding(path, expected_sha256)


def _initial_conversation(question, tool_manifest=None):
    """Build the explicit tool context shared by every evaluated arm."""
    manifest = _load_tool_manifest() if tool_manifest is None else tool_manifest
    text = manifest.text if isinstance(manifest, LoadedToolManifest) else manifest
    _validate_tool_manifest_text(text)
    return [
        {"role": "system", "content": text},
        {"role": "user", "content": question},
    ]


def _evaluation_metadata(
    backend,
    args,
    items,
    model_revision=None,
    harness_path=None,
    tool_manifest_binding=None,
):
    """Bind an output packet to inputs, controls, and evaluator source."""
    item_bytes = json.dumps(
        items, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    manifest = tool_manifest_binding or _load_tool_manifest_binding(
        _tool_manifest_path(args),
        getattr(args, "tool_manifest_sha256", None),
    )
    paired = bool(args.adapter)
    return {
        "schema": EVALUATION_SCHEMA_VERSION,
        "backend": backend,
        "model": args.model,
        "model_sha256": _sha256_path(args.model),
        "requested_model_revision": getattr(args, "revision", None),
        "model_revision": model_revision,
        "local_files_only": getattr(args, "local_files_only", False),
        "output": getattr(args, "out", None),
        "adapter": args.adapter,
        "adapter_sha256": _sha256_path(args.adapter),
        "adapter_config": (
            _validate_adapter_config_identity(args.model, args.adapter)
            if args.adapter and backend == "cuda"
            else _adapter_config_snapshot(args.adapter)
            if args.adapter
            else None
        ),
        "db": args.db,
        "db_sha256": _sha256_path(args.db),
        "probe": args.probe,
        "probe_sha256": _sha256_path(args.probe),
        "train_jsonl": args.train_jsonl,
        "train_jsonl_sha256": _sha256_path(args.train_jsonl),
        "evaluation_items_sha256": hashlib.sha256(item_bytes).hexdigest(),
        "evaluation_rows": len(items),
        "max_tokens": args.max_tokens,
        "max_hops": args.max_hops,
        "seed": args.seed,
        "sampling_seed_scheme": (
            "sha256(base_seed + NUL + exact_question_utf8), first_32_bits; "
            "common random numbers across arms, not bitwise replay authority"
        ),
        "deterministic_replay_claimed": False,
        "runtime_controls": getattr(args, "runtime_controls", None),
        "frozen_generation_controls": getattr(
            args, "frozen_generation_controls", None
        ),
        "adapter_loading": getattr(args, "adapter_loading", None),
        "loaded_model_name_or_path": getattr(
            args, "loaded_model_name_or_path", None
        ),
        "requested_arms": ["base", "adapter"] if paired else ["base"],
        "tool_manifest": manifest.path,
        "tool_manifest_version": manifest.version,
        "requested_tool_manifest_sha256": getattr(
            args, "tool_manifest_sha256", None
        ),
        "tool_manifest_sha256": manifest.sha256,
        "tool_manifest_text": manifest.text,
        "answer_instrument": ANSWER_INSTRUMENT_VERSION,
        "tool_call_parser": TOOL_CALL_PARSE_VERSION,
        "sampling": {
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 50 if backend == "cuda" else None,
        },
        "evaluator_sha256": _sha256_path(__file__),
        "backend_harness_sha256": _sha256_path(harness_path),
        "oracle_lookup_sha256": _sha256_path(
            Path(__file__).with_name("lookup.py")
        ),
        "trace_helpers_sha256": _sha256_path(
            Path(__file__).with_name("gen_query_traces.py")
        ),
    }


def _validate_output_target(args, harness_path=None):
    """Reject evidence output that aliases or nests inside a bound input."""
    output = getattr(args, "out", None)
    if not output:
        return
    target = Path(output).expanduser().resolve(strict=False)
    candidates = {
        "model": getattr(args, "model", None),
        "adapter": getattr(args, "adapter", None),
        "database": getattr(args, "db", None),
        "probe": getattr(args, "probe", None),
        "training JSONL": getattr(args, "train_jsonl", None),
        "tool manifest": _tool_manifest_path(args),
        "evaluator": __file__,
        "backend harness": harness_path,
    }
    for label, raw_path in candidates.items():
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve(strict=False)
        aliases = target == candidate
        if target.exists() and candidate.exists():
            aliases = aliases or os.path.samefile(target, candidate)
        nested = candidate.is_dir() and candidate in target.parents
        if aliases or nested:
            raise ValueError(
                f"output path must not alias or be inside bound {label}: "
                f"{raw_path}"
            )


def _validate_evidence_bindings(args, backend="cuda"):
    """Require predeclared identities and return the exact prompt binding."""
    expected_manifest_sha256 = getattr(args, "tool_manifest_sha256", None)
    manifest = _load_tool_manifest_binding(
        _tool_manifest_path(args),
        expected_manifest_sha256,
    )
    adapter = getattr(args, "adapter", None)
    evidence_output = bool(getattr(args, "out", None))
    if evidence_output and adapter and _sha256_path(adapter) is None:
        raise ValueError(
            "evidence output requires --adapter to name local hashable bytes"
        )
    if adapter and backend == "cuda":
        _validate_adapter_config_identity(args.model, adapter)
    if not evidence_output:
        return manifest
    model_is_local = _sha256_path(args.model) is not None
    revision = getattr(args, "revision", None)
    revision_is_exact = bool(
        revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
    )
    if not model_is_local and not revision_is_exact:
        raise ValueError(
            "evidence output for a remote model requires an exact --revision"
        )
    if expected_manifest_sha256 is None:
        raise ValueError(
            "evidence output requires a predeclared --tool-manifest-sha256"
        )
    return manifest


def _adapter_config_snapshot(adapter_path):
    """Load a local adapter config for evidence without assuming its backend."""
    if not adapter_path:
        return None
    root = Path(adapter_path).expanduser()
    config_path = root / "adapter_config.json" if root.is_dir() else root
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read adapter config {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"adapter config must be a JSON object: {config_path}")
    return config


def _adapter_config_identity(adapter_path):
    """Load the PEFT lineage fields that must travel with CUDA evidence."""
    config = _adapter_config_snapshot(adapter_path)
    identity = {
        field: config.get(field)
        for field in ("base_model_name_or_path", "peft_type", "task_type")
    }
    if any(not identity[field] for field in identity):
        raise ValueError(
            "adapter config must declare base_model_name_or_path, peft_type, "
            f"and task_type: {adapter_path}"
        )
    return identity


def _validate_adapter_config_identity(model_path, adapter_path):
    """Reject an adapter whose declared task or base conflicts with the model."""
    identity = _adapter_config_identity(adapter_path)
    if identity["peft_type"] != "LORA" or identity["task_type"] != "CAUSAL_LM":
        raise ValueError(
            "adapter config must be a LORA adapter for CAUSAL_LM"
        )
    declared_base = identity["base_model_name_or_path"]
    model_root = Path(model_path).expanduser()
    aliases = {str(model_path)}
    if model_root.exists():
        config_path = model_root / "config.json" if model_root.is_dir() else None
        if config_path and config_path.is_file():
            try:
                model_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot read local model config {config_path}: {exc}"
                ) from exc
            aliases.update(
                value
                for value in (
                    model_config.get("_name_or_path"),
                    model_config.get("name_or_path"),
                )
                if value
            )
    if declared_base not in aliases:
        raise ValueError(
            "adapter base_model_name_or_path does not match the evaluated "
            f"model: declared={declared_base!r}, accepted={sorted(aliases)!r}"
        )
    return identity


def _evaluate_base_then_adapter(
    base_model,
    tokenizer,
    adapter_path,
    run_arm,
    attach_adapter,
):
    """Dependency-injected ordering guard for a same-process paired run."""
    base_model.eval()
    base_results = run_arm("base", base_model, tokenizer)
    if not adapter_path:
        return base_results, None
    adapter_model = attach_adapter(base_model, adapter_path)
    adapter_model.eval()
    adapter_results = run_arm("adapter", adapter_model, tokenizer)
    return base_results, adapter_results


_PAIRED_ITEM_FIELDS = (
    "category",
    "table",
    "key",
    "question",
    "expected",
    "kind",
    "second_table",
    "second_key",
)


def _paired_item_identity(result):
    """Project one arm result onto fields that define evaluation-row order."""
    return {
        field: result.get(field)
        for field in _PAIRED_ITEM_FIELDS
        if field in result
    }


def _validate_paired_results(base_results, adapter_results):
    """Fail closed unless both arms used identical rows, seeds, and prompts."""
    if adapter_results is None:
        raise ValueError("paired evidence requires a completed adapter arm")
    if len(base_results) != len(adapter_results):
        raise ValueError("paired arms have different row counts")
    if not base_results:
        raise ValueError("paired evidence requires at least one evaluation row")
    identities = []
    for index, (base, adapter) in enumerate(zip(base_results, adapter_results)):
        base_identity = _paired_item_identity(base)
        adapter_identity = _paired_item_identity(adapter)
        if base_identity != adapter_identity:
            raise ValueError(f"paired arm row order differs at index {index}")
        if base.get("sampling_seed") != adapter.get("sampling_seed"):
            raise ValueError(f"paired arm sampling seed differs at index {index}")
        base_turns = base.get("raw_turns") or []
        adapter_turns = adapter.get("raw_turns") or []
        if not base_turns or not adapter_turns:
            raise ValueError(f"paired arm lacks a first raw turn at index {index}")
        base_prompt = base_turns[0].get("prompt_sha256")
        adapter_prompt = adapter_turns[0].get("prompt_sha256")
        if not base_prompt or base_prompt != adapter_prompt:
            raise ValueError(
                f"paired arm initial prompt differs at index {index}"
            )
        identities.append(base_identity)
    identity_bytes = json.dumps(
        identities, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "validated": True,
        "row_count": len(identities),
        "paired_items_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "row_order_parity": True,
        "sampling_seed_parity": True,
        "initial_prompt_parity": True,
    }


def _build_evidence_payload(metadata, base_results, adapter_results=None):
    """Add comparison claims only after validating the completed arm outputs."""
    bound_metadata = dict(metadata)
    if adapter_results is None:
        if bound_metadata.get("requested_arms") == ["base", "adapter"]:
            raise ValueError(
                "adapter evaluation was requested but did not complete"
            )
        bound_metadata.update({
            "arms": ["base"],
            "arm_order": ["base"],
            "comparison_mode": "single_arm",
            "same_process": True,
        })
        return {"metadata": bound_metadata, "base": base_results}
    pairing = _validate_paired_results(base_results, adapter_results)
    bound_metadata.update({
        "arms": ["base", "adapter"],
        "arm_order": ["base", "adapter"],
        "comparison_mode": "within_backend_paired_items",
        "same_process": True,
        "base_evaluated_before_adapter_load": True,
        "base_evaluated_before_adapter_attach": (
            bound_metadata.get("backend") == "cuda"
        ),
        "initial_prompt_parity": True,
        "pairing_validation": pairing,
    })
    return {
        "metadata": bound_metadata,
        "base": base_results,
        "adapter": adapter_results,
    }


def _finalize_evaluation_metadata(before, after, model_revision=None):
    """Fail if any hash-bound input changed while evaluation was running."""
    changed = {
        field: {"before": before.get(field), "after": after.get(field)}
        for field in _BOUND_HASH_FIELDS
        if before.get(field) != after.get(field)
    }
    if changed:
        raise RuntimeError(f"evaluation inputs changed during run: {changed}")
    _validate_model_revision(before.get("requested_model_revision"), model_revision)
    result = dict(before)
    result["model_revision"] = model_revision
    result["evidence_binding_complete"] = bool(
        result.get("model_sha256") or model_revision
    ) and bool(
        not result.get("adapter") or result.get("adapter_sha256")
    ) and bool(result.get("tool_manifest_sha256"))
    result["inputs_stable_during_run"] = True
    result["post_run_hashes"] = {
        field: after.get(field) for field in _BOUND_HASH_FIELDS
    }
    return result


def _validate_model_revision(requested_revision, model_revision):
    """Reject a resolved base that differs from an explicitly requested one."""
    if requested_revision and model_revision != requested_revision:
        raise RuntimeError(
            "loaded model revision does not match requested revision: "
            f"requested={requested_revision!r}, loaded={model_revision!r}"
        )


def _validate_run_controls(max_tokens, max_hops):
    """Reject degenerate generation controls before allocating model work."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if max_hops < 1:
        raise ValueError("max_hops must be at least 1")


def _write_json_atomic(path, payload):
    """Publish a complete JSON packet without exposing a partial write."""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bare(i):
    return i.split(":", 1)[-1].replace("_", " ")


def _subject(i):
    """Return the player-facing subject from a namespaced or path key."""
    return i.split(":", 1)[-1].rsplit("/", 1)[-1].replace("_", " ")


def _contains_phrase(text, phrase):
    """Match a normalized phrase without token-prefix collisions."""
    pattern = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def _bind_answer_subject(category, key, answer):
    """Bind precision-first recipe/loot answers to their queried subject."""
    subject = re.escape(_subject(key)).replace(r"\ ", r"\s+")
    article = r"(?:(?:a|an|the)\s+)?"
    if category == "recipe":
        return re.search(
            rf"\b(?:craft|make)\s+{article}{subject}(?![a-z0-9])", answer
        ) is not None
    if category == "loot":
        return re.search(
            rf"(?<![a-z0-9]){subject}\s+(?:drops?|can\s+drop|will\s+drop)\b",
            answer,
        ) is not None
    return False


def _execute(db, table, key):
    if table == "breeding_food":
        return db.breeding_foods(key)
    if table == "recipe":
        return db.recipe_ingredient_groups(key)
    if table == "loot":
        return db.loot_drops(key)
    if table == "loot_source":
        return db.loot_sources(key)
    if table == "ore_depth":
        return [_depth_row_str(r) for r in db.ore_depths(key)]
    if table == "tag":
        return db.tag_members(key)
    if table == "enchantment":
        ml = db.enchantment_max_level(key)
        return [f"max_level={ml}"] if ml is not None else []
    if table == "villager_trade":
        t = db.trade(key)
        return [f"wants={int(t[1] or 1)} {t[0]}", f"gives={int(t[3] or 1)} {t[2]}"] if t else []
    if table == "jukebox_song":
        length = db.jukebox_length(key)
        return [f"length={length}"] if length is not None else []
    if table == "painting":
        wh = db.painting_size(key)
        return [f"width={wh[0]}", f"height={wh[1]}"] if wh else []
    if table == "item_use":
        u = db.item_use(key)
        return _item_use_facts(u) if u else []
    return []


def build_eval_items(db, n_per=10, train_jsonl=None):
    """Held-out eval items across categories, with expected ground truth."""
    trained = set()
    if train_jsonl:
        for line in open(train_jsonl):
            trained |= set(re.findall(r"lookup\(\w+,\s*\w+='([^']*)'\)", line))
    items = []

    def add(cat, table, key, question, expected):
        items.append({"category": cat, "table": table, "key": key,
                      "question": question, "expected": expected})

    foods = [(a,) for (a,) in db.con.execute("SELECT DISTINCT animal FROM breeding_food")]
    for (animal,) in foods:
        if animal in trained:
            continue
        exp = db.breeding_foods(animal)
        if exp:
            add("breeding_food", "breeding_food", animal,
                f"What do {animal}s eat to breed in Minecraft?", exp)
        if len([i for i in items if i["category"] == "breeding_food"]) >= n_per:
            break
    for (item,) in db.con.execute("SELECT DISTINCT result_item FROM recipe"):
        if item in trained:
            continue
        exp = db.recipe_ingredients(item)
        if exp:
            add("recipe", "recipe", item, f"What items do you need to craft {_bare(item)}?", exp)
        if len([i for i in items if i["category"] == "recipe"]) >= n_per:
            break
    for name, ml in db.con.execute("SELECT name, max_level FROM enchantment WHERE max_level IS NOT NULL"):
        if name in trained:
            continue
        add("enchantment", "enchantment", name,
            f"What is the maximum level of {_bare(name)}?", [f"max_level={ml}"])
        if len([i for i in items if i["category"] == "enchantment"]) >= n_per:
            break
    for src in [r[0] for r in db.con.execute("SELECT DISTINCT source FROM loot")]:
        if src in trained:
            continue
        exp = db.loot_drops(src)
        if exp:
            add("loot", "loot", src, f"What does {src.split('/')[-1].replace('_', ' ')} drop?", exp)
        if len([i for i in items if i["category"] == "loot"]) >= n_per:
            break
    for (trade,) in db.con.execute("SELECT DISTINCT trade FROM villager_trade"):
        if trade in trained:
            continue
        t = db.trade(trade)
        if t:
            add("villager_trade", "villager_trade", trade,
                f"In the villager trade {trade}, what does the villager want and give?",
                [f"wants={int(t[1] or 1)} {t[0]}", f"gives={int(t[3] or 1)} {t[2]}"])
        if len([i for i in items if i["category"] == "villager_trade"]) >= n_per:
            break
    for song, length in db.con.execute("SELECT song, length_seconds FROM jukebox_song WHERE length_seconds IS NOT NULL"):
        if song in trained:
            continue
        add("jukebox_song", "jukebox_song", song,
            f"How many seconds long is the music disc {_bare(song)}?", [f"length={length}"])
        if len([i for i in items if i["category"] == "jukebox_song"]) >= n_per:
            break
    for painting, w, h in db.con.execute("SELECT painting, width, height FROM painting"):
        if painting in trained:
            continue
        add("painting", "painting", painting,
            f"What are the width and height in blocks of the {_bare(painting)} painting?",
            [f"width={w}", f"height={h}"])
        if len([i for i in items if i["category"] == "painting"]) >= n_per:
            break
    for (item,) in db.con.execute("SELECT DISTINCT item FROM item_use"):
        if item in trained:
            continue
        u = db.item_use(item)
        exp = _item_use_facts(u) if u else []
        if exp:
            add("item_use", "item_use", item,
                f"What is {_bare(item)} used for?", exp)
        if len([i for i in items if i["category"] == "item_use"]) >= n_per:
            break
    return items


def _score(item, answer):
    a = answer.lower()
    cat = item["category"]
    # A decline on an item that DOES have real expected data is wrong -- even
    # when the decline sentence happens to name the subject itself ("I don't
    # have data on ancient debris" contains "ancient debris"), which would
    # otherwise slip past the substring checks below by accident. Checked
    # before category dispatch since a false decline can happen in any of them.
    if item["expected"] and any(m in a for m in _NODATA):
        return False
    if cat in ("enchantment", "jukebox_song", "painting", "villager_trade", "item_use"):
        # scalar/structured: each expected value's tokens must appear
        for e in item["expected"]:
            val = e.split("=")[-1].replace("minecraft:", "").replace("_", " ")
            # A villager-trade value includes a count and an item. Keep that
            # phrase together: independent token membership allowed a
            # truncated answer to collect the right words from unrelated
            # clauses and allowed "white glazed terracotta" to satisfy
            # "white terracotta".
            value_matches = _contains_phrase(a, val)
            if not value_matches:
                return False
        # Value membership alone does not prove the answer is about this
        # item: distinct subjects can share a scalar value. Require the
        # subject for simple ids. Compound paths such as villager-trade slots
        # do not expose one unambiguous subject name and remain a known gap.
        key = item.get("key", "")
        if ":" in key and "/" not in key.split(":", 1)[1]:
            subject = _bare(key)
            if not _contains_phrase(a, subject):
                return False
        return True
    if cat == "tag" and len(item["expected"]) > 12:
        # large tag: count + sample, not a full enumeration (see gen_query_traces)
        cited = sum(_bare(e) in a for e in item["expected"])
        return str(len(item["expected"])) in a and cited >= 3
    if cat in ("recipe", "loot") and not _bind_answer_subject(
        cat, item.get("key", ""), a
    ):
        # Precision-first answer scoring: without the subject, a complete
        # answer for a different recipe/loot row can satisfy this row's
        # shorter expected-value set. Reject that ambiguous answer shape.
        return False
    return all(_bare(e) in a for e in item["expected"])


def _extract_final_answer(text):
    """Return only an explicit final-answer span from one raw generation.

    Qwen-style generations may contain ``<think>...</think>`` before the
    user-facing answer.  Scoring the whole generation lets an unfinished
    reasoning trace satisfy expected-token checks merely by restating the
    question.  An opened but unclosed reasoning span therefore has no final
    answer.  Direct responses and text after the last closed reasoning span
    remain eligible; token-level completion is recorded by the backend.
    """
    visible = re.split(r"<\|im_end\|>|<\|im_start\|>", text, maxsplit=1)[0]
    if "<tool_call" in visible or "</tool_call" in visible:
        return "", False
    opens = visible.count("<think>")
    closes = visible.count("</think>")
    if opens != closes or opens > 1:
        return "", False
    if opens:
        reasoning = re.fullmatch(r"\s*<think>.*?</think>\s*(.*?)\s*", visible, re.DOTALL)
        if not reasoning:
            return "", False
        answer = reasoning.group(1).strip()
    else:
        answer = visible.strip()
    return answer, bool(answer)


def _clean(text):
    """Compatibility wrapper returning the isolated final-answer text."""
    return _extract_final_answer(text)[0]


def _classify_token_completion(token_ids, max_tokens, eos_token_ids):
    """Classify a token-bearing generation without trusting decoded prose."""
    ids = [int(token_id) for token_id in token_ids]
    if isinstance(eos_token_ids, int):
        eos_ids = {eos_token_ids}
    else:
        eos_ids = {int(token_id) for token_id in (eos_token_ids or [])}
    ended_eos = bool(ids and ids[-1] in eos_ids)
    if ended_eos:
        return len(ids), "eos", True
    if len(ids) >= max_tokens:
        return len(ids), "max_tokens", False
    return len(ids), "stopped", True


# Markers that signal the model correctly declined instead of fabricating a
# fact for a subject the oracle has no row for.
_NODATA = ("no data", "no information", "no info", "not have", "don't have",
           "doesn't have", "couldn't find", "could not find", "not found",
           "no result", "not available", "no such", "isn't in", "not in the",
           "no recipe", "no drops", "does not exist", "doesn't exist",
           "unable to", "can't find", "cannot find", "nothing",
           "does not contain", "doesn't contain", "contains no", "no items",
           "does not include", "not a real", "isn't a real")


def _score_unknown(answer):
    """Unknown subject: PASS iff the model declined (no fabricated fact)."""
    return any(m in answer.lower() for m in _NODATA)


def load_probe_items(db, path):
    """Load a hand-written probe JSONL and derive ground truth from the DB.

    Each line: {"question", "table", "key", "kind": "known"|"unknown"|"multihop"
    |"usage"}. For known items expected rows come from the live oracle; unknown
    items (subjects with no DB row) expect a decline, not a fabricated answer;
    usage items (a REAL item, but a purpose/mechanics question no table
    models) expect a decline WITHOUT calling any tool at all.
    """
    items = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        p = json.loads(line)
        kind = p.get("kind", "known")
        if kind == "multihop":
            # ground truth = second hop's ingredients (make the craftable
            # ingredient/food Y = p["second_key"]); the answer must at least cover
            # those. First hop may be recipe or breeding_food (p["table"]).
            expected = _execute(db, p.get("second_table", "recipe"), p["second_key"])
        elif kind == "known":
            expected = _execute(db, p["table"], p["key"])
        else:
            expected = []
        item = {"category": p.get("table", "usage"), "table": p.get("table"),
                "key": p.get("key"), "question": p["question"], "expected": expected,
                "kind": kind}
        for opt in ("second_key", "second_table"):
            if opt in p:
                item[opt] = p[opt]
        items.append(item)
    return items


def _tool_call_verdict(item, first_call, calls, call_attempted=False):
    """Apply one explicit call contract and return ``(ok, contract_name)``."""
    kind = item.get("kind", "known")
    if kind == "usage":
        return first_call is None and not calls and not call_attempted, "no_call"
    if not first_call:
        contract = "exact_first_two_calls" if kind == "multihop" else "exact_first_call"
        return False, contract

    first_expected = (item["table"], item["key"])
    if kind == "multihop":
        second_expected = (
            item.get("second_table", "recipe"),
            item["second_key"],
        )
        return calls[:2] == [first_expected, second_expected], "exact_first_two_calls"
    return (first_call.group(1), first_call.group(3)) == first_expected, "exact_first_call"


def answer_question(
    model,
    tok,
    db,
    question,
    max_tokens=DEFAULT_EVAL_MAX_TOKENS,
    max_hops=3,
    sampler=None,
    seed=None,
    tool_manifest=None,
):
    """Run the agentic loop for ONE question and return the composed answer plus
    the lookups the model made. Shared by the eval harness and the REPL server.

    Keeps executing the model's tool_calls against the oracle until it composes
    an answer (no tool_call) or the hop budget is exhausted -- handling single-
    hop, no-tool, and multi-hop (chained lookups) uniformly. The returned dict
    includes the isolated answer, raw final generation, calls, and completion
    evidence. The MLX string API does not expose token counts or a structured
    finish reason, so completion is observed only when its decoded string keeps
    ``<|im_end|>``. Every raw turn is retained instead of being irreversibly
    cleaned away.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    _validate_run_controls(max_tokens, max_hops)
    sampling_seed = _question_seed(seed, question)
    if sampling_seed is not None:
        import mlx.core as mx

        mx.random.seed(sampling_seed)
    if sampler is None:
        sampler = make_sampler(temp=0.2, top_p=0.9)
    convo = _initial_conversation(question, tool_manifest)
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
    for hop in range(max_hops):
        prompt = tok.apply_chat_template(
            convo, add_generation_prompt=True, tokenize=False
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        step = generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        call = _parse_tool_call(step)
        attempted = _tool_call_attempted(step)
        call_attempted = call_attempted or attempted
        turn_finish_reason = "eos_text" if "<|im_end|>" in step else "unobserved_mlx"
        raw_turns.append({
            "text": step,
            "prompt_sha256": prompt_sha256,
            "generated_tokens": None,
            "finish_reason": turn_finish_reason,
            "completion_observed": turn_finish_reason == "eos_text",
            "tool_call_attempted": attempted,
            "parsed_call": _call_record(call),
        })
        if hop == 0:
            first_call = call
        if not call:
            raw_answer = step
            answer, answer_span_ok = _extract_final_answer(step)
            completion_observed = "<|im_end|>" in step
            answer_evaluable = answer_span_ok and completion_observed
            finish_reason = "eos_text" if completion_observed else "unobserved_mlx"
            break
        calls.append((call.group(1), call.group(3)))
        call_records.append(_call_record(call))
        rows = _execute(db, call.group(1), call.group(3))
        assistant_turn = step.split("</tool_call>")[0] + "</tool_call>"
        convo.append({"role": "assistant", "content": assistant_turn})
        convo.append({"role": "user",
                      "content": f"<result>{', '.join(rows) if rows else '(no rows)'}</result>"})
        raw_answer = step
    completion_observed = finish_reason == "eos_text"
    return {"answer": answer, "raw_answer": raw_answer,
            "raw_turns": raw_turns,
            "answer_span_ok": answer_span_ok,
            "answer_evaluable": answer_evaluable,
            "completion_observed": completion_observed,
            "finish_reason": finish_reason,
            "generated_tokens": None,
            "sampling_seed": sampling_seed,
            "tool_call_attempted": call_attempted,
            "first_call": first_call, "n_calls": len(calls), "calls": calls,
            "call_records": call_records}


def run(
    model,
    tok,
    db,
    items,
    max_tokens,
    max_hops=3,
    seed=None,
    tool_manifest=None,
):
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.2, top_p=0.9)
    results = []
    for it in items:
        kind = it.get("kind", "known")
        res = answer_question(
            model,
            tok,
            db,
            it["question"],
            max_tokens,
            max_hops,
            sampler,
            seed,
            tool_manifest,
        )
        answer, first_call, n_calls = res["answer"], res["first_call"], res["n_calls"]
        calls = res.get("calls", [])
        answer_evaluable = res.get("answer_evaluable", False)
        call_attempted = res.get("tool_call_attempted", bool(first_call or calls))
        tool_call_ok, tool_call_contract = _tool_call_verdict(
            it, first_call, calls, call_attempted
        )
        if kind == "usage":
            # correct behavior: NO tool call at all (no table models purpose/
            # mechanics), and the answer must decline.
            answer_ok = answer_evaluable and _score_unknown(answer)
        elif kind == "unknown":
            answer_ok = answer_evaluable and _score_unknown(answer)
        elif kind == "multihop":
            answer_ok = answer_evaluable and all(
                _bare(e) in answer.lower() for e in it["expected"]
            )
        else:
            answer_ok = answer_evaluable and _score(it, answer)
        results.append({**it, "tool_call_ok": tool_call_ok, "n_calls": n_calls,
                        "calls": calls, "call_records": res.get("call_records", []),
                        "tool_call_contract": tool_call_contract,
                        "answer": answer, "raw_answer": res.get("raw_answer", answer),
                        "raw_turns": res.get("raw_turns", []),
                        "tool_call_attempted": call_attempted,
                        "answer_span_ok": res.get("answer_span_ok", bool(answer)),
                        "answer_evaluable": answer_evaluable,
                        "completion_observed": res.get("completion_observed", False),
                        "finish_reason": res.get("finish_reason", "unobserved"),
                        "generated_tokens": res.get("generated_tokens"),
                        "sampling_seed": res.get("sampling_seed"),
                        "answer_instrument": ANSWER_INSTRUMENT_VERSION,
                        "tool_call_parser": TOOL_CALL_PARSE_VERSION,
                        "answer_ok": answer_ok})
    return results


def summarize(label, results):
    tc = sum(r["tool_call_ok"] for r in results)
    an = sum(r["answer_ok"] for r in results)
    n = len(results)
    print(f"\n=== {label} ({n} items) ===")
    print(f"  tool-call contract pass: {tc}/{n} ({100*tc//max(n,1)}%)")
    print(f"  mechanical answer match: {an}/{n} ({100*an//max(n,1)}%)")
    not_evaluable = sum(not r.get("answer_evaluable", True) for r in results)
    finish_reasons = {}
    for result in results:
        reason = result.get("finish_reason", "unobserved")
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    if not_evaluable:
        print(f"  answer non-evaluable:    {not_evaluable}/{n}")
    print("  finish reasons:          " + ", ".join(
        f"{reason}={count}" for reason, count in sorted(finish_reasons.items())
    ))
    known = [r for r in results if r.get("kind", "known") == "known"]
    unknown = [r for r in results if r.get("kind") == "unknown"]
    multihop = [r for r in results if r.get("kind") == "multihop"]
    if unknown:
        ka = sum(r["answer_ok"] for r in known)
        ua = sum(r["answer_ok"] for r in unknown)
        print(f"  known-subject matches:   {ka}/{len(known)} (mechanical)")
        print(f"  unknown decline markers: {ua}/{len(unknown)} (mechanical)")
    if multihop:
        ma = sum(r["answer_ok"] for r in multihop)
        mc = sum(r["tool_call_ok"] for r in multihop)
        print(f"  multi-hop chains:        {ma}/{len(multihop)} matched, "
              f"{mc}/{len(multihop)} used the expected first two lookups")
    usage = [r for r in results if r.get("kind") == "usage"]
    if usage:
        ua = sum(r["answer_ok"] for r in usage)
        ut = sum(r["tool_call_ok"] for r in usage)
        print(f"  usage decline markers:   {ua}/{len(usage)} (mechanical), "
              f"{ut}/{len(usage)} passed the no-call contract")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--probe", default=None,
                    help="JSONL of hand-written robustness/unknown-subject probes; "
                         "overrides the auto-built held-out set")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--tool-manifest",
        default=str(DEFAULT_TOOL_MANIFEST_PATH),
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
        default=DEFAULT_EVAL_MAX_TOKENS,
        help=(
            "per-turn generation cap (default: 4096); a final answer that "
            "reaches the cap is recorded as non-evaluable"
        ),
    )
    args = ap.parse_args()

    try:
        _validate_run_controls(args.max_tokens, args.max_hops)
        _validate_output_target(args)
        tool_manifest = _validate_evidence_bindings(args, backend="mlx")
    except ValueError as exc:
        ap.error(str(exc))

    db = OracleDB(args.db)
    if args.probe:
        items = load_probe_items(db, args.probe)
        print(f"{len(items)} probe items ({sum(i['kind']=='unknown' for i in items)} unknown-subject)")
    else:
        items = build_eval_items(db, train_jsonl=args.train_jsonl)
        print(f"{len(items)} held-out eval items")

    metadata_before = (
        _evaluation_metadata(
            "mlx", args, items, tool_manifest_binding=tool_manifest
        )
        if args.out
        else None
    )

    from mlx_lm import load

    base, tok = load(args.model)
    base_res = run(
        base,
        tok,
        db,
        items,
        args.max_tokens,
        args.max_hops,
        args.seed,
        tool_manifest,
    )
    summarize("BASE", base_res)
    ft_res = None
    if args.adapter:
        ft, tok2 = load(args.model, adapter_path=args.adapter)
        ft_res = run(
            ft,
            tok2,
            db,
            items,
            args.max_tokens,
            args.max_hops,
            args.seed,
            tool_manifest,
        )
        summarize("FINE-TUNED", ft_res)
    if args.out:
        manifest_after = _load_tool_manifest(
            args.tool_manifest, args.tool_manifest_sha256
        )
        metadata_after = _evaluation_metadata(
            "mlx", args, items, tool_manifest_binding=manifest_after
        )
        metadata = _finalize_evaluation_metadata(
            metadata_before, metadata_after
        )
        payload = _build_evidence_payload(metadata, base_res, ft_res)
        _write_json_atomic(args.out, payload)


if __name__ == "__main__":
    main()
