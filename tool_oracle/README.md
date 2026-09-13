# tool_oracle -- offline fact oracle + verified query-traces

The round-2 architecture for the Minecraft domain-model case study: instead of
fine-tuning a model to *memorize* facts (round 1, which memorized specific
facts but couldn't generalize and could still fabricate), teach it to *query*
an offline fact table. One source -- the pinned data report -- becomes the
verifier, the inference-time tool, and the training data.

See `docs/reports/2026-08-16-minecraft-domain-case-study.md` for the full
result and rationale.

## Pieces

- **`build_db.py`** -- dumps the data report into a ~1 MB SQLite oracle
  (`breeding_food`, `tag`, `recipe`, `enchantment`, `villager_trade`, `loot`),
  with recursive `#tag` reference resolution. Regenerable per game version.
- **`lookup.py`** -- `OracleDB`: the `lookup_*` TOOL (ground-truth queries) and
  the `is_*` VERIFIER (rejects a proposed item that isn't in the table -- so
  "corn" for pigs, or a real item for the wrong animal, can't reach an answer).
- **`gen_query_traces.py`** -- verified `question -> <tool_call> -> <result> ->
  answer` training traces, each re-checked by `verify_trace()` (result equals
  the DB and the generated answer mechanically covers the returned rows).
  This admission check does not prove that a model will call the tool or state
  only supported facts at inference time.
- **`eval_tool_skill.py`** and **`tool_oracle_cuda/eval_tool_skill_cuda.py`** --
  agentic evaluation against the live oracle. Evidence packets retain every
  raw turn, rendered-prompt hash, and exact parsed call; record completion;
  and hash their bound inputs before and after the run. Both arms receive the
  same validated `eval_tool_manifest.txt` system context. Evidence output
  requires its operator-predeclared raw-byte SHA-256. CUDA evaluates base
  before attaching LoRA and publishes only after paired row, seed, and initial
  prompt parity succeeds.

## Run (from the repo root)

```bash
PYTHONPATH=. python3 tool_oracle/build_db.py <data_report_dir> minecraft.db
PYTHONPATH=. python3 tool_oracle/gen_query_traces.py minecraft.db traces.jsonl
PYTHONPATH=. pytest tests/test_tool_oracle.py -q
sha256sum tool_oracle/eval_tool_manifest.txt
```

Manifest-enabled measurements are a new prompt condition. They must not be
presented as direct replays of the historical v20 runs, which gave both arms
only the user question and therefore did not expose the interface to the base.

## Why this is the stronger case study

| | round 1 (memorize) | tool_oracle (query) |
|---|---|---|
| Fact source | model weights | pinned table for successfully executed calls |
| Fabrication | observed | still possible if the model skips or misuses the tool |
| New game version | full retrain | rebuild the DB |
| Fine-tuning's role | its weakest (memorize) | its strongest (a verifiable skill) |
| Mirrors strict-C flagship | no | yes (skill + oracle) |
