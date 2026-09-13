# v20 Evaluation Measurement Correction

> Status: historical evidence audited; manifest-enabled paired rerun pending

This note corrects the September 2026 four-arm comparison between
`Qwen/Qwen3-1.7B` and the v20 LoRA adapter. It preserves what the saved packet
actually supports and retracts what its evaluator could not measure.

## Evidence Identity

The correction was recomputed from these exact preserved artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `held_out_base.json` | `4eca6240cb3ee5038ca415698a13d3edfdfca33e925de8e4728d9ba56319513b` |
| `held_out_v20.json` | `07118b2f14360f5108daf28bb6daea9004fe9c56f70d2b3cb588df59026c1043` |
| `probe_base.json` | `bb39c5905a3f7189bf8a3610e7332fa813753d07437e94cbe06e5b8c4e1b8f13` |
| `probe_v20.json` | `7ecd3dc2ce8bee5383816d52ffeba0bb34efc3109526485d485cbc07a590d701` |
| v20 adapter weights | `87b4d79a6a2278628f9b45f944c36ec9ca08c8e37a693949858dcad1aea9e0a5` |
| v20 adapter config | `f479515dbb623f9623fbde49c5fdc8ce468acedb447917b642b93bfbf6cebd35` |

The historical base was fetched as `Qwen/Qwen3-1.7B` without a recorded Hub
revision. That prevents exact base-weight replay and is another reason the old
packet remains historical rather than promotion evidence.

The historical harness supplied each arm only the user question. It supplied
no system message, Minecraft context, tool manifest, function schema, or
signature list. The executable loop could recognize and execute the custom
`lookup(...)` syntax if a model produced it, but the base was never told that
the syntax or oracle existed. The saved comparison therefore measures an
adapter that learned the custom lookup convention during training against an
unprompted base. It does not measure adapter skill beyond a base given the
same documented interface.

## Retraction

Do not cite any saved base-model `answer_ok` count as answer accuracy. The
stored instrument values were 51/106 on held-out and 6/43 on the
adversarial probe. A stricter offline matcher changes those to 48/106 and
3/43, but those replacements are also non-authoritative.

The historical harness generated at most 256 new tokens per model turn and
saved only cleaned text, not raw generations, original token IDs, EOS state,
or finish reason.
Retokenizing all saved text with the preserved Qwen tokenizer gives:

| Arm | Rows | Reconstructed tokens | Mean |
| --- | ---: | ---: | ---: |
| base / held-out | 106 | 252-254 | 253.9623 |
| base / adversarial | 43 | 253-254 | 253.9302 |
| v20 / held-out | 106 | 5-28 | 12.7925 |
| v20 / adversarial | 43 | 6-59 | 19.2791 |

All 149 cleaned base responses sit within four tokens of the configured cap;
twelve directly inspected rows end without a settled answer. This is strong
reconstruction evidence of total base cap exhaustion, though not preserved
raw finish metadata. Only a new run can measure the base answer outcome.

## Supported Tool-Use Result

The saved `tool_call_ok` field is narrower than strict protocol compliance.
Its parser accepted a `lookup(...)` match anywhere in generated text and saved
no actual call. Held-out rows checked the first recognized table and key. The
adversarial aggregate mixed three contracts:

- known rows: 27/33 v20 versus 0/33 base for recognized first table + key;
- unknown rows: the tool-call flag recorded 9/9 v20 versus 0/9 base for the
  expected table only--the key was
  neither checked nor retained;
- one usage row: 0/1 v20 versus 1/1 base on the no-call behavior check.

The defensible headline is therefore: on the saved held-out draw, the legacy
parser recognized the matching first table and key on 104/106 v20 rows versus
0/106 base rows; on adversarial known rows it recorded 27/33 versus 0/33.
This is evidence that the adapter contributed recognized lookup behavior. It
is not strict-envelope compliance, semantic answer accuracy, a stable
population rate, or a causal estimate unique to data verification.

"Held-out" here means that the evaluated keyed identities were absent from
the training traces. The lookup categories, question templates, protocol, and
underlying task semantics were not held out. The adversarial pool was mostly a
phrasing test: 35 of its 42 keyed identities appear in training. These are
identity-generalization and phrasing-robustness results, not evidence of a new
tool category or out-of-domain transfer.

Two additional adapter-only held-out runs recorded 104/106 and 103/106 under
the legacy parser. On the changed beehive row, the third run retained one
recognized call (`n_calls=1`) and then falsely declined. Because the saved
packet does not retain the call payload, it cannot show which table or key was
called. It is incorrect to describe this row as having skipped the tool.

## Replacement Gate

The prospective evaluator supplies both arms the same explicit Minecraft tool
manifest as system context. It validates the manifest version and all 11
signatures against the executable parser, requires the operator to predeclare
its raw-byte SHA-256 for evidence output, injects the same loaded bytes into
both arms, and stores those exact bytes and their hash in the packet. This is a
new prompt condition, not a direct replay of the historical no-manifest rates.

The evaluator retains every raw turn, exact rendered-prompt hash, and exact
`{table, argument, key}` call; requires one complete tool-call turn; rejects
malformed attempts; authenticates unknown keys and both expected multihop
calls; isolates the post-reasoning answer span; records EOS/cap state; and
makes incomplete answers non-evaluable. It uses stable per-question seeds, a
4,096-token per-turn cap, exact model revision selection, before/after input
hashes, local hashed adapter bytes, and atomic output publication. CUDA runs
base first, attaches PEFT only after the base results materialize, and writes a
paired packet only after row order, seed, and initial-prompt hashes match.
Matched seeds are a common-random-number control, not a claim of bitwise CUDA
determinism.

Before reporting a replacement answer rate, merge that instrument, rerun
matched base and v20 arms, and calibrate its mechanical answer matcher against
a blinded human-adjudicated sample. Unit tests and corpus-format replay verify
the instrument contract; they do not validate semantic correctness.
