# Benchmark claim ledger — frozen 2026-09-02, before any arm reports

Written before reading `held_out_base.json`. Purpose: prevent post-hoc claim
selection once numbers exist. Do not amend after seeing results except to add
new arms with their own pre-registered criteria.

## Pinned state (SHAs, before any result is read)

- mcAgent worktree HEAD: `b8f3f6eb14ae2d2f8488b865680089a8c992ded4`
  (branch `worktree-fact-pipeline-generator-oracle`)
- `fact_pipeline/minecraft.db` md5: `68e60d4f7d846749034a0819e5c2bc04`
- `fact_pipeline/tool_oracle_cuda/train_v20_action.jsonl` md5:
  `4545401208d7b0ef8623e3f545beaf5d`
- `fact_pipeline/tool_oracle/robustness_probe_full.jsonl` md5:
  `87a1f5d5a61326ddc91471c14f11fcf2`
- Adapter under test (fine-tuned arm): `adapter-tool-v20-cuda`
- Base model (control arm): `Qwen/Qwen3-1.7B`, no adapter, same tool/oracle
  access, same eval harness (`eval_tool_skill_cuda.py`)

## Primary contrast

Fine-tuned (v20 adapter) vs. prompted-base (same base model, no adapter,
identical tool access) — NOT fine-tuned vs. no-tools. Isolates whether the
adapter's training contributes capability beyond what tool access alone gives
a prompted base model.

## Planned arms (all four, declared before any run completes)

1. Base, held-out set, `--held-out --train-jsonl train_v20_action.jsonl
   --n-per 50` → RUNNING (PID 1037145 at freeze time)
2. Base, adversarial probe, `--probe robustness_probe_full.jsonl` → NOT YET
   STARTED
3. v20 adapter, held-out set, same command + `--adapter adapter-tool-v20-cuda`
   → NOT YET STARTED (current v20 number, not the stale round-11/case-study
   100%/88% figures — those are a different adapter lineage predating the
   recipe AND/OR + loot_source/ore_depth fixes this session made)
4. v20 adapter, adversarial probe → NOT YET STARTED

All four must be run and reported together. No arm gets reported alone as if
it were the whole picture.

## What result licenses which claim

- If v20 beats base by a wide margin on BOTH held-out and adversarial-probe →
  licenses "fine-tuning on the verified corpus contributes capability beyond
  tool access alone."
- If v20 beats base only on held-out but not adversarial-probe → licenses a
  narrower claim: fine-tuning improves recall of trained-distribution facts,
  not robustness/generalization; adversarial-probe gap is tool-access-driven,
  not architecture-driven.
- If base matches or nears v20 on both → the paper does NOT get to claim a
  capability gain from the fine-tuning/architecture. Per review: describe the observed system and its artifacts, not a causal
  capability gain, until this arm closes.

## What would undercut or narrow the claim

- Base scoring close to v20 on held-out specifically (would mean the
  held-out set is answerable from tool access + base-model general
  knowledge, not requiring the trained corpus).
- Any harness/prompting asymmetry between the base and adapter runs (e.g.
  different system prompt, different tool-call format expectations) that
  advantages one arm — must be ruled out, not just assumed absent, before
  citing a gap as capability.

## Exclusions decided now, not after seeing results

- Any run that errors, times out, or produces a malformed `--out` JSON is
  EXCLUDED from the headline number but MUST be recorded in the attempt
  census below with the failure reason — never silently dropped.
- No cherry-picking n-per or subset size after seeing a result; `--n-per 50`
  is fixed for all four arms.

## Attempt census (append every attempt, success or failure)

| # | Arm | Started | Result | Notes |
|---|-----|---------|--------|-------|
| 1 | base / held-out | 2026-09-02 (prior turn) | pending | PID 1037145, log `eval_base_held_out.log` |

(Rows to be appended as arms 2–4 run; failed/invalid launches get a row too.)

## Note (post-arm-2, non-criteria)

Arm 2 (base/adversarial-probe) landed at 6/43 (13%). This numerically matches
the round-2 robustness-collapse figure (97%->13%) in the 2026-08-16 case
study, which is a DIFFERENT model (early fine-tuned adapter, not base) and a
DIFFERENT finding, with no actual connection between the two. Flagging here
so the writeup disambiguates wherever both numbers appear near each other —
e.g. 'the base-model control's 13%' vs. 'the round-2 collapse's 13%'. Does
not touch any pre-registered criterion above.

## Results (all four arms, filled in only after all four completed)

| Arm | tool_call correct | answer correct |
|---|---|---|
| base / held-out | 0/106 (0%) | 51/106 (48%) |
| base / adversarial-probe | 1/43 (2%) | 6/43 (13%) |
| v20 / held-out | 104/106 (98%) | 102/106 (96%) |
| v20 / adversarial-probe | 36/43 (83%) | 37/43 (86%) |

**Verdict per the pre-registered criteria above**: v20 beats base by a wide
margin on BOTH held-out and adversarial-probe (tool_call correctness in
particular: 0%/2% base vs 98%/83% v20 -- the base model essentially never
emits a correct tool call at all, with or without an adversarial phrasing).
This licenses the wide claim: fine-tuning on the verified corpus contributes
capability beyond tool access alone. Tool access alone (prompted base, same
harness) does not confer working tool-calling behavior for this model size.

Raw outputs: held_out_base.json, probe_base.json, held_out_v20.json,
probe_v20.json (all under tool_oracle_cuda/, gitignored disposable
artifacts -- referenced by path + the SHAs/hashes pinned above, not
committed).

## Attempt census (final)

| # | Arm | Result | Notes |
|---|-----|--------|-------|
| 1 | base / held-out | 51/106 answer, 0/106 tool_call | PID 1037145, clean |
| 2 | base / adversarial-probe | 6/43 answer, 1/43 tool_call | clean |
| 3 | v20 / held-out | 102/106 answer, 104/106 tool_call | clean |
| 4 | v20 / adversarial-probe | 37/43 answer, 36/43 tool_call | clean |

No excluded/invalid runs this round -- all four launches completed cleanly
on first attempt.

## Harness negative control (2026-09-02, per a reviewer's fleet-operation critique)

Per review: an eval only ever pointed at a good model has never actually
demonstrated it CAN fail. Ran two synthetic stress tests directly against
`_score()` in `tool_oracle/eval_tool_skill.py`, offline, using the real
held_out_v20.json + probe_v20.json items (149 total) as the base corpus. No
new GPU run -- this exercises the scorer, not the model.

**Test 1 -- garbage/unrelated answer text.** For each of the first 60 items
with expected data, scored 5 obviously-wrong answers ("banana", "I like
turtles", "", "42", "the sky is blue today") against that item's real
`expected`. 300 checks, **0 wrongly passed**. The scorer is not universally
vacuous -- it does reject content with zero relation to the expected value.

**Test 2 -- shuffled ground truth.** Paired each item's own (real, correct)
v20 answer text with a DIFFERENT item's `expected` (offset-by-one shuffle).
139 checks, **7 wrongly passed** -- i.e. 7 cases where a correct answer to
item A was scored as satisfying item B's expected value. All 7 share one
mechanism: `_score()` checks token/value membership, not subject identity.
It passes when two different subjects happen to share the same scalar
expected value, or when one item's full expected set is a subset of
another's answer text:

- Two different enchantments sharing `max_level=1` (multishot/mending),
  `max_level=1` again (mending/channeling), `max_level=5`
  (sharpness/efficiency) -- 3 of 7.
- Two villager trades with identical want/give quantities but different
  item names not distinguished by the scalar check -- 1 of 7 (the
  `wants=N minecraft:X / gives=N minecraft:Y` template only checks the
  string forms present, and a near-miss can pass if the surrounding prose
  differs but the checked tokens coincide).
- One recipe's full ingredient list containing another recipe's single
  ingredient as a subset (piston's 4-ingredient answer contains "iron
  ingot," which is shears' entire single-ingredient expected value) -- 1 of
  7.
- Two loot items with overlapping short item-name tokens -- 1 of 7.
- One cross-category false pair (villager_trade answer scored against a
  painting's width/height expected) that only passed due to a scoring-code
  quirk in this synthetic shuffle, not a realistic failure mode (a
  villager-trade answer would never actually be shown against a painting
  question in the live harness) -- flagged separately as an artifact of
  the shuffle test itself, not a live risk.

**What this means:** the scorer has a real, narrow blind spot -- it is not
subject-identity-aware for templated scalar/quantity categories
(enchantment max_level, villager_trade wants/gives) and for recipe
ingredient-subset overlaps. It is NOT vacuous in general (test 1 is clean).
This is a legitimate harness-improvement item, distinct from the model
capability numbers: the 98%/96% v20 figures are not invalidated by this
(they were scored against the CORRECT item/answer pairing, not shuffled
ones), but the scorer's true false-positive rate on live model output is
not exactly zero, and is currently unmeasured. Recommend: add a subject-
identity check to `_score()` for scalar/quantity categories before citing
"answer correct" as a hard number in any external-facing material, or
caveat it as "token-membership scored, not subject-verified" until fixed.

## Artifact identity -- checkpoint hash for the 2026-09-02 benchmark numbers

Hashed the exact adapter files the harness loaded to produce the 98%/96%/
83%/86% v20 numbers in this ledger, so any later citation can be checked
against the checkpoint actually evaluated, not assumed:

- `tool_oracle_cuda/adapter-tool-v20-cuda/adapter_model.safetensors`
  sha256: `87b4d79a6a2278628f9b45f944c36ec9ca08c8e37a693949858dcad1aea9e0a5`
- `tool_oracle_cuda/adapter-tool-v20-cuda/adapter_config.json`
  sha256: `f479515dbb623f9623fbde49c5fdc8ce468acedb447917b642b93bfbf6cebd35`

Base model: `Qwen/Qwen3-1.7B` (HF Hub, unauthenticated pull at eval time --
no local pin; a Substack citation should note this is the public HF weight,
not a locally-frozen copy). If a headline number ends up on Substack,
whoever publishes it should re-hash the adapter referenced from that page
against the two hashes above before it goes live, and flag a mismatch
rather than assume continuity. The general case for this (a filename is
not proof of file identity over time) stands on its own regardless of any
specific incident.

**Correction (2026-09-02, post-commit):** this section originally cited a
specific precedent -- an adapter believed destroyed and recovered intact on
an unexpected machine -- attributed to a reviewer. On follow-up, the reviewer
traced their own sourcing and found the load-bearing detail (that the
headline number came from v11 specifically) was relayed secondhand, not
independent knowledge, and the one document they could check
(2026-08-16-minecraft-domain-case-study.md, 559 lines)
never mentions v11 -- the highest adapter named there is v9. The incident
may still be real and simply postdate that document, but as of this
correction no reviewer can point to a source for it. Downgraded from
"cited precedent" to "unverified anecdote" -- not used as evidence for
the hashing recommendation above, which holds on its own logic.

## Follow-up: verbosity-inflation check and real (non-shuffled) collision scope (2026-09-02)

Per further review: does the scorer's substring/membership check reward
verbosity, and does that explain some of the v20-vs-base delta rather than
capability? Checked directly against the saved raw outputs, both arms.

**Answer length vs. pass rate, base and v20, both arms (control included):**

| Arm | n | mean answer length (chars) | pass rate | mean length, passing | mean length, failing | corr(length, pass) |
|---|---|---|---|---|---|---|
| base / held-out | 106 | 1110 | 48% | 1101 | 1118 | -0.16 |
| base / probe | 43 | 1130 | 14% | 1140 | 1129 | +0.06 |
| v20 / held-out | 106 | 47 | 96% | 45 | 84 | -0.43 |
| v20 / probe | 43 | 73 | 86% | 72 | 81 | -0.07 |

Verbosity-inflation hypothesis does not survive this check, and fails in the
opposite direction from what it predicts: v20's answers are dramatically
SHORTER than base's (47-73 chars vs. 1100+), not longer, and within v20's
own output, longer answers correlate with FAILURE, not spurious pass
(-0.43 on held-out). Base is the verbose arm and it has the lower pass
rate. The delta is not explained by the scorer rewarding length.

**Why "which scoring mechanism fired" isn't a loggable distinction here:**
checked `_score()` directly -- every known-item category (enchantment,
villager_trade, item_use, tag, recipe, loot, ...) resolves through the same
substring/token-membership check; there is no separate strict-exact-match
path in the code to distinguish. The uniform mechanism is exactly what the
shuffle test in the prior section already characterizes -- the length check
above is the more direct test of whether that mechanism differentially
favors v20, and it says no.

**Real (non-shuffled) collision scope, not a synthetic worst case:**
grouped the actual 149 eval items (held-out + probe combined) by expected-
value signature within enchantment/villager_trade/recipe. 3 real ambiguity
groups exist (excluding decline/fake items, which score via a separate
no-fabrication path, not `_score()`): `iron_ingot`-only recipes (heavy
weighted pressure plate, shears), `max_level=1` enchantments (multishot,
mending, channeling), `max_level=5` enchantments (sharpness, efficiency)
-- 7 of 149 real items (4.7%) sit in a genuine ambiguity group, consistent
with (not exceeding) the ~5% floor the synthetic shuffle found.

Checked all 7 directly against their real (unshuffled) v20 output:
**all 7 scored answer_ok=True, and all 7 name their own subject explicitly**
in the answer text ("Multishot has a maximum level of 1.", "Sharpness has a
maximum level of 5.", etc.) -- the model's actual template puts the subject
name first, so even though `_score()` itself does not check subject
identity, none of the 7 real at-risk items in this run were false passes
in practice. Zero exploited collisions found in the actual reported
numbers, on direct inspection of every item that COULD have been exploited.

**Net read:** the collision blind spot in `_score()` is real and confirmed
(prior section), but on the real item population (not the synthetic
shuffle) it's a ~4.7% exposure surface, and a full audit of every exposed
item in this run shows 0/7 actually false-passed. The caveat language
should say the mechanism exists and bound it at "at least ~5% of items are
structurally exposed, 0 confirmed false-passes found on manual audit of
every exposed item this run" -- not "the numbers may be wrong," which this
check does not support, and not "the numbers are proven right," which no
single run can support either. Per the earlier scope note, the adversarial-
probe arm remains structurally more exposed than held-out (adversarial
items are built as near-misses, exactly the collision shape) even though
this run's manual audit found nothing exploited -- caveat it harder if
caveating asymmetrically.

## Follow-up: accidental vs. structural safety, and a real fix (2026-09-02)

Per further review: "all 7 name their own subject" was accidental safety
(a model habit `_score()` doesn't enforce), not structural safety (a
property the scorer guarantees). Nothing in the harness required v20's
template to include the subject name; a future retrain could drop it and
the scorer would keep returning clean while the true false-pass rate rose
silently. Correct to not let the audit read as "the gap is closed."

**Fix applied**: added a subject-identity check to `_score()`'s scalar/
structured branch (`tool_oracle/eval_tool_skill.py`), requiring the item's
own bare name to appear in the answer, for items with a simple (non-
compound) key -- covers `enchantment`, `jukebox_song`, `painting`,
`item_use`. `villager_trade` keys are compound paths (`slot/tier/trade`)
with no single name to check, so intentionally left out of this check;
that category was not part of the confirmed real-collision set anyway (see
prior section).

**Verified before merging anything into the reported numbers**:
- Re-scored all four arms' saved raw outputs (known-kind items only --
  fake/decline items score via the separate `_score_unknown()` path, not
  `_score()`, confirmed by checking `kind` field) against the patched
  function: **0 flips across all 149+149 known-item scorings, both arms**.
  The fix changes nothing about the reported 98%/96%/83%/86% numbers. Read
  this as evidence the patch is SAFE to add, not evidence it was
  UNNECESSARY -- it's currently inert on real output precisely because
  v20 happens to name its subjects (the accidental protection this
  started from). It earns its keep on the next model, not necessarily
  this one; it is not dead code.
- Test suite: `pytest tests/test_tool_oracle.py` -- 12/12 pass, unaffected.
- Re-ran the shuffled-ground-truth negative control from the prior section
  with the patched scorer: **7 wrongly-passed pairs -> 3**. The 3
  remaining (loot item-name substring overlap, villager_trade compound-key
  collision, recipe ingredient-subset-answer) go through the scorer's
  generic fallback path, a different code path this fix does not touch --
  documented as a known remaining gap, not silently left unaddressed.

**Net effect**: the enchantment scalar-value collision class (3 of the
original 7 real at-risk items) now has structural protection -- the
scorer itself verifies subject identity, not just this run's model
phrasing. The remaining 3 real-collision items (loot, villager_trade,
recipe-subset) are a distinct mechanism, unfixed, and would need a
separate, more careful change (exact-set vs. subset-membership checking
for recipe/loot; villager_trade has no cheap subject-name check available
given its compound key). Not attempted this session -- flagged as a
follow-up, not implied-fixed.

**Two reframings from review, kept as stated rather than dropped as
nulls**:
- The 4.7% (7/149) figure is an *affirmative structural bound*, not just
  this run's observed rate -- only 3 genuine ambiguity groups exist in the
  eval set's real item population, independent of what any model does.
  That's a more durable claim than "zero this run."
- The -0.43 length-vs-pass correlation within v20 (prior section) may be
  signal, not noise: consistent with the model getting more verbose when
  uncertain, which would make answer length a usable uncertainty proxy.
  Not tested further this session -- recorded as an observation worth a
  follow-up, not a validated claim.

## Follow-up: CI regression test, and bounding the remaining 3 (2026-09-02)

Per further review: a one-off audit that finds a bug and confirms a fix
doesn't stop the bug from returning -- the only thing that ever caught the
enchantment collision was a manual run someone happened to do once. Wired
it into CI instead.

**Regression test added**: `tests/test_tool_oracle.py::
test_score_subject_identity_regression`. Pins the fixed enchantment case
(two enchantments sharing `max_level=1`; a correctly-named answer for one
must not pass the other -- would have wrongly passed before the fix) and
pins the three still-open cases (recipe ingredient-subset, loot substring
overlap, villager_trade compound-key) at their CURRENT behavior, so a
future change to any of them shows up as a test diff here, not as a
rediscovery. 13/13 tests pass with the new one included. If the pinned
"still open" assertions ever start failing, that means one of the three
remaining mechanisms got fixed -- update the test (and this doc) rather
than treating it as a break.

**Remaining 3, bounded the same way the enchantment class was bounded**
(affirmative structural count on the real 149-item eval population, not
just this run's observed rate):

- **recipe/loot generic-path collision** (exact-signature match + proper-
  subset containment, the mechanism behind both the piston/shears and
  bamboo cases): **7/149 items (4.7%)** exposed -- 2 exact-signature pairs
  (`iron_ingot`-only recipes; `bamboo`-only loot) plus 13 subset-
  containment pairs, covering 7 distinct items. Unfixed by this session's
  patch (different code path -- the generic fallback, not the scalar
  branch). **Correction below: this 7 overlaps 2 items with the fixed
  enchantment class's 7 (both scans scoped `recipe`) -- see the
  "Correction" section for the true, non-double-counted union.**
- **villager_trade compound-key collision**: **0/24** real villager_trade
  items in this eval set share an exact expected-value signature with
  another. The mechanism the shuffle test found (near-name substring
  overlap, e.g. "white terracotta" inside "white glazed terracotta") is
  real and the category has no subject-identity check at all, but it did
  not materialize as an exact-signature collision anywhere in this
  specific 24-item population. Exposure would require two real trades
  with genuinely overlapping item-name substrings, which happens not to
  occur in this set -- a narrower bound than "unmeasured," but not "zero
  risk," since the DB's full trade table is larger than this eval sample.

Total remaining real exposure across both unfixed mechanisms: 7/149 (4.7%)
confirmed via recipe/loot, plus an unquantified-but-not-observed-here risk
in villager_trade. Not fixed this session -- recipe/loot needs an exact-
set-vs-subset scoring change (risk of new false negatives on legitimately
paraphrased answers, so not attempted without more care); villager_trade
has no cheap subject-identity check available given its compound key.

## Correction: the two "7"s overlap, and "same magnitude" was an
overclaim (2026-09-02, post-commit)

Per further review: before asserting "same magnitude as the fixed class"
anywhere else, checked whether the fixed-class 7 (enchantment + recipe,
exact-signature scan) and the remaining-gap 7 (recipe + loot, exact +
subset scan) are the same population counted twice, since both scans
included the `recipe` table.

**They are not disjoint.** Set-intersected the two item lists directly:
`minecraft:shears` and `minecraft:heavy_weighted_pressure_plate` appear in
BOTH sets -- caught by the recipe-table exact-signature check in the first
scan and again by the recipe/loot scan in the second, because both scans
scoped `recipe` in. That is double-counting from overlapping measurement
scope, not two independent domain mechanisms converging on the same rate.
"Same magnitude, 7 and 7" is wrong as stated; the correct **union** across
every mechanism found (enchantment scalar collision, now fixed; recipe/
loot generic-path collision, still open) is **12/149 items (8.05%)**, not
14/149 and not two separate 4.7% figures.

**Is even 12/149 more than noise, or is one measurement enough to assert
a domain-wide rate?** It isn't enough alone -- a single 12/149 draw has a
wide interval (~95% CI roughly 3.7%-12.4%, normal approximation). Checked
the cheapest available replication using data already in hand: split the
union-12 set across the two item pools already in this eval -- held-out
(9/106 = 8.5%) and adversarial-probe (4/43 = 9.3%). The two land close
together, which is weak supportive evidence a real domain-level rate
exists, but two small-n draws landing in the same wide band is close to
unavoidable whether or not a shared constant is real -- this does NOT
license asserting "structurally ~1-in-20" as a settled ledger claim.

**Correction (2026-09-02, caught by an independent recompute pass):**
9 + 4 = 13, not 12, and the split above originally implied held-out and
probe partition the eval set by ITEM the way they partition it by ROW
(106+43=149 examples). They don't: `minecraft:mending` is asked about in
both pools as two separate questions, so it's counted once in the
held-out hit-count (9) and once in the probe hit-count (4), inflating the
naive per-pool sum past the true distinct-item union. The union figure
itself (12/149, computed directly as a set union over item keys, not by
summing the two pool counts) was and remains correct -- verified again
directly against the raw JSON. This is a different mistake from the
recipe-table double-count corrected earlier in this document: that one
was two SCANS overlapping in measurement scope; this one is two POOLS
overlapping in item content. Both are the same underlying discipline
failure (assuming disjointness instead of checking it), caught the same
way both times -- an independent recompute that treated the split as
a claim to verify, not a fact to restate.

**Recorded as a hypothesis with a stated test, not an assertion**:
Hypothesis -- roughly 8% of items in this domain sit in some value-
collision-prone family (scalar or set/subset), independent of any
specific model. Test that would settle it -- the rate should replicate
near ~8% on a genuinely independent item population (a future eval-set
redraw, or DB-wide enumeration rather than just the 149 items used in
this run's held-out+probe sets). Held-out-vs-probe landing close (8.5%
vs. 9.3%) is a first, weak data point in favor, not confirmation.

## Follow-up: load-conditioned and repeat-rollout variance (2026-09-02)

Per further review (a reviewer's other three angles): gave a real feasibility
read rather than an assumed one, and ran the two that were actually cheap
against the real machines/harness.

**Environment-conditioned variance (multi-host)**: checked directly,
not assumed. a second CUDA host (2x RTX 5060 Ti) is reachable, but no mcAgent
checkout at all -- needs a fresh clone, venv, and model/adapter/DB
transfer before anything can run. an Apple M4 host is reachable, same
gap, plus it would need an MPS code path since eval_tool_skill_cuda.py is
CUDA-specific. Genuinely not cheap. Deferred, not silently dropped.

**Load-conditioned results**: ran the real thing instead of speculating.
the eval host was under real CPU contention at the time -- two unrelated
CI runners at ~60% each across 28 cores, GPU idle (0%
util). Re-ran v20's held-out eval under that exact load and diffed every
item against the original idle run: **0 differences, byte-identical
answer text on all 106 items**. Load (at least CPU load with the GPU
otherwise idle) does not change this eval's output.

**Repeat-rollout variance**: the load-conditioned run's 0-diff result was
itself a first repeat, so ran a third pass to separate "is this
deterministic" from "did I get lucky twice." It was not fully
deterministic: **1/106 items differed** (run1/run2 agreed; run3 differed).
The item: `minecraft:blocks/beehive`. Run1/run2 correctly called the loot
tool and answered "Beehive drops: beehive." (answer_ok=True,
tool_call_ok=True). Run3 skipped the tool call entirely and answered "I
don't have any data on the beehive - it isn't in the loot table and isn't
a real Minecraft block/tag." (answer_ok=False, tool_call_ok=False) --
a **false decline**, not a benign phrasing difference: the model
confidently asserted a real, correctly-answerable item doesn't exist.
Run3's aggregate: 103/106 (97%) tool_call, 101/106 (95%) answer -- a real
±1-2 point swing from the reported 104/106 (98%) / 102/106 (96%), from
sampling variance alone (`do_sample=True, temperature=0.2, top_p=0.9` in
`eval_tool_skill_cuda.py`, no seed set anywhere in the harness).

**Read**: the headline 98%/96% numbers are a single draw from a
distribution with a small amount of real run-to-run spread (this sample:
95-98% depending on draw), not a fixed constant -- worth stating as a
range or noting "one held-out draw" rather than a bare point estimate if
this number goes on the public page. The specific failure mode found
(confidently declaring a real item nonexistent, rather than a near-miss
or a vague answer) is also worth a line on its own: it's the kind of
error a user would notice and distrust immediately, distinct from a
recall miss. Sample size (1 flip across 2 comparisons on 106 items) is far
too small to characterize the failure family beyond "exists, rare, and
concentrated enough to be worth watching for" -- not enough to say
whether beehive specifically is a recurring weak point or one draw's
noise.

## Follow-up: verified the loot-subset items' live scoring, for the second post (2026-09-02)

Per drafting the second Substack post: the claim "every exposed item
scored correctly anyway" had only been directly audited for the original
7 (mostly enchantment + the 2 recipe items) before the double-count
correction. Checked the 5 items unique to the corrected 12-item union
(the loot exact+subset additions: acacia_sapling, andesite, azalea,
bamboo, bamboo_sapling) against their real (non-shuffled) v20 output:
**all 5 scored answer_ok=True and all 5 name their own subject**
("Acacia sapling drops: acacia sapling.", etc.). The "every exposed item
scored correctly, because the model names its own subject" claim holds
across the full corrected 12-item union, not just the original 7 -- this
was verified now, not assumed from the earlier, narrower audit.

## Citable pre-registration timestamp (for the second post's "criteria fixed
before either arm was run" claim)

Per second-post review: the pre-registration claim was asserted but not
independently checkable from artifacts (numbers/hashes don't prove
chronology). Git history does: this document was first committed at
`48d0a5b` (2026-09-02T17:36:02Z), containing the four planned arms, the
primary contrast, and the exclusion/failure criteria, BEFORE any arm's
result was recorded -- the doc's own "Attempt census" at that commit lists
arm 1 as "pending." The commit that first records all four arms' actual
results is `9704d83` (2026-09-02T17:47:06Z), ~11 minutes later. Anyone
with repo access can verify this directly: `git log --format='%H %ad %s'
--date=iso-strict -- docs/reports/2026-09-02-benchmark-claim-ledger.md`,
or view `48d0a5b` and `9704d83` on GitHub and compare their diffs and
timestamps. This is the citable artifact behind the "criteria fixed
before either arm was run" claim -- link `48d0a5b` directly if this needs
to be checkable from the post rather than asserted.

## Operating rule: validate the instrument before you freeze it (2026-09-02)

Surfaced while scoping pre-registration for the cross-domain transfer test,
worth stating explicitly rather than leaving implicit in how this ledger
was run: **a freeze only protects a result if the thing being frozen
already works.** Pre-registering criteria against a harness that hasn't
been dry-run yet doesn't just risk a null result -- it risks freezing a
mechanical bug (a parser that doesn't match a new function name, a format
mismatch, an off-by-one in a schema) and then either reporting a false
negative caused by the bug, or "fixing" the bug after seeing bad output,
which is iterating after the freeze in every way that matters even if the
stated criteria never change on paper.

The fix is a real, separate pilot phase: build and dry-run the harness on
throwaway items first, confirm the mechanics actually work end to end,
THEN freeze the real item set and register the confirmatory test. The
pilot's own output is explicitly not evidence for or against the
hypothesis -- it exists only to validate that the measurement apparatus
functions, the same way this session's negative-control work asked
"can the scorer ever report failure" before trusting anything it reported
as a pass. Skipping the pilot and registering directly against an
unvalidated harness would be a subtler version of the same failure this
project has spent this session catching in other forms: a clean-looking
result whose cleanliness comes from the instrument, not the finding.

Applies going forward to any new eval arm added to this project (the
cross-domain transfer test currently scoped is the first case), and is
recorded here because it's operational eval practice specific to how this
project runs its own audits, not a general claim that belongs anywhere
else.

## Refinement: pilot/freeze disclosure needs more than item-disjointness (2026-09-02)

Follow-up to the validate-before-you-freeze rule above, worked out jointly
while scoping the cross-domain transfer test's pre-registration. Splitting
the pilot into an infra smoke test (re-run against the already-published
Minecraft held-out set, zero new items) and a schema pilot (a disjoint
throwaway toy domain, never reused) closes the largest leak -- a model
transferring intuitions from real-but-excluded domain items -- but item
disjointness alone is not a complete disclosure. Three residual leak paths,
identified in order of fixability:

1. **Structural leakage.** Even with zero item overlap, if the pilot
   domain and the frozen domain share template shape / parameter structure
   / response-length pattern, lessons learned piloting ("multi-word names
   need special handling," "responses run N tokens") can still shape how
   the frozen criteria get worded, without any item repeating. Fix:
   disclose that the two domains are structurally dissimilar too -- not
   just different vocabulary -- and describe the pilot domain's structure
   explicitly enough that a reader can judge for themselves whether
   transfer is plausible.
2. **Domain-selection-timing leakage.** If the frozen domain is one
   candidate among several and gets chosen (even informally) after seeing
   how piloting went, that's leakage at the selection step, not the
   wording step. Fix: state plainly whether the frozen domain was
   committed to before piloting began, or whether its selection method was
   pre-specified independent of pilot results. One sentence, cheap to
   state if true.
3. **Calibration-of-expectations leakage -- probably irreducible.** Even a
   fully disjoint, structurally dissimilar pilot shapes intuition about
   what pass rate to expect, and that intuition carries into criteria-
   writing whether or not anyone intends it to. No clean fix exists for
   this one -- it's the residual prior-knowledge problem every
   pre-registration scheme lives with. The honest move is naming it as a
   known, open limitation in the disclosure, not implying zero-overlap
   solved everything.

Net: "zero item overlap" is a real improvement, not a complete answer.
Complete disclosure language: "zero item overlap, structurally
dissimilar templates, frozen domain fixed before piloting began" -- with
leak #3 stated explicitly as an open, accepted limitation rather than
silently absorbed into "we piloted responsibly."

## Correction: the real test for prior-knowledge-carried-forward isn't
"old and public" (2026-09-02)

When asked whether the beehive false-decline finding (from the repeat-
rollout section above) could poison the transfer test's criteria the same
way an undisclosed pilot would, the first answer given here leaned on "the
items are old and public" as the reason it's safe. That's not actually the
right test, and the correct one is sharper: **does the methodology change
apply the same way regardless of what the frozen set's data turns out to
contain, or does it specifically anticipate what THIS data will show?**
Data-agnostic lessons ("this model class is unstable across repeated
samples, so require N repeats") are exactly what pre-registration is
supposed to let a researcher carry forward -- you're allowed to know your
instrument, you're not allowed to know your data. What would actually
poison a registration is a lesson narrow enough that knowing it tells you
something about items adjacent-in-kind to what the frozen set likely
contains, not just about the model class in general.

The beehive finding clears this test specifically because the transfer
test is genuinely cross-domain (Minecraft to an unrelated toy domain) --
a Minecraft-specific instability lesson doesn't narrow expectations about
a different domain's items. If a future "transfer" test were instead
another Minecraft-adjacent domain, the same adjacency-disclosure treatment
used for the pilot design above would apply here too, and the boundary
would get genuinely blurrier. Recording the correct general rule
(data-agnostic methodology vs. data-anticipating criteria) rather than
the weaker "it's old and public" justification, since the newness/
publicness was never actually what made it safe.
