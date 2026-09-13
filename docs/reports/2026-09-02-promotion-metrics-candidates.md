# Metrics and benchmarks to promote the pipeline — candidate list

Draft. Not a committed publishing plan — a working list of candidate metrics,
prioritized after cross-session review. Private repo only; nothing here is
scrubbed for public consumption.

Structure mirrors `2026-09-02-benchmark-claim-ledger.md` on purpose: what's
in, what's deferred, and what's explicitly excluded and why, all stated up
front rather than discovered by a reader. The pitch is stronger for what it
declines to claim, not just what it proves — keep that visible in the doc's
shape, not just its content.

## Tier 1 — have data now (2026-09-02 benchmark run, see
`2026-09-02-benchmark-claim-ledger.md`)

- **tool_call correctness**, base vs. v20, held-out + adversarial split.
  0%/2% (base) vs. 98%/83% (v20). Sharpest headline number — base model
  with tool access essentially never emits a working tool call at all.
- **answer correctness**, same split. 48%/13% (base) vs. 96%/86% (v20).
- **fine-tuning-vs-tool-access gap** (the prompted-base-plus-tool control).
  Closed as of this run — licenses the "fine-tuning contributes capability
  beyond tool access alone" claim per the pre-registered ledger criteria.

## Tier 2 — cheap, reuses existing eval harness, ranked by priority

1. **Fabrication rate and admission rate as standalone headline numbers**
   (top priority). Both are instantly legible to a reader with zero context
   on the harness — "fabrication rate: X%" needs no explanation to land.
   Data already exists (`unknown-subject declines: N/M` in probe output;
   generated-vs-admitted counts in `build_training_set.py`'s summary line);
   just needs to be pulled out and reported instead of left buried.
2. **Per-domain fake-item decline rate + adapter version-over-version trend
   (v16→v20)** — real value, but appendix/detail-page material rather than
   top-line numbers. For a reader already convinced who wants depth, not
   for the initial pitch.
3. **Rephrasing robustness** — lower priority than it first looks. The
   adversarial-probe numbers already carry this signal implicitly (the
   2%→83% / 13%→86% tool-call/answer gap under harder phrasing IS the
   robustness claim). A standalone "robustness" stat restating the same
   thing risks reading as padding rather than new information — fold it
   into the existing adversarial-set framing rather than mint a separate
   headline number for it.
4. **Usage-question decline correctness** — real but currently 1-item
   sample size in the probe set; expand the probe set before citing this
   as anything beyond an appendix note.

## Tier 3 — needs new harness work, ranked by priority

1. **Unseen-item generalization** (top priority for this tier). This is the
   sharpest objection a skeptical reader has in reserve — "did it just
   memorize the training distribution's item shapes?" — and there is
   currently no answer to it. Needs a held-out split excluded from the
   *item universe*, not just excluded question rows (i.e. items with zero
   training exposure, not just unseen phrasings of seen items). Better to
   find a real gap here internally than have a reader find it first.
2. **Eval-blind-spot honesty metric** — count of bugs a clean automated
   eval missed that a live player caught (the AND/OR bug, the
   breeding_food accuracy bug). INCLUDE, not deferred to "later, if there's
   time" — this is the same discipline the claim ledger runs on results,
   applied to the eval process itself: a promotion doc that quietly drops
   the one uncomfortable metric is the exact failure that discipline exists
   to signal it doesn't commit. Doesn't need the full harness to start —
   ship a qualitative list of known blind spots now (currently two data
   points, from the case study), formalize a running count later. Don't
   let harness-completeness gate publishing something already statable
   honestly today.
3. **Multihop chain re-scoring and AND/OR structural fidelity** —
   re-scoped: these are pipeline/data-integrity checks, not model-
   capability claims, so they don't belong in THIS document. Route to an
   internal reliability doc instead. A promotion audience is judging the
   model; these judge the harness's own internal consistency, which is a
   different (and legitimate) but separate document's job.
4. **Inference latency per query** — lowest priority for this doc
   specifically. It's an adoption/ops question ("is this practical to
   run"), not a "does fine-tuning matter" argument. Belongs with a
   deployment guide, not the capability pitch.

## External / comparative — unresolved, exploratory

- **Plancraft** (arXiv 2412.21033, COLM 2025) — Minecraft planning-agent
  benchmark, open-source code at `github.com/gautierdag/plancraft`. Fit is
  UNRESOLVED and no session has an independent read yet: it tests broader
  planning + tool-use + RAG against a handcrafted planner baseline, not
  specifically fact-knowledge-declining the way this project's thesis is
  framed. Whether its "intentionally unsolvable examples" mechanism maps
  onto this project's decline-on-fake-item behavior needs a primary-source
  read (the actual paper or repo, not name-recognition or an abstract
  skim) before it's treated as a real external comparison point. If it
  matters for the promotion timeline, that read needs to happen before a
  decision, not be reasoned about secondhand.
- **Zero-shot bigger-model sanity check** — same task, no fine-tuning, no
  tool access, a larger general-purpose model. Establishes an upper bound
  on "does a bigger model just know this from pretraining" — useful as a
  ceiling reference, not a competitor claim (different size class,
  different cost, not an apples-to-apples pipeline comparison).

## Explicitly not a metric worth chasing, and why

- **Raw parameter-count or FLOPs comparisons against frontier models** —
  wrong comparison class, invites a straw-man rebuttal ("of course a 1.7B
  model loses to GPT-5 on general capability") instead of the actual claim
  (verified small-corpus fine-tuning beats tool-access-only prompting on
  the SAME small model).
- **Multihop/AND-OR data-integrity metrics, in THIS document** — not
  excluded from measurement, just excluded from the promotion pitch
  specifically; see Tier 3 item 3. Real engineering value, wrong audience.
- **Inference latency, as a capability claim** — not excluded from being
  measured eventually, just excluded from ranking alongside capability
  numbers; see Tier 3 item 4. Conflating "fast" with "correct" would blur
  the actual claim being made.

## Failure taxonomy — v20, run against existing eval outputs (2026-09-02)

Per cross-session review: read what v20 actually gets wrong instead of only
citing the aggregate adversarial-probe rate (14% answer-fail / 17% tool-call-
fail on the probe set). Source: per-item results already present in
`held_out_v20.json` / `probe_v20.json` (category/table/tool_call_ok/
answer_ok fields), no new eval run needed.

- **Held-out (4 fails / 106)**: 2 recipe, 2 loot. Scattered, no cluster.
- **Adversarial-probe (8 fails / 43)**: item_use 3, tag 2, recipe/painting/
  unmatched 1 each. `item_use` is a concentrated failure family -- 3 of 8
  probe failures, and it does not appear at all among the held-out
  failures, meaning it specifically breaks under adversarial phrasing, not
  in general.

Reportable finding: not "14-17% noise" -- specifically, item_use questions
(equip/consume-context) are where v20 actually breaks under harder
phrasing. Cite the concentrated failure, not just the aggregate rate.

## Additional candidates raised in review, not yet started

Ranked by engineering cost, cheapest first:

1. **Failure taxonomy** -- done, see above.
2. **Live red-teaming as a first-class arm** (not folded into the honesty
   metric as blind-spot detection -- a human trying to break the agent in a
   real session, reported as its own channel). Cheap engineering-side (a
   reporting template, no new code); the actual red-teaming is human time,
   not a build task.
3. **Cross-domain transfer, in-context variant**: define a novel tool in a
   system prompt (same call-format delimiters as training, different
   domain, e.g. a toy weather/calculator tool) and test whether v20 forms
   correct calls from the schema alone, no retraining. Isolates "learned
   the calling convention generally" from "memorized Minecraft's specific
   ontology." A few hours against the existing harness, no new
   environment/verifier infra. Proposed as the first cut before any full
   cross-environment version.
   - **Full cross-domain transfer** (a genuinely different tool API/
     environment, e.g. standing up an existing benchmark like BFCL) is the
     sharpest possible test of the actual headline claim, but real cost
     (days, not hours) and has a confound to watch: a naive foreign-schema
     run can't distinguish "no general tool-use skill" from "right skill,
     wrong syntax dialect" without the in-context variant run first as a
     control.
4. **Blind third-party replication** -- checked: no merge/GGUF/export path
   exists yet in `tool_oracle_cuda/` (grepped for `merge_and_unload` /
   `save_pretrained`, no matches). This is blocked on unstarted export
   engineering, not just on finding a disinterested third party. Most
   expensive of the four candidates; sequence last, no committed owner yet.
