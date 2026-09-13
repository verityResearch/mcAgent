# mcAgent → Shippable Embodied Minecraft Agent — Plan

Date: 2026-08-18
Status: **approved for execution** (operator go-ahead 2026-08-19). Originally proposal-only.
Author: design pass with the operator.

> **Historical claim boundary:** this plan predates the completion-aware v20 audit. Oracle
> returns are grounded, but a free-form model can skip or misuse a tool and can add unsupported
> prose. Read "impossible" claims below as design intent, not verified inference behavior; see
> [the measurement correction](../reports/2026-09-03-v20-evaluation-measurement-correction.md).
> Historical evaluator prompts also omitted the tool manifest, so their base
> arm was not exposed to the custom lookup interface.

---

## 1. The thesis, and why it is stronger for an embodied agent than for Q&A

**Voyager needed GPT-4 because the model had to *know* Minecraft. We can ship a 1.7B local model
because our model doesn't have to know — it queries a verified oracle.**

That sentence is the whole project. It is worth unpacking, because it is both the product wedge and
a real research claim.

**How the incumbents work.** Voyager (arXiv 2305.16291) is the canonical result: GPT-4 + Mineflayer
+ an ever-growing skill library, obtaining 3.1× more unique items and hitting tech-tree milestones
15.3× faster than prior SOTA. Mindcraft is the maintained open-source descendant. Both work by
prompting a frontier model that carries Minecraft knowledge *from pretraining*, having it write
executable JavaScript, running it, and **iterating on execution errors when it turns out to be
wrong**. The retry loop is load-bearing precisely because pretrained Minecraft knowledge is
unreliable.

**What mcAgent changes.** The round-2..11 result is that a small model does not need to carry the
facts — it needs to learn the *skill* of routing to a verified oracle. Applied to embodiment:

| | Voyager / Mindcraft | mcAgent embodied |
|---|---|---|
| source of game knowledge | model weights (pretraining) | **queried oracle (game's own data)** |
| wrong recipe | code fails → retry loop | **cannot happen — recipe is looked up** |
| model size needed | GPT-4 class | **1.7B, local** |
| cost per episode | frontier API tokens | ~0, runs on the user's box |
| modded items | unknown to the model | **works — oracle reads the live registry** |
| failure mode | confident wrong action | **safe decline** |

The last two rows are the ones no incumbent can match. A pretrained model has never seen the items
from the modpack a user installed last week; an oracle built from that user's running game has.

**And it extends the verified-data thesis one step further.** The case study already shows *verified data
teaches skills, not facts*. Embodiment adds: **verified data can teach actions, because the game is
the oracle.** Every action returns real world-state, so an action trace is self-verifying in exactly
the way a query trace is. That is the round-12+ research direction and it falls out of the product.

---

## 2. Two substrates, and a recommendation

### Path A — Mineflayer bot (recommended for v1)

Node.js library; the agent joins as a **separate client** over the protocol. What Voyager and
Mindcraft both use.

- **Pros:** mature high-level API (move, mine, craft, fight, pathfind); **no mod install**, works
  against vanilla and most servers; the incumbents' tooling and prior art are directly comparable;
  fastest path to a working agent.
- **Cons:** the agent is a *second player*, not an in-game companion of yours; no access to
  client-side rendering; some server anti-bot friction.

### Path B — Fabric/NeoForge mod

- **Pros:** true in-game presence; **direct access to live registries** — `RecipeManager` is a
  singleton on the server holding all loaded recipes, and tags/loot are equally enumerable, so the
  oracle can be built from the *running game including installed mods*; distributable on
  Modrinth/CurseForge where the audience is.
- **Cons:** substantially more work; Java port of everything; must target both loaders (the
  Forgified Fabric API makes one codebase serve both) and track versions (currently 1.21.x and
  26.1.x).

### Recommendation

**Build A first, keep B as the distribution endgame.** Path A gets a working embodied agent fastest
and is directly benchmarkable against Voyager/Mindcraft — which is what makes the result legible.
Path B is what makes it *installable by strangers*, and it can reuse everything except the transport
layer if the tool interface is defined as a protocol boundary from day one (see §4).

---

## 3. The tool surface

Three tiers. The split matters: **knowledge is verified, state is observed, actions are executed and
their results observed.** The model never asserts anything it did not read back.

### 3.1 Knowledge tools — the existing oracle, unchanged

Already built and adversarially tested (`fact_pipeline/tool_oracle/lookup.py`):

`lookup_recipe(item)` · `lookup_loot(source)` · `lookup_food(animal)` · `lookup_tag(tag)` ·
`lookup_enchantment(name)` · `lookup_trade(trade)` · `lookup_jukebox(song)` ·
`lookup_painting(name)` · `lookup_item_use(item)`

Plus the verifier half — `is_recipe_ingredient()`, `is_breeding_food()` — which rejects a proposed
item that isn't in the table. This is the fabrication barrier and it already works: 9/9 unknown
subjects declined across the probes, zero fabrications since round 3.

### 3.2 State tools — perception, read-only

New, thin wrappers over Mineflayer's existing state:

`inventory()` · `position()` · `health_and_hunger()` · `time_of_day()` · `nearby_blocks(radius)` ·
`nearby_entities(radius)` · `equipped()` · `biome()`

### 3.3 Action tools — the new capability

`goto(x, y, z)` · `goto_block(type)` · `follow(entity)` · `mine(block_or_pos)` ·
`place(item, pos)` · `craft(item, count)` · `smelt(item, fuel)` · `attack(entity)` ·
`equip(item, slot)` · `drop(item, count)` · `chest_deposit/withdraw(item, count)` ·
`eat(item)` · `sleep()`

**Every action returns real post-state**, not a success string. `craft` returns the resulting
inventory delta; `mine` returns what was actually collected; `goto` returns the position actually
reached. That is what makes an action trace self-verifying — and what makes fabricating a completed
action impossible in the same way fabricating a fact already is.

### 3.4 The composition that is the whole point

> *"Build me a stone pickaxe."*
> `lookup_recipe(stone_pickaxe)` → needs cobblestone ×3, stick ×2 (**verified, not recalled**)
> `inventory()` → has 0 cobblestone, 4 sticks
> `lookup_recipe(stick)` → planks ×2 · `goto_block(stone)` → `mine(stone) ×3` → `craft(...)`
> → `inventory()` confirms the pickaxe exists

No step depends on the model *remembering* Minecraft. Every fact is queried; every action is
confirmed. A wrong recipe is structurally impossible, so the retry loop that dominates Voyager's
cost is not needed for knowledge errors — only for genuine world friction (blocked path, mob
interrupt), which is what retries should be for.

---

## 4. Architecture

```
  ┌────────────────────┐   OpenAI-compatible /v1/chat/completions
  │  Model runtime     │◄──────────────────────────────────────┐
  │  Ollama / llama.cpp│   (user-supplied endpoint — the        │
  │  Qwen3-1.7B + LoRA │    established norm in this ecosystem) │
  └────────────────────┘                                        │
                                                                │
  ┌─────────────────────────────────────────────────────────────┴───┐
  │  AGENT CORE  (the port of answer_question, generalised)          │
  │   loop: prompt → parse <tool_call> → dispatch → <result> → ...   │
  │   hop budget, decline handling, error surfacing                  │
  └───────┬──────────────────────────┬───────────────────────────────┘
          │                          │
  ┌───────▼────────┐        ┌────────▼─────────────────────────┐
  │ KNOWLEDGE      │        │ EMBODIMENT ADAPTER (interface)   │
  │ oracle (SQLite)│        │  state + action tools            │
  │ read-only      │        ├──────────────┬───────────────────┤
  └────────────────┘        │ Mineflayer   │ Fabric/NeoForge   │
                            │  (v1)        │  (v2)             │
                            └──────────────┴───────────────────┘
```

**The load-bearing design decision:** the embodiment adapter is an **interface**, not an
implementation. Define the tool schema once (JSON), implement it twice. Path B then reuses the agent
core, the oracle schema, the prompts, and the training data — only the adapter is rewritten.

**Why the existing loop ports cleanly.** `answer_question` is ~35 lines and already speaks
OpenAI-style message lists with a chat template; the only MLX-specific call is `generate()`.
Swapping that for an HTTP POST is the entire port. The oracle is 8 lookups + 2 verifiers, all
parameterised `SELECT`s with `COLLATE NOCASE` — trivially portable to JS or Java.

---

## 5. Legal and data-provenance constraint (resolved, and it improves the design)

Mojang permits mods but forbids redistributing "anything we've made." The current `minecraft.db` is
derived from the Java data-generator report — i.e. Mojang's game data. **Do not ship the database.**

**Ship the generator instead.** Two clean forms:

- **Path A:** build the DB on first run from the user's own installed game jar / data report.
- **Path B:** build it from the **live registries of the running game** — which is *better* than the
  data report, because it automatically includes every mod the user has installed and is
  version-correct by construction.

This turns a legal constraint into the feature that beats every pretrained competitor: **the oracle
knows the user's actual modpack.**

Model licensing is clean by comparison — Qwen3 is Apache-2.0, the LoRA is yours, and the adapter
publishes to HuggingFace with a one-line pull.

---

## 6. Build order

Each stage ships something usable and de-risks the next. **Do not start at stage 3.**

### Stage 0 — Honest baseline (days)
Fix the measurement before building on it. The headline 97% is the hand-written probe on
**in-distribution subjects**; the auto-built eval (`build_eval_items`, which genuinely excludes
trained subjects by parsing `lookup(...key='X')` out of the training JSONL) is the held-out number.
Run the held-out eval at scale and publish *that* as the Q&A capability figure. Shipping to
strangers means arbitrary phrasings on arbitrary items.

### Stage 1 — Knowledge server (1 week)
Wrap the existing oracle + agent loop behind a small HTTP service with a stable JSON tool schema.
This is the seam everything else plugs into, and it makes the Python↔Node boundary a non-issue.
Deliverable: `POST /ask` answers a Minecraft question, verified or declined.

### Stage 2 — Mineflayer agent, knowledge-only (1–2 weeks)
Bot joins a server, reads chat, answers questions in-game using stage 1. No actions yet.
**This is already a shippable product** — an in-game assistant that cannot lie about the game.
Genuinely differentiated: every existing AI Minecraft mod can fabricate.

### Stage 3 — State tools (1 week)
Add perception. The agent can now answer "what do I need for a pickaxe, and do I have it?" —
combining verified knowledge with real inventory. Still no writes; safe to test freely.

### Stage 4 — Action tools (2–4 weeks)
Add the write surface, guarded: dry-run mode, an action allowlist, a confirmation gate for
destructive actions, and a hard budget per task. Every action returns real post-state.

### Stage 5 — Verified action traces (research, ongoing)
Generate training traces against a real server the same way query traces were generated: task →
tool_call → **real world result** → next. Self-verifying by construction. Retrain; this is where the
1.7B model learns to *act* rather than just route. **This is the publishable part.**

### Stage 6 — Distribution
Path B mod for Modrinth/CurseForge (Fabric + NeoForge via Forgified Fabric API, 1.21.x + 26.1.x),
reusing everything but the adapter. Model via Ollama pull. README leads with the decline behaviour
as a feature.

---

## 7. Risks, honestly

| risk | severity | mitigation |
|---|---|---|
| **1.7B is too small to plan multi-step actions** | **high** | unknown until stage 4; the oracle removes the *knowledge* burden but not the *planning* burden. Mitigation: pre-canned skill templates (Voyager's skill-library idea) so the model composes rather than invents. Fall back to a larger local model if needed — the architecture is model-agnostic. |
| Action-space safety (griefing, world damage) | medium | dry-run default, allowlist, confirmation gate, per-task budget |
| Server-side anti-bot friction | medium | single-player and self-hosted first |
| Minecraft version churn breaks the adapter | medium | oracle is regenerated per-version by design; adapter pinned to a version matrix |
| Java port cost (path B) | medium | interface-first design; defer until path A proves the product |
| Crowded field | low | none of the incumbents have a verified oracle — that *is* the differentiation |
| Model distribution size | low | user-supplied endpoint, ecosystem norm |

**The honest headline risk is the first row.** Everything else is engineering. Whether a 1.7B model
can *plan* — not merely route — is the open empirical question, and stage 4 is where it gets
answered. Stages 1–3 are worth doing regardless of how that lands, which is the main argument for
this build order.

---

## 8. Strategic note

This is a genuinely different asset from the strict-C verifier. It is far **more legible** — "the
Minecraft agent that can't make things up, running a 1.7B model locally, because it queries the
game instead of remembering it" lands with an audience that would bounce off a GCC oracle. It is
also directly comparable to Voyager, which is a well-known result, and it demonstrates the corpus's
central thesis in a domain people can see.

It is also **weeks of engineering that is not research**. Both things are true. Stage 5 is the part
that is research; stages 1–4 are the substrate it needs. Worth being deliberate about which is being
bought at each stage.

## References

- Voyager: An Open-Ended Embodied Agent with LLMs — arXiv 2305.16291 · https://voyager.minedojo.org/
- Mineflayer — https://github.com/PrismarineJS/mineflayer
- Mindcraft — https://github.com/mindcraft-bots/mindcraft
- NeoForge docs, recipes/tags at runtime — https://docs.neoforged.net/docs/resources/server/recipes/
- Forgified Fabric API (one codebase, both loaders) — https://modrinth.com/mod/forgified-fabric-api
- Minecraft Usage Guidelines / EULA — https://www.minecraft.net/en-us/usage-guidelines
- Prior art with local inference: AI Companion Fabric, AI-Player, Bafchat, CreatureChat, LLMjs
