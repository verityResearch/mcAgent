// Stage 5 (verified action traces). Tier-1 (9 tools) call action_tools.js
// directly -- they were never gated, so generation can run unattended with
// no dispatcher involved at all. Tier 2/3 (mine, attack, chest ops) go
// through action_dispatcher.js in UNGATED mode (makeDispatcher({mode:
// 'ungated'}), see action_dispatcher.js's own comment): request() resolves
// AND executes immediately, no dry-run/nonce/confirm, because a generation
// script is not a human that needs asking. This is per the reviewed design -- going through the dispatcher (not calling
// action_tools_tier2.js resolvers directly) matters because it's the SAME
// code path a gated caller uses (deny-by-default, per-tier enable, the
// construction-level guards in resolveAttack/resolvePlace), just without
// the confirm wait. `place` is deliberately EXCLUDED from tier-2 generation:
// bot.placeBlock() is still broken on this mineflayer/server combination
// (isolated to 1.21.11 specifically via a differential test against a real
// 1.20.4 server -- see action_tools_tier2.js), and generating place() traces
// now would poison the corpus with un-executable actions.
//
// TWO GATES per the review, not one -- authenticity alone is not
// trajectory quality:
//   Gate 1 (authenticity): every observation is the tool's REAL return
//     value, captured at generation time, never predicted or hand-edited.
//     checkTraceIntegrity() enforces this structurally (on-disk JSON is an
//     exact round-trip of what was captured, catches a serialization bug --
//     correctness of any single tool call already comes from action_tools.js
//     itself, live-verified in stage 4).
//   Gate 2 (outcome): for DAG-sourced craft-chain tasks, did the task
//     ACTUALLY complete -- checked against real inventory state (before/after
//     item-count diff), not inferred from the last step's own return value.
//     The review's point: a model could call craft() fifteen times with wrong
//     ingredients, get fifteen REAL (gate-1-passing) failures, stumble into
//     success, and every step would be "verified" under authenticity alone --
//     training on that teaches flailing. The state tools from stage 3
//     (inventory here) are the outcome oracle, same "don't judge it, query
//     it" move as the rest of this project.
//
// FAILURE TRACES ARE REQUIRED, not merely permitted -- same shape as round
// 2->3's fabrication fix (train only on success, model learns "always
// succeed" and reports success it didn't achieve; round 3's fix was
// deliberate DECLINE traces). This generator keeps three distinct kinds:
//   - in-trajectory failure steps that still lead to overall success (kept
//     automatically -- this is recovery, the good stuff)
//   - a DELIBERATE slice of wholly-failed-but-honestly-reported trajectories
//     (HONEST_FAILURE_TASKS below -- items genuinely unreachable via
//     tier-1-only crafting, e.g. needing mined materials)
//   - harness artifacts (a genuine JS exception, not a tool-reported
//     failure) -- discarded, these teach nothing real
//
// TASK SOURCING, option (d) from the review: stratify by recipe-DAG
// depth (recipe_dag.js) rather than sampling uniformly or hand-authoring
// only. (a) [[hand-authored]] traces stay as the seed/regression set below.
//
// SCALED, 2026-08-21, per the stage-5 priority reset: "stage 5's actual research question is still open... scale
// the tier-1 generator, measure depth-bucket population as a first-class
// metric while generating." RAW_MATERIALS widened from a single item
// (oak_log) to every standard overworld wood type plus a few other
// hand-gatherable materials; DAG_SAMPLES_PER_BUCKET raised from 2 to 15.
// Raw-material stock is now self-managed via resupplyRawMaterials()
// (mineflayer's built-in creative-mode inventory API -- no OP account
// needed, unlike setup_fixtures.js's /give-based pattern), called once at
// startup and before every DAG task, replacing the earlier "wait for an
// external resupply process" comment (that process was never committed --
// this generator now handles its own supply). bucketPopulation/
// bucketCounts/actualBucketCounts are all persisted in the manifest, not
// just console-logged -- a flat difficulty gradient discovered only AFTER a
// corpus exists nullifies a whole class of A/B comparisons by construction.
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const fs = require('fs')
const path = require('path')
const actionTools = require('./action_tools')
const { fetchAllRecipes, resolveAllTags, reachableTier1 } = require('./recipe_dag')
const { makeDispatcher } = require('./action_dispatcher')

// tier 2/3 tools this generator will attempt -- 'place' deliberately absent
const TIER23_TOOLS_ENABLED = { 2: true, 3: true }

const HOST = '127.0.0.1'
const PORT = 25566
const USERNAME = 'trace_gen' // separate account from stage2_probe (the live-serving bot) -- doesn't interrupt it
const KNOWLEDGE_SERVER = 'http://127.0.0.1:8420'
// SCALED per the stage-5 priority reset:
// "scale the tier-1 generator -- more raw materials, more per-bucket samples."
// Was a single item (oak_log); now every standard overworld wood type plus a
// few other hand-gatherable-without-mining materials. Each wood color mostly
// widens EXISTING depth buckets (its planks/doors/boats/etc. sit at the same
// depths oak's do) rather than unlocking new depths -- that's the point: real
// per-bucket population, not just deeper reach. reachableTier1() decides what
// each material actually unlocks; listing an item here that turns out to
// unlock nothing is harmless, not a correctness risk.
const RAW_MATERIALS = [
  'minecraft:oak_log', 'minecraft:birch_log', 'minecraft:spruce_log', 'minecraft:jungle_log',
  'minecraft:acacia_log', 'minecraft:dark_oak_log', 'minecraft:mangrove_log', 'minecraft:cherry_log',
  'minecraft:sugar_cane', 'minecraft:bamboo', 'minecraft:cactus',
]
// Self-resupply + fixture-relocation logic now lives in creative_resupply.js
// (factored out so live_eval_action_tasks.js can reuse the exact same,
// already-fixed logic -- see that module's own header for the full history
// of the 3 real bugs found+fixed while building this).
const { resupplyRawMaterials, relocateToFixture } = require('./creative_resupply')

// action(tool, k='v', k2=42) -- mirrors lookup(table, key='...')'s syntax so
// a future inference-time parser can extend the same _CALL_RE-style regex
// rather than inventing a second grammar.
function formatToolCall(tool, namedArgs) {
  const parts = Object.entries(namedArgs).map(([k, v]) =>
    typeof v === 'string' ? `${k}='${v}'` : `${k}=${v}`)
  return parts.length ? `action(${tool}, ${parts.join(', ')})` : `action(${tool})`
}

// lookup(table, key='value') -- the SAME syntax round 11's knowledge corpus
// already trained the model on (confirmed by reading its own train.jsonl:
// "<tool_call>lookup(recipe, result_item='minecraft:ender_eye')</tool_call>"),
// reused here rather than inventing a second grammar, so a lookup step in an
// action trace looks EXACTLY like one in a knowledge trace to the model.
function formatLookupCall(table, namedArgs) {
  const parts = Object.entries(namedArgs).map(([k, v]) => `${k}='${v}'`)
  return `lookup(${table}, ${parts.join(', ')})`
}

function makeRecorder(bot, ungatedDispatcher) {
  const steps = []
  async function call(tool, namedArgs, positionalArgs, think) {
    const toolCall = formatToolCall(tool, namedArgs)
    const result = await actionTools[tool](bot, ...positionalArgs)
    steps.push({ think, toolCall, resultText: JSON.stringify(result) })
    return result
  }
  // Same as call(), but lets the caller attach extra REAL, independently-
  // computed fields onto the tool's own real result before it's recorded --
  // e.g. a missing-ingredient diff computed from real inventory state
  // against the real oracle recipe data. Gate 1 still holds: the underlying
  // tool call is real, and augment() only adds verified facts about real
  // world state, never fabricates the call's own outcome. Mirrors
  // live_eval_action_tasks.js's own missingIngredientsDiff pattern, kept
  // as a distinct helper here since generation and live-eval don't share a
  // module (see RAW_MATERIALS's own header comment for why).
  async function callWithAugment(tool, namedArgs, positionalArgs, think, augment) {
    const toolCall = formatToolCall(tool, namedArgs)
    const result = await actionTools[tool](bot, ...positionalArgs)
    if (augment) augment(result)
    steps.push({ think, toolCall, resultText: JSON.stringify(result) })
    return result
  }
  // Tier-2/3 path: goes through the ungated dispatcher's request(), which
  // resolves+executes in one call (see the header comment). The recorded
  // <result> is the REAL tool result (dispatcherResponse.result), not the
  // dispatcher's {ok,message} wrapper -- keeps the trace format identical
  // to tier-1's regardless of which path produced it.
  async function callTier23(tool, namedArgs, positionalArgs, think) {
    const toolCall = formatToolCall(tool, namedArgs)
    const response = await ungatedDispatcher.request(bot, 'trace_gen', tool, positionalArgs.map(String))
    if (!response.ok) {
      // Dispatcher-level refusal (deny-by-default, tier disabled, or a
      // resolve()-time refusal like place()'s deny-list or attack()'s
      // no-matching-hostile) -- still a REAL result to record, just shaped
      // like {refused: true, reason} rather than a tool's own return object.
      const result = { refused: true, reason: response.message }
      steps.push({ think, toolCall, resultText: JSON.stringify(result) })
      return result
    }
    steps.push({ think, toolCall, resultText: JSON.stringify(response.result) })
    return response.result
  }
  // TRACE-SHAPE FIX per the design review:
  // "Do the action traces include a lookup_recipe() before the craft()?"
  // -- checked, they did NOT (0/73 DAG traces contained any lookup() call).
  // The action half was trained on "just craft," running on memorized
  // recipes, while the knowledge half was trained on "look it up, then
  // answer" -- two epistemics, the weaker one failing. This method adds
  // the missing half of the unit: a REAL lookup_recipe(item) call, with a
  // REAL result (the item's actual ingredient list from recipe_dag.js's
  // already-fetched recipe data, formatted with the SAME comma-joined
  // shape round 11's own knowledge traces use -- confirmed by reading
  // train.jsonl, not assumed), recorded as its own step BEFORE the craft()
  // call it justifies. No live tool call happens here -- this is a real,
  // already-known-true fact about the game's own recipe data, the same
  // "verified, not recalled" discipline as everywhere else in this
  // project, just surfaced as an explicit trained step instead of silently
  // assumed.
  function lookup(table, namedArgs, ingredients, think) {
    const toolCall = formatLookupCall(table, namedArgs)
    // Matches round 11's own rendering exactly (confirmed by reading its
    // train.jsonl: "<result>minecraft:blaze_powder, minecraft:ender_pearl</result>"
    // -- namespaced, comma-and-space-joined, no JSON/brackets).
    const resultText = ingredients.join(', ')
    steps.push({ think, toolCall, resultText })
  }
  return { bot, steps, call, callWithAugment, callTier23, lookup }
}

function toTrace(task, recorder, answer) {
  const messages = [{ role: 'user', content: task }]
  for (const { think, toolCall, resultText } of recorder.steps) {
    messages.push({ role: 'assistant', content: `<think>${think}</think>\n<tool_call>${toolCall}</tool_call>` })
    messages.push({ role: 'user', content: `<result>${resultText}</result>` })
  }
  messages.push({ role: 'assistant', content: answer })
  return { messages }
}

function checkTraceIntegrity(trace, recorder) {
  const resultMsgs = trace.messages.filter((m) => m.role === 'user' && m.content.startsWith('<result>'))
  if (resultMsgs.length !== recorder.steps.length) return false
  return resultMsgs.every((m, i) => {
    const stated = m.content.slice('<result>'.length, -'</result>'.length)
    return stated === recorder.steps[i].resultText
  })
}

function itemCount(bot, bareName) {
  return bot.inventory.items().filter((it) => it.name === bareName)
    .reduce((s, it) => s + it.count, 0)
}

const CRAFT_BATCH = 8 // generous default craft count per intermediate -- the oracle DB has no
// per-recipe output QUANTITY data, only ingredient lists, so this over-crafts intermediates
// rather than under-crafting and having the final step fail for a wrong guess. The outcome
// gate below is what actually decides success/failure, not this number.

// Executes a real multi-step craft-chain trajectory for `targetItem`, given
// its real prerequisite items (from recipe_dag.js's reachableTier1, already
// verified craftable purely from RAW_MATERIALS). Returns {completed, answer}
// where `completed` is the GATE-2 (outcome) verdict: inventory of the
// target item genuinely increased, not inferred from the last step's report.
//
// `recipes` is the SAME full recipe map fetchAllRecipes() already fetched
// once for the whole generation run (recipe_dag.js) -- reused here, not
// re-fetched, purely to render the query-before-act lookup() step's REAL
// result. Each craft() is now preceded by its own lookup(recipe,
// result_item=...) step per the trace-shape fix:
// the action corpus previously went straight to craft() with the recipe
// knowledge computed OFFLINE and never shown to the model, teaching it to
// craft from memorized recipes instead of querying first -- the opposite
// epistemic from the knowledge half's "look it up, then answer."
async function craftChain(recorder, targetItem, prereqItems, recipes) {
  const bareTarget = targetItem.replace('minecraft:', '')
  const before = itemCount(recorder.bot, bareTarget)

  // A crafting table is needed for most non-trivial recipes; positioning
  // near one is harmless even for 2x2 recipes that don't strictly need it.
  await recorder.call('goto_block', { block: 'crafting_table', radius: 16, range: 1 },
    ['crafting_table', 16, 1], 'Position near a crafting table before attempting any non-trivial craft.')

  for (const prereq of prereqItems) {
    const barePrereq = prereq.replace('minecraft:', '')
    recorder.lookup('recipe', { result_item: prereq }, recipes[prereq] || [],
      `Check the real recipe for ${barePrereq} before crafting it as an intermediate.`)
    await recorder.call('craft', { item: barePrereq, count: CRAFT_BATCH }, [barePrereq, CRAFT_BATCH],
      `${bareTarget} needs ${barePrereq} as an intermediate; craft a batch of it first.`)
  }

  recorder.lookup('recipe', { result_item: targetItem }, recipes[targetItem] || [],
    `Check the real recipe for ${bareTarget} before crafting the target.`)
  const finalResult = await recorder.call('craft', { item: bareTarget, count: 1 }, [bareTarget, 1],
    `All prerequisites for ${bareTarget} should now be in inventory; craft the target.`)

  const after = itemCount(recorder.bot, bareTarget)
  const completed = after > before
  const answer = completed
    ? `Successfully crafted ${bareTarget} (had ${before}, now have ${after}).`
    : `Could not complete crafting ${bareTarget} -- still have ${after} (started with ${before}). ` +
      `Last step reported: ${JSON.stringify(finalResult)}.`
  return { completed, answer }
}

// Category-3 fix, per the failure-taxonomy read: HONEST_FAILURE_ITEMS
// below already teaches "attempt a genuinely unreachable item, report
// failure honestly" -- but that's a SINGLE-STEP pattern. The live-eval
// failure mode actually observed was different: a multi-step chain where
// the model stops PARTWAY (crafts planks, skips the stick) and then
// reports the whole thing as done anyway -- fabrication in action space,
// the same shape round 2->3's knowledge-decline fix addressed (train only
// on success -> model learns "always report success"). This slice runs a
// REAL multi-step chain, deliberately stops BEFORE the final target craft,
// and reports the REAL partial state (what got crafted, what didn't)
// instead of fabricating completion.
async function craftChainPartial(recorder, targetItem, prereqItems, recipes) {
  const bareTarget = targetItem.replace('minecraft:', '')
  const before = itemCount(recorder.bot, bareTarget)

  await recorder.call('goto_block', { block: 'crafting_table', radius: 16, range: 1 },
    ['crafting_table', 16, 1], 'Position near a crafting table before attempting any non-trivial craft.')

  const craftedPrereqs = []
  for (const prereq of prereqItems) {
    const barePrereq = prereq.replace('minecraft:', '')
    recorder.lookup('recipe', { result_item: prereq }, recipes[prereq] || [],
      `Check the real recipe for ${barePrereq} before crafting it as an intermediate.`)
    await recorder.call('craft', { item: barePrereq, count: CRAFT_BATCH }, [barePrereq, CRAFT_BATCH],
      `${bareTarget} needs ${barePrereq} as an intermediate; craft a batch of it first.`)
    craftedPrereqs.push(barePrereq)
  }

  // Deliberately stop here -- the final target craft() is never attempted.
  // This is not a bug or a timeout cut short; it is the trace's whole
  // point, so outcomeVerified is honestly false, not inferred as success.
  const after = itemCount(recorder.bot, bareTarget)
  const answer = craftedPrereqs.length > 0
    ? `Crafted intermediate(s) ${craftedPrereqs.join(', ')} toward ${bareTarget}, but did not reach the final craft step -- ${bareTarget} count is still ${after} (started at ${before}). Reporting this as incomplete, not as a finished craft.`
    : `Did not make progress toward ${bareTarget} -- no intermediates were crafted yet. Reporting this as incomplete.`
  return { completed: false, answer }
}

// Mid-trajectory recovery traces, per the diagnosis: checked the corpus directly -- every honest_failure
// and honest_partial trace is 100% terminal, ending AT the failure/
// incompletion with zero examples of a diagnosed failure followed by real
// recovery. Live-eval showed the direct consequence: the model correctly
// diagnoses a missing ingredient in its own final answer ("missing
// ingredient stick") and then simply stops -- not a knowledge gap, not a
// visibility gap, an ACTION-CHAINING gap the corpus never taught it to
// close. This category generates exactly the missing lesson: attempt(target)
// -> REAL failure -> REAL diagnostic (the failed craft's own result gains a
// `missing` field, same presence-only diff live_eval_action_tasks.js's
// missingIngredientsDiff() computes, attached via callWithAugment so gate 1
// still holds -- the underlying craft() call and its base result are real,
// only a verified fact about current inventory state is added) -> craft the
// genuinely-missing ingredient(s) -> retry the target -> real outcome.
// KEEPS the terminal honest-failure/honest-partial traces untouched
// elsewhere -- this is a dosage addition, not a replacement, per the reviewer's
// explicit caution against re-opening fabrication by removing "I genuinely
// cannot do this" in favor of "this failed, so I fix it."
async function craftChainWithRecovery(recorder, targetItem, prereqItems, recipes) {
  const bareTarget = targetItem.replace('minecraft:', '')
  const before = itemCount(recorder.bot, bareTarget)

  await recorder.call('goto_block', { block: 'crafting_table', radius: 16, range: 1 },
    ['crafting_table', 16, 1], 'Position near a crafting table before attempting any non-trivial craft.')

  // Craft only the FIRST prereq, deliberately skip the rest -- matches the
  // real observed failure shape (model already has planks, missing stick)
  // and guarantees the target attempt below is a genuine, not staged,
  // failure: nothing here fakes an outcome, it just controls which real
  // ingredients exist before a real attempt.
  const [firstPrereq] = prereqItems
  if (firstPrereq) {
    const bareFirst = firstPrereq.replace('minecraft:', '')
    recorder.lookup('recipe', { result_item: firstPrereq }, recipes[firstPrereq] || [],
      `${bareTarget} needs ${bareFirst} as an intermediate; craft a batch of it first.`)
    await recorder.call('craft', { item: bareFirst, count: CRAFT_BATCH }, [bareFirst, CRAFT_BATCH],
      `${bareTarget} needs ${bareFirst} as an intermediate; craft a batch of it first.`)
  }

  recorder.lookup('recipe', { result_item: targetItem }, recipes[targetItem] || [],
    `Check the real recipe for ${bareTarget} before attempting the target.`)

  // The diagnostic augment: computed from the SAME oracle recipe data
  // (recipes[targetItem], already fetched) diffed against real inventory
  // counts -- presence-only, exactly like the live-eval harness's
  // missingIngredientsDiff, for the same schema reason (the DB has no
  // per-grid-slot quantities, only distinct ingredient types).
  function attachMissingDiff(result) {
    const ingredients = recipes[targetItem] || []
    const missing = ingredients
      .filter((ing) => !ing.startsWith('#'))
      .map((ing) => ing.replace('minecraft:', ''))
      .filter((bare) => itemCount(recorder.bot, bare) === 0)
    if (missing.length > 0) result.missing = missing
  }
  const firstAttempt = await recorder.callWithAugment('craft', { item: bareTarget, count: 1 }, [bareTarget, 1],
    `Attempt to craft ${bareTarget} now.`, attachMissingDiff)

  const afterFirst = itemCount(recorder.bot, bareTarget)
  if (afterFirst > before) {
    // Genuinely succeeded on the first try (this DAG item's real recipe
    // didn't need the skipped prereq after all) -- report honestly rather
    // than force a recovery step that didn't actually happen.
    return {
      completed: true,
      answer: `Successfully crafted ${bareTarget} (had ${before}, now have ${afterFirst}) -- no recovery needed.`,
    }
  }

  const missing = (firstAttempt && firstAttempt.missing) || []
  if (missing.length === 0) {
    // No concrete missing ingredient identified (e.g. the real gap is
    // behind an unresolvable tag reference) -- an honest terminal failure,
    // same as the existing honest_failure/honest_partial shapes, not a
    // forced recovery that isn't real.
    return {
      completed: false,
      answer: `Could not craft ${bareTarget} -- still have ${afterFirst} (started with ${before}). ` +
        `Last step reported: ${JSON.stringify(firstAttempt)}. No concrete missing ingredient could be identified to recover with.`,
    }
  }

  for (const bare of missing) {
    recorder.lookup('recipe', { result_item: `minecraft:${bare}` }, recipes[`minecraft:${bare}`] || [],
      `${bareTarget} craft failed, missing ${bare} -- check its real recipe before crafting it.`)
    await recorder.call('craft', { item: bare, count: CRAFT_BATCH }, [bare, CRAFT_BATCH],
      `${bareTarget} craft failed because ${bare} was missing; craft it now, then retry.`)
  }

  const secondAttempt = await recorder.call('craft', { item: bareTarget, count: 1 }, [bareTarget, 1],
    `All previously-missing ingredients for ${bareTarget} should now be in inventory; retry the craft.`)

  const after = itemCount(recorder.bot, bareTarget)
  const completed = after > before
  const answer = completed
    ? `First attempt at ${bareTarget} failed (missing ${missing.join(', ')}); crafted the missing ingredient(s) and retried -- now have ${after}.`
    : `Could not complete crafting ${bareTarget} even after crafting the previously-missing ingredient(s) ${missing.join(', ')} -- ` +
      `still have ${after} (started with ${before}). Last step reported: ${JSON.stringify(secondAttempt)}.`
  return { completed, answer }
}

// Hand-authored seed task list (option (a) from the stage-5 proposal) -- small, high-quality, exercises all 9 tier-1 tools at least once.
const HAND_AUTHORED_TASKS = [
  {
    task: 'Craft 4 oak planks from your oak logs.',
    async run(recorder) {
      const think = 'Crafting oak_planks from oak_log needs no crafting table (2x2 grid); call craft directly.'
      const result = await recorder.call('craft', { item: 'oak_planks', count: 4 }, ['oak_planks', 4], think)
      return result.crafted > 0
        ? `Crafted ${result.crafted} oak planks (now have ${result.totalNow} total).`
        : `Could not craft oak planks: ${result.reason}.`
    },
  },
  {
    task: 'Go stand next to the nearest furnace.',
    async run(recorder) {
      const think = 'Locate the nearest furnace and path to it; goto_block does both in one call.'
      const result = await recorder.call('goto_block', { block: 'furnace', radius: 16, range: 1 }, ['furnace', 16, 1], think)
      return result.found
        ? `Found a furnace at (${result.block.position.x},${result.block.position.y},${result.block.position.z}) and ${result.reached ? 'reached it' : 'got as close as possible'}.`
        : `No furnace found nearby: ${result.reason}.`
    },
  },
  {
    task: 'Smelt raw iron into iron ingots using coal as fuel.',
    async run(recorder) {
      const think = 'Need a furnace nearby (just confirmed one), raw_iron and coal in inventory; call smelt.'
      const result = await recorder.call('smelt', { item: 'raw_iron', fuel: 'coal' }, ['raw_iron', 'coal'], think)
      return result.smelted > 0
        ? `Smelted ${result.smelted}x ${result.item}.`
        : `Could not smelt: ${result.reason}.`
    },
  },
  {
    task: 'Equip a stick in your main hand.',
    async run(recorder) {
      const think = 'A stick should already be in inventory from earlier steps; call equip with destination hand.'
      const result = await recorder.call('equip', { item: 'stick', destination: 'hand' }, ['stick', 'hand'], think)
      return result.equipped
        ? `Equipped stick in the main hand.`
        : `Could not equip stick: ${result.reason}.`
    },
  },
  {
    task: 'Eat some bread.',
    async run(recorder) {
      const think = 'Bread should be in inventory; call eat. Note food may already be full, which is a real, valid outcome to report honestly.'
      const result = await recorder.call('eat', { item: 'bread' }, ['bread'], think)
      return result.ate
        ? `Ate bread; health/hunger now ${JSON.stringify(result.state)}.`
        : `Could not eat bread: ${result.reason}.`
    },
  },
  {
    task: 'Drop 1 oak plank on the ground.',
    async run(recorder) {
      const think = 'oak_plank is an ordinary item (not enchanted/netherite/diamond-tier), so the value guard should allow this.'
      const result = await recorder.call('drop', { item: 'oak_planks', count: 1 }, ['oak_planks', 1], think)
      return result.dropped > 0
        ? `Dropped ${result.dropped} oak plank (${result.remaining} remaining in inventory).`
        : `Could not drop oak planks: ${result.reason}.`
    },
  },
  {
    task: 'Go to position 3 blocks east of your current position.',
    async run(recorder) {
      const pos = recorder.bot.entity.position
      const target = { x: Math.floor(pos.x) + 3, y: Math.floor(pos.y), z: Math.floor(pos.z), range: 1 }
      const think = `Current position is roughly (${Math.floor(pos.x)},${Math.floor(pos.y)},${Math.floor(pos.z)}); goto the target 3 blocks east.`
      const result = await recorder.call('goto', target, [target.x, target.y, target.z, target.range], think)
      return result.reached
        ? `Reached (${target.x},${target.y},${target.z}).`
        : `Did not fully reach the target; ended at ${JSON.stringify(result.position)}.`
    },
  },
  {
    task: 'Try to find and follow a nearby player named "trace_probe" for a few seconds.',
    async run(recorder) {
      const think = 'follow() needs a connected player entity named trace_probe nearby; this may genuinely fail if none is present -- report that honestly rather than fabricating success.'
      const result = await recorder.call('follow', { username: 'trace_probe', range: 2, durationMs: 3000 }, ['trace_probe', 2, 3000], think)
      return result.found
        ? `Followed trace_probe for a few seconds, ending distance ${result.distance}.`
        : `Could not follow trace_probe: ${result.reason}.`
    },
  },
  {
    task: 'Try to sleep in the nearest bed.',
    async run(recorder) {
      const think = 'sleep() needs a real bed within 16 blocks; this environment may not have one right now, in which case declining is correct, not a failure to hide.'
      const result = await recorder.call('sleep', {}, [], think)
      return result.slept
        ? `Slept successfully.`
        : `Could not sleep: ${result.reason}.`
    },
  },
]

// Deliberate honest-failure slice, per the review: items genuinely
// UNREACHABLE via tier-1-only crafting (need mined materials -- ore, stone,
// etc. -- which needs mine(), a tier-2 tool not available here). The
// trajectory is real and honestly attempted; it is EXPECTED to fail, and
// that failure is the point, the action-space analog of "excalibur isn't in
// the enchantment table" from round 3's knowledge-decline traces.
const HONEST_FAILURE_ITEMS = ['minecraft:iron_pickaxe', 'minecraft:diamond_sword', 'minecraft:stone_axe']

// Tier-2/3 seed tasks -- hand-authored, matching how tier-1 also started
// with a seed set before DAG-depth sourcing existed. mine/attack don't have
// a natural recipe-DAG-depth analog (there's no "recipe" for "mine stone" or
// "fight a creeper"), so this stays hand-authored rather than forcing a
// depth-stratification scheme that wouldn't mean anything here. `place` is
// absent -- see the header comment.
const TIER23_TASKS = [
  {
    task: 'Mine the block directly beneath you.',
    async run(recorder) {
      const pos = recorder.bot.entity.position.floored()
      const target = pos.offset(0, -1, 0)
      const think = `Mine the block at (${target.x},${target.y},${target.z}), directly below current position.`
      const result = await recorder.callTier23('mine', { x: target.x, y: target.y, z: target.z },
        [target.x, target.y, target.z], think)
      return result.mined
        ? `Mined ${result.blockName} at (${target.x},${target.y},${target.z}), collected ${JSON.stringify(result.collected)}.`
        : `Could not mine there: ${result.reason}.`
    },
  },
  {
    task: 'Deposit some sticks into the nearest chest.',
    async run(recorder) {
      const think = 'chest_deposit needs a real chest nearby and sticks in inventory; try depositing 2.'
      const result = await recorder.callTier23('chest_deposit', { item: 'stick', count: 2 }, ['stick', 2], think)
      return result.deposited > 0
        ? `Deposited ${result.deposited} stick(s), ${result.remainingInInventory} left in inventory.`
        : `Could not deposit: ${result.reason}.`
    },
  },
  {
    task: 'Withdraw a stick from the nearest chest.',
    async run(recorder) {
      const think = 'chest_withdraw needs a real chest nearby with sticks already in it.'
      const result = await recorder.callTier23('chest_withdraw', { item: 'stick', count: 1 }, ['stick', 1], think)
      return result.withdrawn > 0
        ? `Withdrew ${result.withdrawn} stick(s), now have ${result.nowInInventory} in inventory.`
        : `Could not withdraw: ${result.reason}.`
    },
  },
  {
    task: 'Attack a nearby hostile mob if one is present.',
    async run(recorder) {
      // Real-world-friction case built into the task itself: whether a
      // hostile mob exists nearby is genuinely unknown until attempted --
      // resolveAttack()'s own candidate search is the source of truth, not
      // a pre-check here. A "no hostile mob nearby" refusal is a real,
      // honest, keepable failure trace, not a harness artifact.
      const think = 'Search for the nearest hostile mob and attack it if one exists; declining honestly if none is nearby is the correct outcome, not a failure to hide.'
      const result = await recorder.callTier23('attack', { target: 'zombie' }, ['zombie'], think)
      if (result.refused) return `Could not attack: ${result.reason}.`
      return result.attacked
        ? `Attacked ${result.target} at real melee range (${result.distanceAtSwing} blocks).`
        : `Attack attempt did not land: ${result.reason}.`
    },
  },
]

async function main() {
  // Default output lives under data/verified/, matching the
  // existing dataset_*.jsonl convention (gitignored -- generated data is
  // disposable and regenerable from this script, not committed). Renamed
  // from mc_bot_tier1_action_traces.jsonl now that tier 2/3 traces are
  // included too -- trace.meta.source still distinguishes origin per-trace.
  const outPath = process.argv[2] ||
    path.join(__dirname, '..', 'data', 'verified', 'mc_bot_action_traces.jsonl')
  fs.mkdirSync(path.dirname(outPath), { recursive: true })

  const bot = mineflayer.createBot({ host: HOST, port: PORT, username: USERNAME, version: '1.21.11' })
  bot.loadPlugin(pathfinder)

  await new Promise((resolve, reject) => {
    bot.once('spawn', resolve)
    bot.once('error', reject)
    setTimeout(() => reject(new Error('spawn timeout')), 20000)
  })
  bot.pathfinder.setMovements(new Movements(bot))
  console.log('trace_gen spawned at', bot.entity.position)

  await relocateToFixture(bot)
  console.log('relocated to fixture area at', bot.entity.position)

  // UNGATED dispatcher, per the reviewer's design: a SEPARATE construction from
  // bot.js's live-serving dispatcher (which is always the plain, confirm-
  // gated makeDispatcher()). Enabling tier 2/3 here only opts THIS
  // generation script's own dispatcher instance in -- it has no effect on
  // and no relationship to bot.js's dispatcher or its enableTier() calls.
  const ungatedDispatcher = makeDispatcher({ mode: 'ungated' })
  if (TIER23_TOOLS_ENABLED[2]) ungatedDispatcher.enableTier(2, true)
  if (TIER23_TOOLS_ENABLED[3]) ungatedDispatcher.enableTier(3, true)

  console.log('initial resupply...')
  await resupplyRawMaterials(bot, RAW_MATERIALS)

  // Build the DAG-depth-stratified craft task list, option (d) from the reviewer's
  // review. Pick up to DAG_SAMPLES_PER_BUCKET items per reachable-depth
  // bucket (bucketed by prerequisite-step count, not the raw recipe depth --
  // that's what's actually achievable starting from RAW_MATERIALS) so the
  // difficulty gradient exists in the generated corpus by construction, not
  // by luck.
  console.log('fetching recipe DAG...')
  const recipes = await fetchAllRecipes(KNOWLEDGE_SERVER)
  const tagMembers = await resolveAllTags(KNOWLEDGE_SERVER, recipes)
  const reachable = reachableTier1(recipes, RAW_MATERIALS, tagMembers)
  const byBucket = new Map()
  for (const [item, steps] of reachable) {
    const bucket = steps.length
    if (!byBucket.has(bucket)) byBucket.set(bucket, [])
    byBucket.get(bucket).push({ item, steps })
  }
  // bucketPopulation is the FULL reachable count per bucket, independent of
  // how many get sampled into actual tasks -- per the reviewer's stage-5 priority
  // reset: "MEASURE DEPTH-BUCKET POPULATION as a first-class corpus metric
  // WHILE GENERATING, not retrospectively." A gradient found flat only after
  // the corpus exists nullifies a whole class of A/B by construction -- this
  // is captured up front instead, in the manifest, not just logged and lost.
  const DAG_SAMPLES_PER_BUCKET = 15
  const bucketPopulation = {}
  const dagTasks = []
  const bucketCounts = {}
  for (const [bucket, items] of [...byBucket.entries()].sort((a, b) => a[0] - b[0])) {
    bucketPopulation[bucket] = items.length
    const sample = items.slice(0, DAG_SAMPLES_PER_BUCKET)
    bucketCounts[bucket] = sample.length
    for (const { item, steps } of sample) {
      const bareItem = item.replace('minecraft:', '')
      dagTasks.push({
        task: `Craft ${bareItem.replace(/_/g, ' ')}.`,
        source: 'dag',
        depthBucket: bucket,
        async run(recorder) {
          await resupplyRawMaterials(recorder.bot, RAW_MATERIALS)
          const { completed, answer } = await craftChain(recorder, item, steps, recipes)
          recorder.outcomeVerified = completed
          return answer
        },
      })
    }
  }
  console.log(`DAG task sourcing: ${dagTasks.length} tasks across buckets ${JSON.stringify(bucketCounts)} (population: ${JSON.stringify(bucketPopulation)})`)

  // Category-3 fix source set: sample DISTINCT items from each bucket with
  // at least one real prereq (bucket 0 has none, so it's skipped -- there's
  // nothing to partially complete), starting AFTER the slice already
  // claimed by dagTasks above so the two categories don't reuse the same
  // item.
  const PARTIAL_SAMPLES_PER_BUCKET = 3
  const partialTasks = []
  const partialBucketCounts = {}
  for (const [bucket, items] of [...byBucket.entries()].sort((a, b) => a[0] - b[0])) {
    if (bucket === 0) continue
    const sample = items.slice(DAG_SAMPLES_PER_BUCKET, DAG_SAMPLES_PER_BUCKET + PARTIAL_SAMPLES_PER_BUCKET)
    partialBucketCounts[bucket] = sample.length
    for (const { item, steps } of sample) {
      const bareItem = item.replace('minecraft:', '')
      partialTasks.push({
        task: `Craft ${bareItem.replace(/_/g, ' ')}.`,
        source: 'honest_partial',
        depthBucket: bucket,
        async run(recorder) {
          await resupplyRawMaterials(recorder.bot, RAW_MATERIALS)
          const { completed, answer } = await craftChainPartial(recorder, item, steps, recipes)
          recorder.outcomeVerified = completed
          return answer
        },
      })
    }
  }
  console.log(`Honest-partial task sourcing: ${partialTasks.length} tasks across buckets ${JSON.stringify(partialBucketCounts)}`)

  // Mid-trajectory recovery traces, per the diagnosis: the corpus's failure traces are 100% terminal
  // (checked directly: honest_failure/honest_partial always end AT the
  // failure), so the model was never taught that a diagnosed failure is a
  // cue to act rather than a stopping point -- confirmed as the live-eval
  // mechanism (model correctly restates "missing ingredient stick" then
  // simply stops). Scoped to BUCKET 2 specifically -- the exact bucket the
  // live-eval re-measurement targets, and the one whose real failure
  // shape (single missing intermediate, e.g. planks-have/stick-missing)
  // this recovery pattern is built to match. 'recovery' is its own source
  // label, NOT part of the DAG population prepare_action_corpus.py holds
  // out from -- these traces always go to train, same as hand_authored/
  // honest_failure/honest_partial, so reusing bucket-2 items already
  // covered by dagTasks above is fine and intentional: the same item gets
  // BOTH a clean-success trace and a fail-then-recover trace, strengthening
  // the "diagnose -> act" signal without touching held-out coverage.
  const RECOVERY_SAMPLES_PER_BUCKET = 15
  const recoveryTasks = []
  const recoveryBucket = 2
  if (byBucket.has(recoveryBucket)) {
    const sample = byBucket.get(recoveryBucket).slice(0, RECOVERY_SAMPLES_PER_BUCKET)
    for (const { item, steps } of sample) {
      const bareItem = item.replace('minecraft:', '')
      recoveryTasks.push({
        task: `Craft ${bareItem.replace(/_/g, ' ')}.`,
        source: 'recovery',
        depthBucket: recoveryBucket,
        async run(recorder) {
          await resupplyRawMaterials(recorder.bot, RAW_MATERIALS)
          const { completed, answer } = await craftChainWithRecovery(recorder, item, steps, recipes)
          recorder.outcomeVerified = completed
          return answer
        },
      })
    }
  }
  console.log(`Recovery task sourcing: ${recoveryTasks.length} tasks from bucket ${recoveryBucket}`)

  const failureTasks = HONEST_FAILURE_ITEMS.map((item) => ({
    task: `Craft ${item.replace('minecraft:', '').replace(/_/g, ' ')}.`,
    source: 'honest_failure',
    async run(recorder) {
      // Deliberately attempt with NO prerequisite resolution (the item is
      // known unreachable via tier-1 crafting) -- a single real, honest
      // craft() attempt against current inventory is enough to produce the
      // real failure reason.
      const bare = item.replace('minecraft:', '')
      const before = itemCount(recorder.bot, bare)
      const result = await recorder.call('craft', { item: bare, count: 1 }, [bare, 1],
        `Attempting to craft ${bare} directly; if the ingredients aren't reachable this should honestly fail rather than fabricate success.`)
      const after = itemCount(recorder.bot, bare)
      recorder.outcomeVerified = after > before
      return recorder.outcomeVerified
        ? `Unexpectedly crafted ${bare} (${after} now) -- this item was assumed unreachable via tier-1 crafting; treat as a signal the reachability assumption needs updating.`
        : `Could not craft ${bare}: ${result.reason || 'no recipe path from current inventory'}. This is the expected, honest outcome -- ${bare} needs mined materials tier-1 tools can't gather.`
    },
  }))

  const allTasks = [
    ...HAND_AUTHORED_TASKS.map((t) => ({ ...t, source: 'hand_authored' })),
    ...dagTasks,
    ...partialTasks,
    ...recoveryTasks,
    ...failureTasks,
    ...TIER23_TASKS.map((t) => ({ ...t, source: 'tier23' })),
  ]

  const traces = []
  const sourceCounts = {}
  let integrityFailures = 0
  for (const { task, run, source, depthBucket } of allTasks) {
    const recorder = makeRecorder(bot, ungatedDispatcher)
    let answer
    try {
      answer = await run(recorder)
    } catch (err) {
      console.error(`TASK CRASHED, skipping: "${task}" ->`, err.message)
      continue
    }
    const trace = toTrace(task, recorder, answer)
    if (!checkTraceIntegrity(trace, recorder)) {
      integrityFailures++
      console.error(`INTEGRITY CHECK FAILED, skipping: "${task}"`)
      continue
    }
    trace.meta = { source, depthBucket, outcomeVerified: recorder.outcomeVerified ?? null }
    traces.push(trace)
    sourceCounts[source] = (sourceCounts[source] || 0) + 1
    console.log(`OK [${source}]: "${task}" -> ${answer}`)
  }

  // ACTUAL post-generation depth-bucket composition -- distinct from
  // bucketPopulation (total reachable) and bucketCounts (attempted) above:
  // this is what actually SURVIVED into the corpus after crashes/integrity
  // drops, the real number a training run will see per bucket.
  const actualBucketCounts = {}
  for (const t of traces) {
    if (t.meta.source !== 'dag') continue
    const b = t.meta.depthBucket
    actualBucketCounts[b] = (actualBucketCounts[b] || 0) + 1
  }

  fs.writeFileSync(outPath, traces.map((t) => JSON.stringify(t)).join('\n') + '\n')

  // Version-pinned manifest per the review: "pin the mineflayer/library
  // version too, not just the server's 1.21.11 -- placeBlock is your own
  // proof that library version is load-bearing on behavior." A corpus
  // generated against a library with a known-broken primitive needs that
  // recorded, not just the game version.
  const manifestPath = outPath.replace(/\.jsonl$/, '.meta.json')
  const manifest = {
    generatedAt: new Date().toISOString(),
    minecraftServerVersion: '1.21.11',
    mineflayerVersion: require('mineflayer/package.json').version,
    mineflayerPathfinderVersion: require('mineflayer-pathfinder/package.json').version,
    nodeVersion: process.version,
    rawMaterials: RAW_MATERIALS,
    knownLibraryIssues: [
      'bot.placeBlock() does not work on this mineflayer/server combination -- isolated to 1.21.11 specifically via a differential test against a real 1.20.4 server, see action_tools_tier2.js. place() is deliberately excluded from tier-2 generation as a result.',
    ],
    taskCount: traces.length,
    sourceCounts,
    depthBucketMetrics: {
      note: 'Per the stage-5 priority reset: measured as a first-class corpus metric WHILE GENERATING, not retrospectively. population = total items reachableTier1() found in that bucket, independent of sampling; attempted = how many were actually turned into tasks (capped at DAG_SAMPLES_PER_BUCKET); actual = how many survived into the corpus after crashes/integrity-check drops -- the real per-bucket count a training run will see.',
      population: bucketPopulation,
      attempted: bucketCounts,
      actual: actualBucketCounts,
    },
    tier23Generation: {
      dispatcherMode: 'ungated',
      note: 'tier23-sourced traces went through action_dispatcher.js in ungated mode (resolve+execute in one call, no confirm/nonce) -- a SEPARATE dispatcher construction from bot.js\'s live-serving one, per the design sent for review. place is excluded (see knownLibraryIssues).',
    },
    gates: {
      authenticity: 'all traces',
      outcome: 'dag + honest_partial + honest_failure + recovery sources only (see trace.meta.outcomeVerified) -- hand_authored and tier23 tasks are single-step, where the tool return value already IS the outcome. honest_partial is outcomeVerified=false BY DESIGN (the final craft is deliberately never attempted); the point is that the trace answer honestly reports partial progress instead of fabricating completion. recovery traces are outcomeVerified=true when the recovery genuinely succeeds and honestly false when it does not (e.g. no concrete missing ingredient could be identified) -- never forced either way.',
    },
  }
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2))

  console.log(`\nwrote ${traces.length}/${allTasks.length} verified action traces to ${outPath}`)
  console.log(`source breakdown: ${JSON.stringify(sourceCounts)}`)
  console.log(`manifest written to ${manifestPath}`)
  if (integrityFailures) console.log(`${integrityFailures} traces dropped for failing the integrity check`)

  bot.quit()
  process.exit(0)
}

main().catch((err) => {
  console.error('GENERATION FAILED:', err.message)
  process.exit(1)
})
