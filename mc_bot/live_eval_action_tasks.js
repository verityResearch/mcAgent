// Stage 5, the actual ask: "same two
// gates, held-out tasks the generator never sourced" -- genuine LIVE
// re-execution evaluation. Earlier signals this round were real but
// weaker: prepare_action_corpus.py's held-out training loss (aggregate,
// not action-specific) and probe_action_first_hop.py's teacher-forced
// per-turn comparison (compares against REAL ground truth, not live
// re-execution against an uncertain world). This script is the real
// thing: the trained model drives real tool calls against a live
// Minecraft server, and gate 2 (outcome) is checked against real
// inventory state, not a prediction.
//
// This is the "body" half of the split -- tool_oracle_cuda/
// action_eval_server.py is the stateless "brain" half (one HTTP call per
// hop, no loop, no world access). This script owns the loop: POST the
// real conversation so far, get the model's next turn, parse whatever
// tool call it proposes, EXECUTE it for real via action_tools.js, feed
// the REAL result back as the next turn, repeat until the model stops
// calling tools or a hop limit is hit. Mirrors this project's own
// "mod is the oracle, bot is the agent" split (the reviewer's stage-6
// architecture reply) at a smaller scale.
//
// Recomputes the SAME 15 held-out DAG tasks prepare_action_corpus.py and
// probe_action_first_hop.py both use (same stratification: 3 per bucket,
// skipping buckets with population <= 1), so this evaluates exactly the
// targets the trained model never saw a trace for -- not a fresh
// approximation.
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const fs = require('fs')
const actionTools = require('./action_tools')
const { resupplyRawMaterials, relocateToFixture } = require('./creative_resupply')

const HOST = '127.0.0.1'
const PORT = 25566
const USERNAME = 'action_eval' // separate account from trace_gen/stage2_probe -- doesn't disturb either
const BRAIN_SERVER = 'http://127.0.0.1:8423'
// Needed after the trace-shape fix: training traces
// now include a real lookup(recipe, result_item=...) step before every
// craft(), so a live-driven model legitimately emits lookup() calls too --
// this harness must resolve them for real (same oracle DB the trace
// generator itself queried via recipe_dag.js's fetchAllRecipes), not just
// treat an unparsed lookup() as "no tool call, task done" (the bug that
// produced a false 0/50 the first time this ran post-fix: every transcript
// showed the model correctly emitting lookup() and the harness silently
// dropping it, toolCall: null, loop terminated after 1 hop).
const KNOWLEDGE_SERVER = 'http://127.0.0.1:8420'
// Overridable via env for the bucket-4 hop-budget check: the reviewer's ask -- "raise the hop budget and re-run
// those 10, no retraining, no new data, one parameter" -- to convert
// "likely hop-budget-bounded" from a hypothesis into a measured number.
// Default stays 10, matching every prior measurement in this track.
const MAX_HOPS = Number(process.env.MAX_HOPS) || 10
// Toggle for the missing-ingredient diff, off by default so the SAME script
// can produce a clean "diff disabled" run and a "diff enabled" run without
// editing code between them -- isolates the diff's own effect from the
// guard's, per the explicit order (noise floor -> diff -> re-measure).
const ENABLE_INGREDIENT_DIFF = process.env.ENABLE_INGREDIENT_DIFF === '1'

// MUST match gen_action_traces.js's own RAW_MATERIALS exactly -- these are
// the materials the trained model was taught it can rely on having, via
// the SAME resupply mechanism (creative_resupply.js) used during
// generation. Duplicated rather than imported because gen_action_traces.js
// is a run-on-require script (its own main().catch(...) executes
// immediately), not a library module -- importing it here would launch a
// second, unwanted trace-generation run.
const RAW_MATERIALS = [
  'minecraft:oak_log', 'minecraft:birch_log', 'minecraft:spruce_log', 'minecraft:jungle_log',
  'minecraft:acacia_log', 'minecraft:dark_oak_log', 'minecraft:mangrove_log', 'minecraft:cherry_log',
  'minecraft:sugar_cane', 'minecraft:bamboo', 'minecraft:cactus',
]

// Same syntax action_tools.js's own header comment describes: "action(tool,
// k='v', k2=42) -- mirrors lookup(table, key='...')'s syntax." Extracts the
// tool name plus an ORDERED list of argument VALUES (not a key->value map)
// -- gen_action_traces.js's own formatToolCall() always emits keys in the
// SAME order actionTools[tool]'s real positional parameters expect (e.g.
// craft: item, count in that order at every call site; goto_block: block,
// radius, range), confirmed by reading every recorder.call() site in
// gen_action_traces.js, so positionally reusing the VALUES in generated
// order is a real, checked mapping, not an assumption.
const ACTION_CALL_RE = /action\(\s*(\w+)\s*(?:,\s*(.*?))?\)/
const ARG_RE = /(\w+)\s*=\s*(?:'([^']*)'|(-?\d+(?:\.\d+)?))/g

function parseActionCall(text) {
  const m = ACTION_CALL_RE.exec(text)
  if (!m) return null
  const tool = m[1]
  const rawArgs = m[2] || ''
  const values = []
  let am
  ARG_RE.lastIndex = 0
  while ((am = ARG_RE.exec(rawArgs)) !== null) {
    values.push(am[2] !== undefined ? am[2] : Number(am[3]))
  }
  return { tool, values, rawArgs }
}

// Mirrors gen_action_traces.js's formatLookupCall() shape exactly:
// lookup(table, key='value', ...). Only 'recipe' is ever trained/emitted
// (round 11's own knowledge lookups aren't part of this action-task loop),
// but the parser doesn't assume that -- an unrecognized table just
// produces an honest "unknown lookup table" result rather than crashing.
const LOOKUP_CALL_RE = /lookup\(\s*(\w+)\s*(?:,\s*(.*?))?\)/

function parseLookupCall(text) {
  const m = LOOKUP_CALL_RE.exec(text)
  if (!m) return null
  const table = m[1]
  const rawArgs = m[2] || ''
  const namedArgs = {}
  let am
  ARG_RE.lastIndex = 0
  while ((am = ARG_RE.exec(rawArgs)) !== null) {
    namedArgs[am[1]] = am[2] !== undefined ? am[2] : Number(am[3])
  }
  return { table, namedArgs }
}

// Direct query against the same oracle DB the trace generator queried
// (knowledge_server.py's /lookup_recipe, recipe_dag.js's fetchAllRecipes
// data) -- returns the raw ingredient list (or null if not found), for
// both resolveLookupCall's own rendering and missingIngredientsDiff below.
async function fetchRecipeIngredients(item) {
  const res = await fetch(`${KNOWLEDGE_SERVER}/lookup_recipe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item }),
  })
  if (!res.ok) return null
  const { ingredients, found } = await res.json()
  return found && ingredients && ingredients.length > 0 ? ingredients : null
}

// Resolves a lookup() call against the real oracle -- a real, verified
// answer, not a re-derivation from the training corpus. Result is rendered
// as a comma-joined ingredient list, matching gen_action_traces.js's own
// recorder.lookup() format exactly, so the model sees the identical shape
// it was trained on.
async function resolveLookupCall(call) {
  if (call.table !== 'recipe') {
    return { resultText: `error: unknown lookup table '${call.table}'` }
  }
  const ingredients = await fetchRecipeIngredients(call.namedArgs.result_item)
  return { resultText: ingredients ? ingredients.join(', ') : `no recipe found for ${call.namedArgs.result_item}` }
}

// Missing-ingredient diff, per the read: bucket 2's dominant failure isn't wrong knowledge (the model
// already retrieves the correct full ingredient list every time) and isn't
// a lie (it honestly reports "could not craft X") -- it's that the model
// can't COMPUTE THE SET DIFFERENCE between what a recipe needs and what's
// in its own inventory. That's exactly the class of computation this whole
// project's thesis says belongs in a verified tool, not a model's head --
// so instead of retraining the model to do arithmetic over two lists, make
// the TOOL return the gap directly on a failed craft.
//
// PRESENCE-ONLY, not counts: confirmed live against the real oracle DB
// (SELECT ingredient FROM recipe WHERE result_item=...) that it stores one
// row per DISTINCT ingredient type, no per-grid-slot duplicates (e.g.
// acacia_fence -> exactly ['minecraft:acacia_planks','minecraft:stick'],
// two rows, not six even though a real fence needs 4 sticks+2 planks) -- so
// an exact "need N more" count is not derivable from this schema, only
// "you have zero of X." Tag-reference ingredients ('#minecraft:...') are
// skipped, not reported as missing -- resolving "does the bot have ANY
// member of this tag" needs a separate /tag_members query this diff
// doesn't make, and reporting a tag name as a bare missing item would be
// actively wrong (the bot may already satisfy it via a concrete member).
async function missingIngredientsDiff(bot, targetItem) {
  const ingredients = await fetchRecipeIngredients(targetItem)
  if (!ingredients) return null
  const missing = []
  for (const ing of ingredients) {
    if (ing.startsWith('#')) continue
    const bare = ing.replace('minecraft:', '')
    if (itemCount(bot, bare) === 0) missing.push(bare)
  }
  return missing.length > 0 ? missing : null
}

function itemCount(bot, bareName) {
  return bot.inventory.items().filter((it) => it.name === bareName)
    .reduce((s, it) => s + it.count, 0)
}

function loadJsonl(path) {
  return fs.readFileSync(path, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

// Mirrors prepare_action_corpus.py's held_out split EXACTLY -- same
// stratification, same ordering (first N of each bucket in file order).
function heldOutDagTasks(traces, heldoutPerBucket = 3) {
  const dag = traces.filter((t) => t.meta.source === 'dag')
  const byBucket = new Map()
  for (const t of dag) {
    const b = t.meta.depthBucket
    if (!byBucket.has(b)) byBucket.set(b, [])
    byBucket.get(b).push(t)
  }
  const heldout = []
  for (const bucket of [...byBucket.keys()].sort((a, b) => a - b)) {
    const items = byBucket.get(bucket)
    if (items.length <= 1) continue
    const n = Math.min(heldoutPerBucket, items.length - 1)
    heldout.push(...items.slice(0, n))
  }
  return heldout
}

// A tool result is a failure iff it carries a `reason` or `error` field --
// confirmed by reading every return site in action_tools.js: every failure
// path sets `reason` (or, for a thrown exception, `error`), and no success
// path ever includes either. Generic across all 9 tier-1 tools, not
// tool-specific pattern-matching.
function isFailureResult(result) {
  return result != null && (result.reason !== undefined || result.error !== undefined)
}

// Repeated-failed-call guard, per the read: the chest_boat name-collision failures burn their entire hop
// budget re-issuing the SAME failing call (craft(birch_boat) fails, model
// looks it up again, crafts it again, fails again...) because nothing tells
// the model it already tried this exact thing. This is not a knowledge or
// planning bug, it's missing error-state memory -- and it's fixable free,
// harness-side, with no retraining: if a call is byte-identical to one that
// already failed THIS trajectory, skip real execution and hand back an
// explicit "you already tried this, it failed" observation instead. Keyed
// on tool+values only (not lookup calls -- those aren't idempotency-broken
// the same way, and the reviewer's example was specifically a repeated craft()).
function callSignature(tool, values) {
  return `${tool}(${JSON.stringify(values)})`
}

async function nextTurn(messages) {
  const res = await fetch(`${BRAIN_SERVER}/next_turn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, max_tokens: 128 }),
  })
  if (!res.ok) throw new Error(`brain server HTTP ${res.status}`)
  const { content } = await res.json()
  return content
}

async function runLiveTask(bot, task) {
  const taskText = task.messages[0].content
  const targetItem = taskText.replace('Craft ', '').replace(/\.$/, '').replace(/ /g, '_')

  // REAL BUG FOUND LIVE: task 1 found+used the crafting_table fixture fine
  // (goto_block succeeded, craft succeeded), but every task after it
  // reported "no crafting table within 16 blocks" even though nothing in
  // this script moves the bot away or touches the block between tasks.
  // Re-relocating before EVERY task (not just once at startup) is a cheap,
  // defensive fix regardless of exact root cause -- a fresh flyTo also
  // forces a chunk re-sync around the new position as a side effect, which
  // may be masking a client-side world-model desync from creative_resupply's
  // raw packet writes. Root cause not fully isolated; documented honestly
  // rather than assumed understood.
  await relocateToFixture(bot)
  await resupplyRawMaterials(bot, RAW_MATERIALS)
  const before = itemCount(bot, targetItem)

  const convo = [{ role: 'user', content: taskText }]
  const transcript = []
  const failedCalls = new Map() // callSignature -> prior failure result, this trajectory only
  let hops = 0
  for (; hops < MAX_HOPS; hops++) {
    const content = await nextTurn(convo)

    const lookupCall = parseLookupCall(content)
    if (lookupCall !== null) {
      const { resultText } = await resolveLookupCall(lookupCall)
      transcript.push({ hop: hops, modelContent: content, lookupCall, resultText })
      convo.push({ role: 'assistant', content })
      convo.push({ role: 'user', content: `<result>${resultText}</result>` })
      continue
    }

    const call = parseActionCall(content)
    if (call === null) {
      // REAL BUG FOUND LIVE (bucket-4 hop-budget check, 2026-08-22): a
      // cornered model -- its known-good move just refused by the
      // repeated-failed-call guard -- sometimes emits a <tool_call> to a
      // function that doesn't exist anywhere in this project's action
      // space (observed: prompt(syntax_check,...), prompt(suggestCrafting
      // Steps,...)). Genuine final answers never contain a <tool_call>
      // wrapper (confirmed by reading every real final-answer string this
      // harness has produced), so that tag is a clean, safe discriminator
      // between "the model is actually done" and "the model tried to call
      // something that isn't real." The prior behavior silently treated
      // both as task-over, ending the trajectory on the FIRST hallucinated
      // call with no chance to recover -- even though this project's own
      // deny-by-default dispatch already contains the call harmlessly
      // (nothing unsafe executes), the harness was still wasting the rest
      // of the hop budget by giving up instead of telling the model what
      // went wrong. Fixed: an unrecognized-but-attempted tool call gets a
      // real, honest observation (not a fabricated one -- it's simply true
      // that no such function exists) and the loop continues.
      if (content.includes('<tool_call>')) {
        const resultText = 'error: unrecognized tool call -- only action(tool, k=v, ...) and ' +
          "lookup(table, k=v, ...) are real functions in this harness. No function by that name exists."
        transcript.push({ hop: hops, modelContent: content, unrecognizedToolCall: true, resultText })
        convo.push({ role: 'assistant', content })
        convo.push({ role: 'user', content: `<result>${resultText}</result>` })
        continue
      }
      // No tool call generated -- the model considers the task done (or
      // gave up). Record its final text and stop; gate 2 below is the
      // real judge of whether it's actually right.
      transcript.push({ hop: hops, modelContent: content, toolCall: null })
      convo.push({ role: 'assistant', content })
      break
    }

    const signature = callSignature(call.tool, call.values)
    let result
    let skippedAsRepeat = false
    if (failedCalls.has(signature)) {
      skippedAsRepeat = true
      result = {
        error: `you already tried ${signature} this trajectory and it failed with: ` +
          `${JSON.stringify(failedCalls.get(signature))}. Repeating the identical call will fail ` +
          `identically -- try something different (check the item name, or a different approach).`,
      }
    } else if (typeof actionTools[call.tool] !== 'function') {
      result = { error: `unknown tool: ${call.tool}` }
    } else {
      // Defense in depth on top of the real fix in action_tools.js's own
      // craft(): a live run found bot.craft() could hang indefinitely with
      // no timeout of its own, blocking the whole eval process for one
      // task forever. craft() is now fixed at the source, but this harness
      // shouldn't trust every OTHER tool call to never have the same class
      // of bug -- a hard per-call timeout here means a future undiscovered
      // hang degrades to one honestly-failed task, not a stuck process.
      const toolPromise = actionTools[call.tool](bot, ...call.values)
      toolPromise.catch(() => {}) // avoid an unhandled rejection if the timeout below wins
      try {
        result = await Promise.race([
          toolPromise,
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`${call.tool}() did not resolve within 30000ms`)), 30000)),
        ])
      } catch (err) {
        result = { error: `tool threw or timed out: ${err.message}` }
      }
    }

    // REAL BUG FOUND LIVE (2026-08-22, first run of adapter-tool-v15-
    // recovery-cuda): the guard was blocking a LEGITIMATE retry. The
    // recovery-trained model correctly does craft(target) [fails, missing
    // stick] -> craft(stick) [succeeds] -> craft(target) again -- but that
    // second craft(target) is byte-identical to the first, and the guard
    // intercepted it as a dumb repeat even though inventory had genuinely
    // changed in between (confirmed via transcript: 'you already tried
    // craft(["acacia_fence",1])...' fired right after a successful
    // craft(stick,8)). The guard has no notion of "the world changed since
    // that failure," so a correct fail->fix->retry sequence looks
    // identical to a dumb fail->repeat->repeat loop from its own signature-
    // only view. Fix: any SUCCESSFUL craft() invalidates all prior craft()
    // failure records -- a new craft materially changes inventory, so a
    // previously-failed craft() is no longer guaranteed to still fail.
    // Scoped to craft() specifically (not goto_block/equip/etc.) since
    // crafting is the only tool whose success precondition (inventory) this
    // harness is already tracking and reasoning about via the diff.
    if (call.tool === 'craft' && !skippedAsRepeat && !isFailureResult(result)) {
      for (const key of [...failedCalls.keys()]) {
        if (key.startsWith('craft(')) failedCalls.delete(key)
      }
    }

    if (!skippedAsRepeat && isFailureResult(result)) {
      failedCalls.set(signature, result)
      // Missing-ingredient diff, only on a REAL failed craft attempt (not a
      // guarded repeat, which already carries its own explicit message) --
      // ties the diagnostic to a genuine attempt, keeping the observation
      // honest rather than front-loading information the model hasn't
      // earned by trying yet.
      if (ENABLE_INGREDIENT_DIFF && call.tool === 'craft') {
        const missing = await missingIngredientsDiff(bot, `minecraft:${call.values[0]}`)
        if (missing) result.missing = missing
      }
    }

    transcript.push({
      hop: hops, modelContent: content, toolCall: { tool: call.tool, values: call.values }, result, skippedAsRepeat,
    })
    convo.push({ role: 'assistant', content })
    convo.push({ role: 'user', content: `<result>${JSON.stringify(result)}</result>` })
  }

  const after = itemCount(bot, targetItem)
  const outcomeVerified = after > before
  return { taskText, targetItem, depthBucket: task.meta.depthBucket, hops, outcomeVerified, before, after, transcript }
}

async function main() {
  const tracesPath = process.argv[2] || 'mc_bot_action_traces.jsonl'
  // Must match prepare_action_corpus.py's own --heldout-per-bucket for
  // whatever split actually trained the adapter being evaluated -- an
  // eval run against a DIFFERENT held-out set than the one the model was
  // actually excluded from is not a valid held-out measurement.
  const heldoutPerBucket = Number(process.argv[4]) || 3
  const traces = loadJsonl(tracesPath)
  const heldout = heldOutDagTasks(traces, heldoutPerBucket)
  console.log(`${heldout.length} held-out DAG tasks (heldout-per-bucket=${heldoutPerBucket})`)

  const healthRes = await fetch(`${BRAIN_SERVER}/health`)
  if (!healthRes.ok) throw new Error(`brain server not healthy: HTTP ${healthRes.status}`)
  console.log('brain server health:', await healthRes.json())

  const bot = mineflayer.createBot({ host: HOST, port: PORT, username: USERNAME, version: '1.21.11' })
  bot.loadPlugin(pathfinder)
  await new Promise((resolve, reject) => {
    bot.once('spawn', resolve)
    bot.once('error', reject)
    setTimeout(() => reject(new Error('spawn timeout')), 20000)
  })
  bot.pathfinder.setMovements(new Movements(bot))
  console.log('action_eval spawned at', bot.entity.position)
  await relocateToFixture(bot)
  console.log('relocated to fixture area at', bot.entity.position)

  const results = []
  for (const task of heldout) {
    console.log(`\n=== [bucket ${task.meta.depthBucket}] ${task.messages[0].content} ===`)
    try {
      const r = await runLiveTask(bot, task)
      results.push(r)
      console.log(`  outcomeVerified=${r.outcomeVerified} (${r.before} -> ${r.after}), hops=${r.hops}`)
    } catch (err) {
      console.error(`  TASK CRASHED: ${err.message}`)
      results.push({ taskText: task.messages[0].content, depthBucket: task.meta.depthBucket, crashed: true, error: err.message })
    }
  }

  const succeeded = results.filter((r) => r.outcomeVerified).length
  const crashed = results.filter((r) => r.crashed).length
  console.log(`\n=== SUMMARY ===`)
  console.log(`${succeeded}/${results.length} held-out tasks GATE-2 VERIFIED complete via real live execution`)
  console.log(`${crashed} tasks crashed (harness error, not a model failure)`)

  const outPath = process.argv[3] || 'live_eval_results.json'
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2))
  console.log(`full transcripts written to ${outPath}`)

  bot.quit()
  process.exit(0)
}

main().catch((err) => {
  console.error('LIVE EVAL FAILED:', err.message)
  process.exit(1)
})
