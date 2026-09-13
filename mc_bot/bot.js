// Stage 2 (knowledge-only) + stage 3 (read-only state) + stage 4 (action
// tools, tier 1 auto-execute + tier 2/3 confirm-gated) bot: connects to the
// local test server. "!ask <question>" queries stage 1's knowledge server
// and relays the real, oracle-verified (or safely-declined) answer back into
// chat. "!state <tool>" reports real, live Mineflayer state (stage 3,
// state_tools.js) -- no writes. "!needfor <item>" combines a verified recipe
// with real inventory. "!action <tool> <args...>" runs a TIER-1 action tool
// (action_tools.js) immediately -- no terrain change, no combat, per the
// reviewed stage-4 design. "!act2 <tool>
// <args...>" REQUESTS a tier-2/3 action (mine/place/chest_deposit/
// chest_withdraw/attack) via action_dispatcher.js -- produces a dry-run +
// nonce only, never executes directly; "!confirm <nonce>" is the only path
// that can execute it. Tier 2/3 are default-off; see enableTier below.
// "!modloot <source>" is the "bot consumes it" half of the reviewer's stage-6
// architecture reply: fetches LIVE, modpack-aware loot data
// from the Fabric mod's OracleHttpServer over HTTP
// (mod_oracle_client.js) instead of this repo's static minecraft.db.
const mineflayer = require('mineflayer')
const { pathfinder, Movements } = require('mineflayer-pathfinder')
const stateTools = require('./state_tools')
const actionTools = require('./action_tools')
const { makeDispatcher } = require('./action_dispatcher')
const modOracle = require('./mod_oracle_client')

const dispatcher = makeDispatcher()
// DEV/TEST-ONLY DEFAULT: enabled here so this single-operator loopback
// bot-dev server can be exercised end to end without a separate enable step.
// This is NOT a real permission model -- there is no check on who is
// allowed to have called enableTier, matching the confirm-scoping caveat
// already flagged for tier 1/2's design. A shared or public server needs
// stage 6's real permission work before tier 2/3 should ever be turned on
// there.
dispatcher.enableTier(2, true)
dispatcher.enableTier(3, true)

// Tier-2/3 tools take positional args after the tool name, same convention
// as ACTION_ARG_PARSERS below. Order matches action_tools_tier2.js's
// resolve* signatures (see action_dispatcher.js's RESOLVERS).
const ACT2_ARG_ARITY = {
  mine: 3,       // x y z
  place: 4,      // item x y z
  attack: 1,     // targetName
  chest_deposit: 2,   // item count
  chest_withdraw: 2,  // item count
}

// REAL BUG FOUND LIVE: several !action/!act2 replies fired back-to-back (a
// realistic case -- a player issuing a few quick commands, or a probe
// script) got the bot KICKED FOR SPAMMING by the server's own vanilla chat
// rate limit -- confirmed in the server log, not assumed. queuedChat() calls
// now go through a small queue with a minimum gap between sends instead of
// firing immediately, so a burst of replies is paced out rather than tripping
// the server's own throttle.
const CHAT_MIN_GAP_MS = 800
let chatQueue = Promise.resolve()
function queuedChat(text) {
  chatQueue = chatQueue.then(() => new Promise((resolve) => {
    bot.chat(text)
    setTimeout(resolve, CHAT_MIN_GAP_MS)
  }))
  return chatQueue
}

// NOTE: connects to the SEPARATE bot-dev server (real v1.21.11, port 25566),
// not the main 26.2 oracle-matched server (port 25565) -- mineflayer's own
// gameplay implementation tops out at 1.21.11 (a real npm package can't yet
// speak this project's simulated-future game version). This server exists
// purely to prove bot plumbing; the oracle/knowledge-server design is
// version-agnostic by intent (rebuilt per game version) so this doesn't
// conflict with the 26.2 fact-pipeline work.
const bot = mineflayer.createBot({
  host: '127.0.0.1',
  port: 25566,
  username: 'stage2_probe',   // offline-mode server: any username works
  version: '1.21.11',
})

bot.loadPlugin(pathfinder)
bot.once('spawn', () => bot.pathfinder.setMovements(new Movements(bot)))

// Tier-1 action tools take positional args after the tool name; this table
// parses the raw chat tokens into the right types per tool. Order matters --
// it must match action_tools.js's own parameter order.
//
// DENY-BY-DEFAULT (per the review): only tools listed
// here are reachable at all -- anything not in this table is refused, never
// auto-executed. chest_deposit/chest_withdraw are implemented in
// action_tools.js but deliberately NOT listed here: the reviewer's strongest
// pushback was that they're shared, irreversible-without-backup state (the
// top griefing surface after block-breaking), which belongs in tier 2 behind
// a confirm gate, not tier 1's "no terrain change" test. They wait on tier
// 2's confirm-gate design landing.
const ACTION_ARG_PARSERS = {
  goto: (a) => [Number(a[0]), Number(a[1]), Number(a[2]), a[3] !== undefined ? Number(a[3]) : undefined],
  goto_block: (a) => [a[0], a[1] !== undefined ? Number(a[1]) : undefined, a[2] !== undefined ? Number(a[2]) : undefined],
  follow: (a) => [a[0], a[1] !== undefined ? Number(a[1]) : undefined, a[2] !== undefined ? Number(a[2]) : undefined],
  equip: (a) => [a[0], a[1]],
  drop: (a) => [a[0], Number(a[1])],   // value-guarded in action_tools.js -- refuses enchanted/netherite/diamond gear
  eat: (a) => [a[0]],
  sleep: () => [],
  craft: (a) => [a[0], a[1] !== undefined ? Number(a[1]) : undefined],
  smelt: (a) => [a[0], a[1], a[2] !== undefined ? Number(a[2]) : undefined],
}

// Stage 1's knowledge server (assumed running on the same box).
const KNOWLEDGE_SERVER_BASE = 'http://127.0.0.1:8420'

async function postJson(path, body) {
  const res = await fetch(`${KNOWLEDGE_SERVER_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`knowledge server HTTP ${res.status} on ${path}`)
  return res.json()
}

const askKnowledgeServer = (question) => postJson('/ask', { question })
const lookupRecipe = (item) => postJson('/lookup_recipe', { item })

// "!needfor <item>": lookup_recipe (verified, stage 1) + local inventory()
// (real, stage 3) -- diffs ingredients the player is missing. Both halves are
// queried/read, never recalled from the model's own memory.
async function needFor(item) {
  const { ingredients, found } = await lookupRecipe(item)
  if (!found) return `I don't have a verified recipe for ${item}.`
  const have = {}
  for (const it of stateTools.inventory(bot)) {
    have[it.name] = (have[it.name] || 0) + it.count
  }
  const missing = ingredients.filter((ing) => !(have[ing.replace('minecraft:', '')] > 0))
  if (missing.length === 0) return `${item} needs: ${ingredients.join(', ')} -- you have all of it.`
  return `${item} needs: ${ingredients.join(', ')} -- missing: ${missing.join(', ')}`
}

bot.on('spawn', () => {
  clearTimeout(spawnTimeout)
  console.log('SPAWNED: bot joined the world at', bot.entity.position)
  console.log('game version reported by server:', bot.version)
  console.log('Ready -- "!ask", "!state <tool>", "!needfor <item>", "!modloot <source>", ' +
    '"!action <tool> <args>" (tier 1), or "!act2 <tool> <args>" + "!confirm <nonce>" (tier 2/3) in chat.')
})

bot.on('chat', async (username, message) => {
  if (username === bot.username) return
  console.log(`CHAT [${username}]: ${message}`)

  if (message.startsWith('!ask ')) {
    const question = message.slice('!ask '.length)
    try {
      const { answer, tool_calls } = await askKnowledgeServer(question)
      console.log(`  -> tool_calls=${JSON.stringify(tool_calls)}`)
      queuedChat(answer)
    } catch (err) {
      console.error('  knowledge server error:', err.message)
      queuedChat("(couldn't reach the knowledge server)")
    }
    return
  }

  if (message.startsWith('!state ')) {
    const [name, arg] = message.slice('!state '.length).split(' ')
    const fn = stateTools[name]
    if (!fn) {
      queuedChat(`unknown state tool "${name}". Try: ${Object.keys(stateTools).join(', ')}`)
      return
    }
    try {
      const result = arg !== undefined ? fn(bot, Number(arg)) : fn(bot)
      const summary = JSON.stringify(result)
      console.log(`  state.${name} -> ${summary}`)
      queuedChat(summary.length > 250 ? summary.slice(0, 250) + '...(truncated, see log)' : summary)
    } catch (err) {
      console.error(`  state.${name} error:`, err.message)
      queuedChat(`(state tool "${name}" errored: ${err.message})`)
    }
    return
  }

  if (message.startsWith('!needfor ')) {
    const item = message.slice('!needfor '.length)
    try {
      const answer = await needFor(item)
      console.log(`  needfor(${item}) -> ${answer}`)
      queuedChat(answer)
    } catch (err) {
      console.error('  needfor error:', err.message)
      queuedChat("(couldn't check that -- knowledge server or inventory read failed)")
    }
    return
  }

  if (message.startsWith('!modloot ')) {
    // "!modloot <source>": the "bot consumes the mod's oracle" half of
    // the reviewer's stage-6 (d) design -- fetches the Fabric mod's LIVE,
    // modpack-aware loot data over HTTP (mod_oracle_client.js) instead of
    // this repo's static minecraft.db, so this stays correct even for a
    // modpack that changed loot tables from vanilla.
    const source = message.slice('!modloot '.length)
    try {
      const oracle = await modOracle.fetchModOracle()
      const drops = modOracle.lootDropsForSource(oracle, source)
      queuedChat(drops.length > 0
        ? `${source} can drop: ${drops.join(', ')} (live from the mod's own registries)`
        : `no loot rows found for source "${source}" in the mod's live oracle`)
    } catch (err) {
      console.error('  modloot error:', err.message)
      queuedChat("(couldn't reach the mod's oracle HTTP server -- is the Fabric mod's " +
        'OracleHttpServer running on this host?)')
    }
    return
  }

  if (message.startsWith('!action ')) {
    const [name, ...rawArgs] = message.slice('!action '.length).split(' ')
    const fn = actionTools[name]
    const parseArgs = ACTION_ARG_PARSERS[name]
    if (!fn || !parseArgs) {
      queuedChat(`unknown or gated action "${name}". Tier-1 tools: ${Object.keys(ACTION_ARG_PARSERS).join(', ')}`)
      return
    }
    try {
      const result = await fn(bot, ...parseArgs(rawArgs))
      const summary = JSON.stringify(result)
      console.log(`  action.${name}(${rawArgs.join(' ')}) -> ${summary}`)
      queuedChat(summary.length > 250 ? summary.slice(0, 250) + '...(truncated, see log)' : summary)
    } catch (err) {
      console.error(`  action.${name} error:`, err.message)
      queuedChat(`(action "${name}" errored: ${err.message})`)
    }
    return
  }

  // Tier-2/3: bot.js's dispatcher is always the DEFAULT (gated) constructor
  // -- request() here ONLY ever produces a dry-run + nonce, never executes.
  // request() is async (ungated-mode generation dispatchers execute inline
  // and need to await the tool call), hence the await here even though this
  // gated instance never itself awaits an execute() inside request(). All
  // gating (tier classification, enable flag, deny-by-default,
  // construction-level filtering) happens inside action_dispatcher.js.
  if (message.startsWith('!act2 ')) {
    const [name, ...rawArgs] = message.slice('!act2 '.length).split(' ')
    const arity = ACT2_ARG_ARITY[name]
    if (arity === undefined) {
      queuedChat(`unknown tier-2/3 action "${name}". Try: ${Object.keys(ACT2_ARG_ARITY).join(', ')}`)
      return
    }
    if (rawArgs.length < arity) {
      queuedChat(`"${name}" needs ${arity} arg(s), got ${rawArgs.length}`)
      return
    }
    const { ok, message: reply } = await dispatcher.request(bot, username, name, rawArgs)
    console.log(`  act2.${name}(${rawArgs.join(' ')}) -> ok=${ok} ${reply}`)
    queuedChat(reply)
    return
  }

  // The ONLY path that can ever execute a tier-2/3 action. Date.now() at the
  // moment THIS handler runs is used as the confirm message's timestamp --
  // chat events are delivered in the order the server sent them over a
  // single connection, so processing order is a sound proxy for arrival
  // order, and that's all the strictly-after check in the dispatcher needs.
  if (message.startsWith('!confirm ')) {
    const nonce = message.slice('!confirm '.length).trim()
    const confirmedAt = Date.now()
    dispatcher.confirm(username, nonce, confirmedAt).then(({ ok, message: reply }) => {
      console.log(`  confirm(${nonce}) -> ok=${ok} ${reply}`)
      queuedChat(reply.length > 250 ? reply.slice(0, 250) + '...(truncated, see log)' : reply)
    })
    return
  }
})

bot.on('error', (err) => {
  console.error('BOT ERROR:', err.message)
  process.exit(1)
})

bot.on('kicked', (reason) => {
  console.error('KICKED:', reason)
  process.exit(1)
})

const spawnTimeout = setTimeout(() => {
  console.error('TIMEOUT: never spawned within 30s')
  process.exit(1)
}, 30000)
