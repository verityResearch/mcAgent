// Central gate for tier-2 (mine, place, chest_deposit, chest_withdraw) and
// tier-3 (attack) actions. Per the design review:
//
// GUARD LIVES INSIDE THE DISPATCHER, NOT CALLER-SIDE. The only way to reach
// a tier-2/3 tool's execute() is through request() then confirm() below --
// mirrors verify_testbench_adequacy's placement inside admit_c_quad in the
// strict-C engine, so a new chat command or call path added later cannot
// accidentally bypass the gate the way a caller-side pre-check could.
//
// MONOTONIC. A pending action's requirements (who may confirm, that a
// dry-run is mandatory) are fixed at request() time and never loosened by
// anything in confirm(). There is no "upgrade" path.
//
// DEFAULT-OFF. Each tier starts disabled; nothing in this module turns
// itself on. A tier must be explicitly enabled (enableTier) before request()
// will produce anything but a refusal -- ship disabled, calibrate per-world
// by watching dry-runs, then enable.
//
// CONFIRM SCOPING: this whole flow is single-operator-loopback-server scope.
// enableTier() has no auth check of its own -- on a real multi-user server
// that is a real permission model the review and the plan both defer to stage 6,
// not something this module pretends to solve. Flagged here explicitly so
// it isn't assumed-solved by a future reader of just this file.
const crypto = require('crypto')
const tier2 = require('./action_tools_tier2')

const TOOL_TIER = {
  mine: 2,
  place: 2,
  chest_deposit: 2,
  chest_withdraw: 2,
  attack: 3,
}

const RESOLVERS = {
  mine: (bot, args) => tier2.resolveMine(bot, Number(args[0]), Number(args[1]), Number(args[2])),
  place: (bot, args) => tier2.resolvePlace(bot, args[0], Number(args[1]), Number(args[2]), Number(args[3])),
  attack: (bot, args) => tier2.resolveAttack(bot, args[0]),
  // chest_deposit/chest_withdraw reuse the tier-1 implementations from
  // action_tools.js (they were never terrain/combat-risky in themselves --
  // the reviewer's point was about the SHARED, UNRECOVERABLE-WITHOUT-BACKUP state,
  // which the confirm gate itself addresses regardless of which module
  // implements the mechanics). No dry-run "would" text needed since deposit/
  // withdraw don't change based on execution timing the way mine/place do --
  // still gated identically through request()/confirm() for consistency.
  chest_deposit: (bot, args) => {
    const tier1 = require('./action_tools')
    const itemName = args[0]
    const count = Number(args[1])
    return {
      tool: 'chest_deposit',
      describe: () => `would deposit ${count}x ${itemName} into the nearest chest`,
      execute: () => tier1.chest_deposit(bot, itemName, count),
    }
  },
  chest_withdraw: (bot, args) => {
    const tier1 = require('./action_tools')
    const itemName = args[0]
    const count = Number(args[1])
    return {
      tool: 'chest_withdraw',
      describe: () => `would withdraw ${count}x ${itemName} from the nearest chest`,
      execute: () => tier1.chest_withdraw(bot, itemName, count),
    }
  },
}

const CONFIRM_TIMEOUT_MS = 30000

// UNGATED MODE, per the design review: "don't
// invoke the confirm path at all during generation... the confirm gate is a
// property of the dispatcher, not a skill the model learns." This is NOT a
// runtime flag on a live dispatcher instance -- it is a SEPARATE
// CONSTRUCTOR CALL (makeDispatcher({mode: 'ungated'}) vs. the default
// makeDispatcher(), which is unchanged and still confirm-gated). There is
// no code path from a normal, confirm-gated dispatcher to an ungated one --
// bot.js (live serving) always uses the default constructor and can never
// reach ungated mode by any state transition, typo, or bug in caller code.
// Only a trace-generation script would ever call makeDispatcher({mode:
// 'ungated'}) explicitly. In ungated mode, request() resolves AND executes
// immediately (no dry-run text, no nonce, no pending map, no !confirm) --
// there is nothing to confirm because there is no human in the loop to ask.
// Deny-by-default and default-off-per-tier still apply: an ungated
// dispatcher still refuses unknown tools and still requires enableTier()
// before a tier will run, so a generation script has to explicitly opt a
// tier in, same discipline as live serving, just without the confirm step.
function makeDispatcher(opts = {}) {
  const mode = opts.mode === 'ungated' ? 'ungated' : 'gated'
  const enabled = { 2: false, 3: false }
  const pending = new Map() // nonce -> { username, resolved, createdAt, expiresAt } -- unused in ungated mode

  function enableTier(tier, on = true) {
    if (tier !== 2 && tier !== 3) throw new Error(`no such tier ${tier}`)
    enabled[tier] = !!on
  }

  function isEnabled(tier) {
    return !!enabled[tier]
  }

  function cleanupExpired() {
    const now = Date.now()
    for (const [nonce, p] of pending) {
      if (now > p.expiresAt) pending.delete(nonce)
    }
  }

  // request(): in GATED mode, the ONLY entry point that produces a dry-run
  // -- never calls execute() itself. In UNGATED mode, resolves AND executes
  // in the same call, since there is no confirm step to wait for.
  // DENY-BY-DEFAULT in both modes: any tool name not in TOOL_TIER is
  // refused outright.
  async function request(bot, username, tool, args) {
    cleanupExpired()
    const tier = TOOL_TIER[tool]
    if (!tier) return { ok: false, message: `unknown or unclassified action "${tool}" -- refusing (deny-by-default)` }
    if (!isEnabled(tier)) {
      return { ok: false, message: `tier-${tier} actions are disabled on this world (default-off until calibrated) -- an operator must enable it` }
    }

    let resolved
    try {
      resolved = RESOLVERS[tool](bot, args)
    } catch (err) {
      return { ok: false, message: `refused: ${err.message}` }
    }
    if (resolved.denied) return { ok: false, message: `refused: ${resolved.reason}` }

    if (mode === 'ungated') {
      try {
        const result = await resolved.execute()
        return { ok: true, message: JSON.stringify(result), describe: resolved.describe(), result }
      } catch (err) {
        return { ok: false, message: `execution error: ${err.message}` }
      }
    }

    const nonce = crypto.randomBytes(4).toString('hex')
    const createdAt = Date.now()
    pending.set(nonce, { username, resolved, createdAt, expiresAt: createdAt + CONFIRM_TIMEOUT_MS })
    return {
      ok: true,
      message: `${resolved.describe()} -- reply "!confirm ${nonce}" within ${CONFIRM_TIMEOUT_MS / 1000}s to execute, or ignore to auto-cancel`,
    }
  }

  // confirm(): the ONLY path that ever calls execute(). Enforces, in order:
  //   1. nonce exists and hasn't expired
  //   2. confirming username matches the ORIGINAL requester (unrelated
  //      players can't confirm someone else's pending action)
  //   3. confirmedAt is STRICTLY AFTER the dry-run's createdAt -- closes the
  //      pre-typed-confirm race the review flagged: a message composed/sent
  //      before the dry-run was ever printed cannot satisfy this even if it
  //      somehow carried a valid-looking nonce.
  // One-shot: the nonce is consumed (deleted) whether execution succeeds or
  // fails, so it can never be replayed.
  async function confirm(username, nonce, confirmedAt) {
    cleanupExpired()
    const p = pending.get(nonce)
    if (!p) return { ok: false, message: `no pending action for "${nonce}" (expired, wrong, or already used)` }
    if (p.username !== username) return { ok: false, message: `"${nonce}" belongs to a different player's pending action` }
    if (!(confirmedAt > p.createdAt)) {
      return { ok: false, message: 'confirm rejected: arrived no later than the dry-run itself (stale or pre-typed confirm)' }
    }
    pending.delete(nonce)
    try {
      const result = await p.resolved.execute()
      return { ok: true, message: JSON.stringify(result) }
    } catch (err) {
      return { ok: false, message: `execution error: ${err.message}` }
    }
  }

  function pendingCount() {
    cleanupExpired()
    return pending.size
  }

  return { request, confirm, enableTier, isEnabled, pendingCount }
}

module.exports = { makeDispatcher, TOOL_TIER, CONFIRM_TIMEOUT_MS }
