// Stage 4, tier 2 (mine, place) + tier 3 (attack): construction-gated action
// tools, per the design review.
//
// The confirm/nonce/enable-flag GATE lives in action_dispatcher.js, not
// here -- these functions only RESOLVE an action into a {describe, execute}
// pair. Resolution happens exactly once per request; the dispatcher passes
// the SAME resolved object to both the dry-run text and (on confirm) the
// real execution, so the confirm prompt can never drift from what actually
// runs. Each execute() still re-checks the world hasn't changed since the
// dry-run (block/entity gone) and reports that honestly rather than acting
// on stale state.
//
// attack() is CONSTRUCTION, not confirm-gated policy: candidates are
// filtered by mineflayer's own entity.type classification (verified live on
// this server/version: spiders and skeletons report type:"hostile", a
// player reports type:"player" with name:"player" -- the username lives in
// entity.username, not entity.name). A player is filtered out at the
// candidate-list stage, before a target is even named, so the model cannot
// name one as a target in v1 -- there is no confirm step to bypass because
// there is no path that produces a player as a resolvable target at all.
//
// place()'s deny-list is checked at resolve() time, before any dry-run is
// even generated -- these items are categorically worse than placing stone
// (fire/flood risk) and don't belong under generic terrain-tier gating.
const { Vec3 } = require('vec3')
const { goals } = require('mineflayer-pathfinder')

const MELEE_RANGE = 3 // blocks -- bot.attack() is a melee swing, not a walk-and-hit composite

const PLACE_DENY_LIST = new Set([
  'lava_bucket', 'water_bucket', 'flint_and_steel', 'fire_charge', 'tnt', 'tnt_minecart',
])

function resolveMine(bot, x, y, z) {
  const pos = new Vec3(x, y, z)
  const block = bot.blockAt(pos)
  if (!block || block.name === 'air') {
    throw new Error(`no block at (${x},${y},${z}) to mine`)
  }
  const blockName = block.name
  return {
    tool: 'mine',
    describe: () => `would mine ${blockName} at (${x},${y},${z})`,
    execute: async () => {
      const current = bot.blockAt(pos)
      if (!current || current.name !== blockName) {
        return {
          mined: false,
          reason: `block at (${x},${y},${z}) changed since the dry-run (was ${blockName}, now ` +
            `${current ? current.name : 'air/unloaded'}) -- refusing to mine something different than what was shown`,
        }
      }
      const before = {}
      for (const it of bot.inventory.items()) before[it.name] = (before[it.name] || 0) + it.count
      try {
        await bot.dig(current)
      } catch (err) {
        return { mined: false, reason: err.message }
      }
      const after = {}
      for (const it of bot.inventory.items()) after[it.name] = (after[it.name] || 0) + it.count
      const collected = {}
      for (const k of new Set([...Object.keys(before), ...Object.keys(after)])) {
        const d = (after[k] || 0) - (before[k] || 0)
        if (d > 0) collected[k] = d
      }
      return { mined: true, blockName, position: { x, y, z }, collected }
    },
  }
}

// KNOWN LIMITATION, investigated in depth across two sessions, not just
// assumed: on this exact mineflayer / Minecraft-1.21.11-server combination,
// bot.placeBlock() reliably throws "Event blockUpdate:... did not fire
// within timeout of 5000ms" and the block genuinely never lands. RULED OUT,
// each with a real test, not by assumption:
//   - gamemode (set to creative, re-tested -- still fails)
//   - bad target (scanned for a genuinely-empty position with a solid
//     reference block below it, re-tested -- still fails)
//   - held item (confirmed via bot.heldItem right before the call)
//   - mineflayer/Node version mismatch (npm warned mineflayer@4.37.1 wants
//     node>=22 while the host ran v18.19.1 -- upgraded the host to Node 22
//     via NodeSource, clean node_modules reinstall, re-tested -- still fails
//     identically; already the latest published mineflayer, 4.37.1, so no
//     further version bump was available to try)
//   - a REAL bug found and tried as a fix, still insufficient alone:
//     mineflayer's own lib/plugins/generic_place.js hardcodes
//     `sequence: 0` on every block_place packet (comment: "// 1.19.0"),
//     while lib/plugins/inventory.js tracks a real incrementing sequence
//     for other interactions -- a genuine upstream inconsistency, confirmed
//     by reading the source. Patched a monotonically-increasing sequence
//     counter into the outgoing block_place packet (bypassing node_modules,
//     via a bot._client.write() override) and re-tested -- the server still
//     never acknowledges the placement (raw packet trace shows no
//     block_change or acknowledge_block_change for the target position at
//     all, with or without the patch). The hardcoded-sequence bug is real
//     and worth reporting upstream, but it is not, by itself, this
//     failure's root cause.
// ISOLATED to 1.21.11 specifically, via a real differential test: stood up
// a second, real vanilla 1.20.4 server (mineflayer's long-established
// version) on a separate port and ran the IDENTICAL placeBlock logic
// against it -- PLACE SUCCEEDED, block genuinely landed. This conclusively
// rules out anything about this project's dispatcher/action_tools code
// (same code path, only the server version differs) and confirms the bug
// is specific to mineflayer's 1.21.11 support.
//
// Followed up by comparing bot.supportFeature() branch selection (both
// versions select blockPlaceHasInsideBlock=true -- NOT a wrong-branch bug)
// and diffing the two versions' raw protocol.json block_place packet
// schemas directly. Found a real schema difference: 1.21.11 adds a
// `worldBorderHit` (bool) field between `insideBlock` and `sequence` that
// 1.20.4's schema doesn't have at all. Checked whether generic_place.js
// actually supplies it: it does (`worldBorderHit: false // 1.21.3`,
// present in the source) -- so the obvious "missing field" explanation
// doesn't hold either. Root cause remains unisolated at the field level;
// the server still gives zero error feedback (no block_change, no
// acknowledge_block_change) even with correct-looking fields present. Next
// step would be a byte-level diff of the actual outgoing packet on both
// connections rather than JS-object-level inspection -- out of scope for
// now given the depth already invested; this is a clean, well-evidenced
// candidate for an upstream mineflayer issue report (server-version-specific
// repro, differential test proving it's not project code, feature-branch
// and schema-field comparison both already done). The GATING here
// (deny-list check, dry-run text, target-occupied pre-check) is all still
// real and correct; only the final world-mutation call is currently
// non-functional on 1.21.11 specifically.
function resolvePlace(bot, itemName, x, y, z) {
  if (PLACE_DENY_LIST.has(itemName)) {
    throw new Error(`refusing to place ${itemName} -- on the hard deny-list (categorically hazardous: ` +
      `fire/flood risk), not something a confirm gate is trusted to police`)
  }
  const item = bot.inventory.items().find((it) => it.name === itemName)
  if (!item) throw new Error(`no ${itemName} in inventory`)
  const pos = new Vec3(x, y, z)
  const below = bot.blockAt(pos.offset(0, -1, 0))
  if (!below) throw new Error(`can't resolve a reference block below (${x},${y},${z}) -- chunk not loaded?`)
  const dest = bot.blockAt(pos)
  if (dest && dest.name !== 'air') {
    throw new Error(`(${x},${y},${z}) is not empty (currently ${dest.name})`)
  }
  return {
    tool: 'place',
    describe: () => `would place ${itemName} at (${x},${y},${z})`,
    execute: async () => {
      const freshDest = bot.blockAt(pos)
      if (freshDest && freshDest.name !== 'air') {
        return {
          placed: false,
          reason: `(${x},${y},${z}) is no longer empty (now ${freshDest.name}) -- refusing to place ` +
            `over something that wasn't there during the dry-run`,
        }
      }
      const freshItem = bot.inventory.items().find((it) => it.name === itemName)
      if (!freshItem) return { placed: false, reason: `no longer have ${itemName} in inventory` }
      const freshBelow = bot.blockAt(pos.offset(0, -1, 0))
      try {
        await bot.equip(freshItem, 'hand')
        await bot.placeBlock(freshBelow, new Vec3(0, 1, 0))
      } catch (err) {
        return { placed: false, reason: err.message }
      }
      return { placed: true, item: itemName, position: { x, y, z } }
    },
  }
}

function resolveAttack(bot, targetName) {
  // CONSTRUCTION: filter by mineflayer's own type classification, verified
  // live -- 'hostile' never includes players (players report type:'player').
  // The e.type !== 'player' check is redundant with that but kept explicit:
  // it documents the intent directly rather than relying only on 'hostile'
  // meaning the right thing forever.
  // NEAREST match, not first-in-iteration-order: confirmed live that a
  // large world can have multiple same-named hostile mobs loaded at once
  // (e.g. two wild spiders far apart), and Object.values(bot.entities)'s
  // order has no relationship to distance -- an earlier version of this
  // function picked whichever one iterated first and attacked an entity
  // 40+ blocks away instead of the one actually nearby, discovered via a
  // live test where the resolved target's own describe() position gave it
  // away. Sorting by distance before matching fixes that.
  const candidates = Object.values(bot.entities)
    .filter((e) => e !== bot.entity && e.type === 'hostile' && e.type !== 'player')
    .sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position))
  const target = candidates.find((e) => e.name === targetName)
  if (!target) {
    return {
      tool: 'attack',
      denied: true,
      reason: `no hostile mob named "${targetName}" nearby (player-targeting is not offered in v1 -- ` +
        `players are filtered out of the candidate list itself, not blocked by a confirm step)`,
    }
  }
  const entityId = target.id
  const targetLabel = target.name
  const targetPos = { x: target.position.x, y: target.position.y, z: target.position.z }
  return {
    tool: 'attack',
    describe: () => `would attack ${targetLabel} (entity ${entityId}) at ${JSON.stringify(targetPos)}`,
    execute: async () => {
      const current = bot.entities[entityId]
      if (!current) return { attacked: false, reason: 'target no longer present (despawned, died, or out of range)' }
      if (current.type === 'player') {
        // Should be structurally unreachable (entity ids aren't reassigned
        // across types), but a confirm gate must never trust a stale
        // reference blindly -- re-check the construction invariant at the
        // last possible moment too.
        return { attacked: false, reason: 'refusing: resolved target is now a player entity (should be unreachable)' }
      }

      // REAL BUG FOUND LIVE: bot.attack() is a melee swing at whatever
      // entity reference it's given -- it does NOT path to the target
      // first. A 15-hit live campaign against a real wild creeper ~19
      // blocks away returned {"attacked":true} on every single swing
      // (no error, matching all prior "verified" reports) while the
      // creeper's own tracked position never changed and no item drop
      // ever appeared -- conclusive proof every earlier "successful" call
      // was a no-op swing at nothing, not a real hit. Close to melee range
      // before swinging.
      const dist = bot.entity.position.distanceTo(current.position)
      if (dist > MELEE_RANGE) {
        try {
          await bot.pathfinder.goto(new goals.GoalNear(current.position.x, current.position.y, current.position.z, MELEE_RANGE - 1))
        } catch (err) {
          // pathfinder can fail/timeout on a moving or unreachable target --
          // fall through and try from wherever we ended up; the distance
          // check below reports honestly if that wasn't close enough.
        }
      }
      const freshTarget = bot.entities[entityId]
      if (!freshTarget) return { attacked: false, reason: 'target no longer present after moving into range (despawned, died, or fled)' }
      const finalDist = bot.entity.position.distanceTo(freshTarget.position)
      if (finalDist > MELEE_RANGE + 1) {
        return { attacked: false, reason: `could not close to melee range -- still ${finalDist.toFixed(1)} blocks away after pathfinding` }
      }
      try {
        await bot.attack(freshTarget)
      } catch (err) {
        return { attacked: false, reason: err.message }
      }
      return { attacked: true, target: targetLabel, entityId, distanceAtSwing: Number(finalDist.toFixed(2)) }
    },
  }
  // VERIFICATION STATUS, updated after the melee-range fix above: a 15-hit
  // live campaign against a real wild creeper found bot.attack() genuinely
  // was a no-op every single time before the fix (target ~19 blocks away,
  // never closed, no item drop, no position change across 15 confirmed
  // swings) -- root-caused to bot.attack() being a melee-range primitive
  // with no built-in pathing, unlike goto()/craft() which either path
  // explicitly or don't need to. The fix adds a GoalNear pathfinder step
  // before swinging.
  //
  // RE-VERIFIED LIVE after the fix, two campaigns (15 then 25 confirmed
  // swings) against real wild mobs: distanceAtSwing was consistently
  // 0.5-2.3 blocks across every single swing -- the bot now genuinely
  // closes to real melee range before attacking, a categorical fix from
  // the pre-fix state (every swing at 18-19 blocks, zero real hits). One
  // entity (a zombie) took 12 of the 25 confirmed real-range swings in the
  // second campaign and was later confirmed GONE from its last known
  // position via a direct server-side query (`/execute if entity ...`,
  // "No entity was found"). NOT fully confirmed as a bot-caused kill,
  // though: a wide-radius item-drop search around that position also came
  // back empty, which is more consistent with the zombie wandering off
  // than dying (a real zombie death should usually drop something). Both
  // wild-mob campaigns ran in a naturally dense mob cluster, which spread
  // hits across several individuals rather than concentrating on one --
  // real environmental variance, not a code issue. Net: the melee-range
  // mechanism is now proven correct with strong, repeated live evidence;
  // a clean single-target kill-to-drop confirmation was not captured this
  // round and would need either a less crowded test area or a weapon
  // dealing enough damage to guarantee a kill within a couple of hits.
}

module.exports = { resolveMine, resolvePlace, resolveAttack, PLACE_DENY_LIST }
