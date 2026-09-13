// Self-resupply via the creative-mode inventory API, bypassing mineflayer's
// own bot.creative.setInventorySlot/clearInventory entirely -- TWO real,
// independent bugs in that plugin were found live before landing on this.
// Factored out of gen_action_traces.js (where this was first built and
// debugged) so live_eval_action_tasks.js can reuse the exact same,
// already-fixed logic rather than re-deriving or duplicating it. Both are
// documented here so this ground isn't re-covered blind if the library is
// ever upgraded and the workaround needs re-checking.
//
// BUG 1 (harness logic, this project's own, found first): an earlier
// version tried to preserve accumulated crafted items across tasks (topping
// up only materials below a threshold, reusing existing slots). Over a long
// DAG-scaled run the inventory filled with crafted clutter until NO free
// slots remained for raw materials that had been fully consumed
// ("resupply: no free inventory slot for cactus/bamboo" in the live log),
// which then cascaded into DAG tasks failing for a HARNESS reason, not
// genuine tier-1-unreachability, silently corrupting gate 2's meaning for
// those traces. Fixed by clearing the whole inventory before every
// resupply -- also a STRICTER gate 2 as a side effect (no chance of an
// earlier task's leftovers accidentally satisfying a later one's craft
// check).
//
// BUG 2 (mineflayer's own, found while fixing bug 1's clearing step):
// bot.creative.setInventorySlot's default path arms a ONE-TIME
// `updateSlot:${slot}` listener (via a 'noAckOnCreateSetSlotPacket'
// server-feature branch, which this MC version has) whose callback does
// `newItem.itemId` with NO null check -- crashes the whole Node process the
// moment ANY unrelated inventory-update event lands on that slot index
// while the listener is armed, independent of whether calls are batched
// (bot.creative.clearInventory()'s own Promise.all) or serialized
// (confirmed live: same crash, same location, with a from-scratch
// sequential rewrite). Passing waitTimeout:0 avoids arming that listener --
// but leads straight into BUG 3.
//
// BUG 3 (mineflayer's own, found immediately after working around bug 2):
// the waitTimeout:0 early-return path never resets creativeSlotsUpdates[slot]
// back to false, so the SAME slot can never be set a second time for the
// rest of the bot's connection -- confirmed live: "Setting slot 10
// cancelled due to calling bot.creative.setInventorySlot(10, ...) again" on
// the very next resupply cycle.
//
// REAL FIX for both: talk to the protocol layer directly, bypassing
// bot.creative's wrapper (and its private, unreachable creativeSlotsUpdates
// state) entirely -- the same category of workaround this project already
// used for the placeBlock sequence-counter bug (a bot._client.write()
// override, no node_modules edits). set_creative_slot is the real packet
// bot.creative.setInventorySlot itself sends internally (confirmed by
// reading creative.js); bot._setSlot is the same local-state-update call
// creative.js uses for its own noAckOnCreateSetSlotPacket branch. Sending
// both directly gets the identical real effect with none of the buggy
// bookkeeping around it.

// Target stock (per raw material) resupplyRawMaterials() gives after
// clearing the inventory. Generous relative to a typical CRAFT_BATCH so a
// single deep chain (several batched intermediate crafts) doesn't run dry
// mid-task.
const RESUPPLY_TARGET = 64

function setCreativeSlotDirect(bot, Item, slot, item) {
  bot._client.write('set_creative_slot', { slot, item: Item.toNotch(item) })
  bot._setSlot(slot, item)
}

async function resupplyRawMaterials(bot, rawMaterials) {
  const Item = require('prismarine-item')(bot.registry)
  const mcData = require('minecraft-data')(bot.version)
  for (const item of bot.inventory.slots.filter((it) => it)) {
    setCreativeSlotDirect(bot, Item, item.slot, null)
  }
  for (const raw of rawMaterials) {
    const bare = raw.replace('minecraft:', '')
    const entry = mcData.itemsByName[bare]
    if (!entry) {
      console.error(`resupply: unknown item ${bare} in this version's registry, skipping`)
      continue
    }
    const slot = bot.inventory.firstEmptySlotRange(bot.inventory.inventoryStart, bot.inventory.inventoryEnd)
    if (slot == null) {
      console.error(`resupply: no free inventory slot for ${bare}`)
      continue
    }
    setCreativeSlotDirect(bot, Item, slot, new Item(entry.id, RESUPPLY_TARGET))
  }
  // Direct writes update local state synchronously but the server's own ack
  // (if any) is genuinely async -- a short settle delay before the caller
  // starts crafting keeps this from racing a real server-side rejection
  // into a false "no such item" failure downstream.
  await new Promise((r) => setTimeout(r, 250))
}

// Real crafting_table/furnace/chest fixture, placed near trace_gen's actual
// spawn on this world (via a one-off fixture_setup connection, same pattern
// as setup_fixtures.js -- not committed as a reusable script since exact
// coordinates are tied to this specific world's terrain and would need
// re-placing after any world reset anyway). Found live, 2026-08-21, that
// trace_gen's own PERSISTED position (from much earlier in this world's
// history) was ~35 blocks and ~35 Y-levels away from where a freshly-placed
// fixture landed -- different accounts can have very different saved
// positions even in the "same" world, and /tp from an opped account is a
// no-op against an OFFLINE player, so this can't be fixed once outside the
// connecting script. Callers should flyTo this at startup instead of
// assuming wherever the account happens to already be.
const FIXTURE_LOCATION = { x: -31, y: 63, z: -137 }

// SECOND REAL BUG FOUND LIVE (2026-08-22, live_eval_action_tasks.js), root-
// caused by reading mineflayer's own source: calling relocateToFixture a
// second time immediately before a task (not just once at startup, as
// originally designed) HUNG INDEFINITELY -- confirmed via the brain
// server's own request count staying frozen for 2+ minutes with the Node
// process still alive (not a stdout-buffering illusion; the underlying
// request count genuinely stopped growing). Root cause, from
// lib/plugins/creative.js's flyTo(): its LAST step unconditionally does
// `await once(bot, 'move', /* no timeout */ 0)` after setting
// bot.entity.position = destination directly -- with NO timeout of its
// own. mineflayer's 'move' event only fires on an ACTUAL positional
// change; if the bot is ALREADY at (or extremely close to) the
// destination -- exactly the case here, since task 1 already left it
// right next to the fixture -- that assignment produces no real delta,
// 'move' never fires, and the await hangs forever. Fixed by wrapping the
// whole flyTo call in a timeout, same defensive pattern as action_tools.js's
// craft() fix earlier this round.
const RELOCATE_TIMEOUT_MS = 5000

async function relocateToFixture(bot) {
  const { Vec3 } = require('vec3')
  const flyPromise = bot.creative.flyTo(new Vec3(FIXTURE_LOCATION.x, FIXTURE_LOCATION.y, FIXTURE_LOCATION.z))
  flyPromise.catch(() => {}) // avoid an unhandled rejection if the timeout below wins
  await Promise.race([
    flyPromise,
    new Promise((resolve) => setTimeout(resolve, RELOCATE_TIMEOUT_MS)),
    // Resolves (not rejects) on timeout -- if the bot is already at the
    // destination, "did nothing more" is the correct outcome, not a
    // reportable failure the way a stuck craft()/tool call would be.
  ])
  // Settle delay for the SEPARATE issue found alongside this one: even
  // when flyTo does complete normally (a real move happened), the
  // server's corresponding chunk data for the new area is a genuinely
  // async delivery that isn't guaranteed to have landed the instant the
  // client-side 'move' event fires -- confirmed live (bot.findBlock()
  // failed to find a crafting_table it had just successfully used moments
  // earlier). Matches resupplyRawMaterials's own existing 250ms pattern
  // for the same class of "direct/fast client update, real network ack is
  // still async" issue.
  await new Promise((r) => setTimeout(r, 1000))
}

module.exports = { RESUPPLY_TARGET, setCreativeSlotDirect, resupplyRawMaterials, FIXTURE_LOCATION, relocateToFixture }
