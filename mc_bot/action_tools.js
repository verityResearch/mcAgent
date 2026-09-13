// Stage 4, tier 1: auto-executing action tools -- no world-terrain change, no
// combat, so per the stage-4 design hashed out in review
// these run without a confirm gate. Tier 2 (mine/place) and tier 3 (attack)
// are NOT in this file -- they wait on the read of the proposal.
//
// Every function returns REAL post-state (per plan section 3.3: "that is
// what makes an action trace self-verifying"), never a bare success string.
// A failure is reported as a real, specific reason -- never silently
// swallowed or reported as success.
const { goals } = require('mineflayer-pathfinder')
const mcDataLoader = require('minecraft-data')
const stateTools = require('./state_tools')

function getMcData(bot) {
  return mcDataLoader(bot.version)
}

// mineflayer's bot.findBlock() uses an octahedron/chunk-section search whose
// radius rounds DOWN to whole 16-block sections -- confirmed empirically that
// a block 5.1 blocks away was NOT found at maxDistance:8 (only at 10+), even
// though it's well within the requested distance by a straight distanceTo
// check. Using 16 (mineflayer's own default) avoids that undersearch.
const NEARBY_BLOCK_SEARCH_RADIUS = 16

function itemCount(bot, name) {
  return bot.inventory.items().filter((it) => it.name === name).reduce((s, it) => s + it.count, 0)
}

// Hazard guard per the review: "pathfinding into lava/void = death =
// full inventory drop, your drop risk at maximum." mineflayer-pathfinder's
// Movements already avoids lava as unsafe terrain and caps voluntary drops
// at 4 blocks by default (verified by reading movements.js -- blocksToAvoid
// includes lava, maxDropDown=4) -- real protection inherited for free, not
// touched or loosened anywhere in this file. That's PLANNING-time avoidance
// though; it doesn't catch real harm happening mid-transit for any other
// reason (an unplanned lava tick, a mob hit, fire, drowning). This adds a
// live health-monitor: if health drops sharply DURING a goto call, abort
// movement immediately rather than letting the pathfinder run its course
// while the bot is actively taking damage.
const HAZARD_ABORT_HEALTH = 8 // out of 20 -- abort if health drops to/below this
const HAZARD_ABORT_HEALTH_DROP = 5 // abort on any single drop this large, even above the floor

async function goto(bot, x, y, z, range = 1) {
  const goal = new goals.GoalNear(x, y, z, range)
  const startHealth = bot.health
  let lastHealth = bot.health
  let hazardAbort = null
  const onHealth = () => {
    const h = bot.health
    if (h <= HAZARD_ABORT_HEALTH || (lastHealth - h) >= HAZARD_ABORT_HEALTH_DROP) {
      hazardAbort = { healthAtAbort: h, droppedFrom: lastHealth }
      bot.pathfinder.setGoal(null)
    }
    lastHealth = h
  }
  bot.on('health', onHealth)
  try {
    await bot.pathfinder.goto(goal)
  } catch (err) {
    // pathfinder throws on failure/timeout/interruption -- fall through and
    // report the REAL end position rather than treating this as fatal.
  } finally {
    bot.removeListener('health', onHealth)
  }
  const p = bot.entity.position
  const result = {
    target: { x, y, z },
    reached: p.distanceTo({ x, y, z }) <= range + 0.5,
    position: { x: p.x, y: p.y, z: p.z },
    health: bot.health,
  }
  if (hazardAbort) {
    result.reached = false
    result.hazardAborted = true
    result.hazardReason = `health dropped from ${hazardAbort.droppedFrom} to ${hazardAbort.healthAtAbort} during travel -- stopped moving rather than continuing into whatever caused it`
  } else if (bot.health < startHealth) {
    // Took some damage but not enough to trigger an abort -- surfaced
    // honestly rather than silently dropped, without treating it as fatal.
    result.tookDamage = true
  }
  return result
}

async function goto_block(bot, blockType, radius = 32, range = 1) {
  const mcData = getMcData(bot)
  const entry = mcData.blocksByName[blockType]
  if (!entry) return { found: false, reason: `unknown block type "${blockType}"` }
  const block = bot.findBlock({ matching: entry.id, maxDistance: radius })
  if (!block) return { found: false, reason: `no ${blockType} within ${radius} blocks` }
  const res = await goto(bot, block.position.x, block.position.y, block.position.z, range)
  return {
    found: true,
    block: { name: block.name, position: { x: block.position.x, y: block.position.y, z: block.position.z } },
    ...res,
  }
}

async function follow(bot, username, range = 2, durationMs = 5000) {
  const target = Object.values(bot.entities).find((e) => e.username === username)
  if (!target) return { found: false, reason: `no player named "${username}" nearby` }
  bot.pathfinder.setGoal(new goals.GoalFollow(target, range), true)
  await new Promise((resolve) => setTimeout(resolve, durationMs))
  bot.pathfinder.setGoal(null)
  const p = bot.entity.position
  return {
    found: true,
    distance: target.position ? p.distanceTo(target.position) : null,
    position: { x: p.x, y: p.y, z: p.z },
  }
}

async function equip(bot, itemName, destination = 'hand') {
  const item = bot.inventory.items().find((it) => it.name === itemName)
  if (!item) return { equipped: false, reason: `no ${itemName} in inventory` }
  try {
    await bot.equip(item, destination)
  } catch (err) {
    return { equipped: false, reason: err.message }
  }
  return { equipped: true, state: stateTools.equipped(bot) }
}

// Value guard per the reviewer's stage-4 review: a dropped item
// despawns in ~5 minutes with no undo, and the bot's inventory holds items
// the PLAYER handed it -- so this stays tier 1 only by refusing to drop
// anything expensive rather than moving the whole tool to a confirm gate.
// Best-effort: netherite/diamond gear is caught by name; enchantment
// detection checks the modern (1.20.5+) item-components list AND the legacy
// NBT tag, but hasn't been exhaustively verified against every storage shape
// this version might use -- a real gap to close before relying on it, not a
// promise it's airtight today.
function isProtectedFromDrop(item) {
  if (item.name.startsWith('netherite_')) return true
  const DIAMOND_GEAR = ['diamond_sword', 'diamond_pickaxe', 'diamond_axe', 'diamond_shovel',
    'diamond_hoe', 'diamond_helmet', 'diamond_chestplate', 'diamond_leggings', 'diamond_boots']
  if (DIAMOND_GEAR.includes(item.name)) return true
  if (Array.isArray(item.components) && item.components.some((c) =>
    c.type === 'minecraft:enchantments' || c.type === 'minecraft:stored_enchantments')) return true
  const nbtEnchants = item.nbt?.value?.Enchantments || item.nbt?.value?.StoredEnchantments
  if (nbtEnchants?.value?.value?.length > 0) return true
  return false
}

async function drop(bot, itemName, count) {
  const item = bot.inventory.items().find((it) => it.name === itemName)
  if (!item) return { dropped: 0, reason: `no ${itemName} in inventory` }
  if (isProtectedFromDrop(item)) {
    return { dropped: 0, reason: `refusing to drop ${itemName} -- looks like enchanted/netherite/diamond-tier gear (value guard, not a confirm gate)` }
  }
  const n = Math.min(count, item.count)
  try {
    await bot.toss(item.type, null, n)
  } catch (err) {
    return { dropped: 0, reason: err.message }
  }
  return { dropped: n, remaining: itemCount(bot, itemName) }
}

async function eat(bot, itemName) {
  const item = bot.inventory.items().find((it) => it.name === itemName)
  if (!item) return { ate: false, reason: `no ${itemName} in inventory` }
  try {
    await bot.equip(item, 'hand')
    await bot.consume()
  } catch (err) {
    return { ate: false, reason: err.message }
  }
  return { ate: true, state: stateTools.health_and_hunger(bot) }
}

async function sleep(bot) {
  const bedBlock = bot.findBlock({ matching: (b) => b.name.endsWith('_bed') || b.name === 'bed', maxDistance: 16 })
  if (!bedBlock) return { slept: false, reason: 'no bed within 16 blocks' }
  try {
    await bot.sleep(bedBlock)
  } catch (err) {
    return { slept: false, reason: err.message }
  }
  return { slept: true, isSleeping: bot.isSleeping, timeOfDay: bot.time.timeOfDay }
}

// REAL BUG FOUND LIVE (2026-08-22, during the stage-5 live model-driven
// eval harness): bot.craft() -- mineflayer's own core function, not this
// file's -- has NO built-in timeout, unlike smelt() below (which
// deliberately wraps its own furnace-wait in one). Confirmed live: a
// craft(bamboo_block) call HUNG INDEFINITELY (4+ minutes with zero
// progress, zero GPU activity, zero further model calls -- confirmed via
// nvidia-smi and the brain server's own request log both going silent)
// waiting on a windowOpen event that never fired, with no rejection ever
// thrown for the existing try/catch below to catch. This is a DIFFERENT
// failure mode from the "Error: Event windowOpen did not fire within
// timeout of 20000ms" seen in an earlier hand_authored trace -- THAT was a
// real, catchable rejection; THIS is a genuine hang with no timeout at
// all, evidently not universal to every craft() call. Root cause not
// further isolated (out of scope for a live-eval run to chase); fixed
// defensively by wrapping bot.craft() in the SAME Promise.race timeout
// pattern smelt() already used below, so this can never again silently
// block an entire generation or evaluation run -- a real, honest timeout
// failure is always better than an unbounded hang.
const CRAFT_TIMEOUT_MS = 20000

async function craft(bot, itemName, count = 1) {
  const mcData = getMcData(bot)
  const itemData = mcData.itemsByName[itemName]
  if (!itemData) return { crafted: 0, reason: `unknown item "${itemName}"` }
  const craftingTable = bot.findBlock({ matching: (b) => b.name === 'crafting_table', maxDistance: NEARBY_BLOCK_SEARCH_RADIUS })
  const recipes = bot.recipesFor(itemData.id, null, 1, craftingTable || null)
  if (!recipes.length) {
    return {
      crafted: 0,
      reason: `no known recipe for ${itemName} craftable from current inventory` +
        (craftingTable ? '' : ` (no crafting table within ${NEARBY_BLOCK_SEARCH_RADIUS} blocks -- some recipes need one)`),
    }
  }
  const before = itemCount(bot, itemName)
  try {
    const craftPromise = bot.craft(recipes[0], count, craftingTable || undefined)
    // If the timeout below wins the race, craftPromise is still pending and
    // may settle (resolve OR reject) later with nothing awaiting it --
    // without this, a late rejection would surface as an unhandled promise
    // rejection, a real crash risk this fix must not introduce while
    // fixing the original hang.
    craftPromise.catch(() => {})
    await Promise.race([
      craftPromise,
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`bot.craft() did not resolve within ${CRAFT_TIMEOUT_MS}ms (no windowOpen/craft-result event) -- treating as a real failure, not hanging`)), CRAFT_TIMEOUT_MS)),
    ])
  } catch (err) {
    return { crafted: 0, reason: err.message }
  }
  const after = itemCount(bot, itemName)
  return { crafted: after - before, totalNow: after }
}

async function smelt(bot, itemName, fuelName, timeoutMs = 20000) {
  const mcData = getMcData(bot)
  const inputData = mcData.itemsByName[itemName]
  const fuelData = mcData.itemsByName[fuelName]
  if (!inputData || !fuelData) return { smelted: 0, reason: 'unknown item or fuel name' }
  const inputItem = bot.inventory.items().find((it) => it.name === itemName)
  const fuelItem = bot.inventory.items().find((it) => it.name === fuelName)
  if (!inputItem) return { smelted: 0, reason: `no ${itemName} in inventory` }
  if (!fuelItem) return { smelted: 0, reason: `no ${fuelName} in inventory` }
  const furnaceBlock = bot.findBlock({ matching: (b) => b.name === 'furnace' || b.name === 'lit_furnace', maxDistance: NEARBY_BLOCK_SEARCH_RADIUS })
  if (!furnaceBlock) return { smelted: 0, reason: `no furnace within ${NEARBY_BLOCK_SEARCH_RADIUS} blocks` }

  const furnace = await bot.openFurnace(furnaceBlock)
  try {
    await furnace.putFuel(fuelData.id, null, 1)
    await furnace.putInput(inputData.id, null, inputItem.count)
    const output = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(null), timeoutMs)
      furnace.on('update', () => {
        const out = furnace.outputItem()
        if (out && out.count > 0) {
          clearTimeout(timer)
          resolve(out)
        }
      })
    })
    // Reproduced live: the setTimeout can fire a beat before furnace's
    // 'update' event delivers the finished output, which would otherwise
    // silently orphan real smelted items in the furnace (confirmed -- 4
    // raw_iron went in, 4 iron_ingot sat unclaimed after a false-negative
    // timeout). One last direct check before giving up closes that race.
    const finalCheck = output || furnace.outputItem()
    if (!finalCheck || finalCheck.count === 0) {
      furnace.close()
      return { smelted: 0, reason: `no output within ${timeoutMs}ms -- still smelting, or fuel/input rejected` }
    }
    const taken = await furnace.takeOutput()
    furnace.close()
    return { smelted: taken.count, item: taken.name }
  } catch (err) {
    try { furnace.close() } catch (_) { /* already closed */ }
    return { smelted: 0, reason: err.message }
  }
}

async function chest_deposit(bot, itemName, count) {
  const mcData = getMcData(bot)
  const itemData = mcData.itemsByName[itemName]
  if (!itemData) return { deposited: 0, reason: `unknown item "${itemName}"` }
  const item = bot.inventory.items().find((it) => it.name === itemName)
  if (!item) return { deposited: 0, reason: `no ${itemName} in inventory` }
  const chestBlock = bot.findBlock({ matching: (b) => b.name === 'chest' || b.name === 'trapped_chest', maxDistance: NEARBY_BLOCK_SEARCH_RADIUS })
  if (!chestBlock) return { deposited: 0, reason: `no chest within ${NEARBY_BLOCK_SEARCH_RADIUS} blocks` }
  const chest = await bot.openContainer(chestBlock)
  const n = Math.min(count, item.count)
  try {
    await chest.deposit(itemData.id, null, n)
  } catch (err) {
    chest.close()
    return { deposited: 0, reason: err.message }
  }
  chest.close()
  return { deposited: n, remainingInInventory: itemCount(bot, itemName) }
}

async function chest_withdraw(bot, itemName, count) {
  const mcData = getMcData(bot)
  const itemData = mcData.itemsByName[itemName]
  if (!itemData) return { withdrawn: 0, reason: `unknown item "${itemName}"` }
  const chestBlock = bot.findBlock({ matching: (b) => b.name === 'chest' || b.name === 'trapped_chest', maxDistance: NEARBY_BLOCK_SEARCH_RADIUS })
  if (!chestBlock) return { withdrawn: 0, reason: `no chest within ${NEARBY_BLOCK_SEARCH_RADIUS} blocks` }
  const chest = await bot.openContainer(chestBlock)
  const available = chest.containerItems().filter((it) => it.name === itemName).reduce((s, it) => s + it.count, 0)
  if (available === 0) {
    chest.close()
    return { withdrawn: 0, reason: `no ${itemName} in the chest` }
  }
  const n = Math.min(count, available)
  try {
    await chest.withdraw(itemData.id, null, n)
  } catch (err) {
    chest.close()
    return { withdrawn: 0, reason: err.message }
  }
  chest.close()
  return { withdrawn: n, nowInInventory: itemCount(bot, itemName) }
}

module.exports = {
  goto,
  goto_block,
  follow,
  equip,
  drop,
  eat,
  sleep,
  craft,
  smelt,
  chest_deposit,
  chest_withdraw,
}
