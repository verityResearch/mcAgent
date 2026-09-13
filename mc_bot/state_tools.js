// Stage 3: read-only perception tools, thin wrappers over Mineflayer's own
// tracked state. No writes, nothing invented -- every value here comes
// straight from packets the server actually sent this bot, the same
// verified-not-recalled principle as the oracle DB lookups in stage 1.
//
// Each function takes the live `bot` object and returns a plain JSON-safe
// value. Armor-slot indices (5-8) and hotbar convention follow mineflayer's
// documented Window/Inventory layout (stable since well before 4.x).

function inventory(bot) {
  return bot.inventory.items().map((it) => ({
    name: it.name,
    displayName: it.displayName,
    count: it.count,
    slot: it.slot,
  }))
}

function position(bot) {
  const p = bot.entity.position
  return { x: p.x, y: p.y, z: p.z }
}

function health_and_hunger(bot) {
  return {
    health: bot.health,
    food: bot.food,
    saturation: bot.foodSaturation,
  }
}

function time_of_day(bot) {
  return {
    timeOfDay: bot.time.timeOfDay,
    day: bot.time.day,
    isDay: bot.time.isDay,
  }
}

function nearby_blocks(bot, radius = 8) {
  const r = Math.max(1, Math.min(radius, 16)) // cap: a full-cube scan is O(r^3)
  const origin = bot.entity.position.floored()
  const found = []
  for (let dx = -r; dx <= r; dx++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dz = -r; dz <= r; dz++) {
        const pos = origin.offset(dx, dy, dz)
        const block = bot.blockAt(pos)
        if (!block || block.name === 'air') continue
        found.push({
          name: block.name,
          position: { x: pos.x, y: pos.y, z: pos.z },
          distance: bot.entity.position.distanceTo(pos),
        })
      }
    }
  }
  found.sort((a, b) => a.distance - b.distance)
  return found
}

function nearby_entities(bot, radius = 16) {
  return Object.values(bot.entities)
    .filter((e) => e !== bot.entity && e.position)
    .map((e) => ({
      type: e.type,
      name: e.name,
      username: e.username,
      position: { x: e.position.x, y: e.position.y, z: e.position.z },
      distance: bot.entity.position.distanceTo(e.position),
    }))
    .filter((e) => e.distance <= radius)
    .sort((a, b) => a.distance - b.distance)
}

function equipped(bot) {
  const slots = bot.inventory.slots
  const armorSlot = (i) => (slots[i] ? { name: slots[i].name, count: slots[i].count } : null)
  return {
    mainhand: bot.heldItem ? { name: bot.heldItem.name, count: bot.heldItem.count } : null,
    helmet: armorSlot(5),
    chestplate: armorSlot(6),
    leggings: armorSlot(7),
    boots: armorSlot(8),
  }
}

function biome(bot) {
  const block = bot.blockAt(bot.entity.position)
  if (!block || !block.biome) return null
  // block.biome.name is empty on this mineflayer/version combo (only the
  // numeric id is populated) -- resolve the real name via minecraft-data
  // instead of reporting a blank string.
  const mcData = require('minecraft-data')(bot.version)
  const entry = mcData.biomes[block.biome.id]
  return { name: entry ? entry.name : null, id: block.biome.id }
}

module.exports = {
  inventory,
  position,
  health_and_hunger,
  time_of_day,
  nearby_blocks,
  nearby_entities,
  equipped,
  biome,
}
