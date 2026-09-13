// Client for the Fabric mod's live-registry oracle HTTP endpoint
// (mc_mod/src/main/java/com/cmodels/mcagent/OracleHttpServer.java),
// added per the stage-6 architecture reply: "THE MOD IS THE ORACLE. THE BOT IS THE AGENT... Compose
// them: the mod SERVES the oracle over localhost HTTP; the bot consumes
// it." This module is the "bot consumes it" half.
//
// GET http://127.0.0.1:8421/builddb returns the SAME JSON the mod's own
// "/mcagent builddb" in-game command writes to disk -- items, tags,
// breeding_food, recipes, enchantments, jukebox_songs, paintings,
// item_use, villager_trades, loot -- sourced from the actual running
// game's live registries, not this repo's static minecraft.db. That's
// the real differentiated value: it's modpack-aware, which the offline
// DB fundamentally cannot be.
//
// Deliberately thin: no caching, no retry logic, no schema validation
// beyond what JSON.parse already gives for free. Bound to 127.0.0.1 by
// the mod itself (same local-only posture as KNOWLEDGE_SERVER_BASE in
// bot.js), so this only works when mc_bot and the Fabric server share a
// host -- true for every setup this project has used so far.

const MOD_ORACLE_BASE = 'http://127.0.0.1:8421'

async function fetchModOracle() {
  const res = await fetch(`${MOD_ORACLE_BASE}/builddb`)
  if (!res.ok) throw new Error(`mod oracle HTTP ${res.status} on /builddb`)
  return res.json()
}

// Real use, not just a bare fetch: given a live oracle dump (from
// fetchModOracle()) and a block/table source id (e.g.
// "minecraft:blocks/diamond_ore"), return the drop_item names for that
// source -- direct lookup against the SAME loot table structure
// OracleDbBuilder.java's live registry walk produces, no LLM/model call
// involved (matches this project's own "verified, not recalled"
// principle for anything that touches the loot table).
function lootDropsForSource(oracle, source) {
  if (!oracle || !Array.isArray(oracle.loot)) return []
  return oracle.loot
    .filter((row) => row.source === source)
    .map((row) => row.drop_item)
}

module.exports = { MOD_ORACLE_BASE, fetchModOracle, lootDropsForSource }
