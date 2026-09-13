// ONE-OFF dev fixture setup, NOT part of the bot's action-tool surface.
// bot.placeBlock() hit an unresolved mineflayer/1.21.11 protocol quirk (the
// blockUpdate event never fired even with a confirmed held item and a valid
// adjacent face -- worth revisiting when tier-2 place() is designed, but not
// a blocker for tier-1 testing). Sidestepped here via vanilla /setblock and
// /give commands -- this account (fixture_setup) is opped on the bot-dev
// server ONLY (ops.json), purely so a developer can set up test fixtures
// quickly. Not how the bot itself will ever act.
const mineflayer = require('mineflayer')

const bot = mineflayer.createBot({
  host: '127.0.0.1', port: 25566, username: 'fixture_setup', version: '1.21.11',
})

function cmd(c) {
  return new Promise((resolve) => {
    bot.chat(c)
    setTimeout(resolve, 400)
  })
}

bot.on('spawn', async () => {
  try {
    console.log('FIXTURE BOT SPAWNED at', bot.entity.position, 'gamemode', bot.game.gameMode)
    // Build near stage2_probe's own (persisted, different) spawn point, not
    // this throwaway account's -- stage2_probe is the bot that will actually
    // execute the action tools.
    const p = { x: -32, y: 64, z: -153 }

    await cmd(`/setblock ${p.x + 1} ${p.y - 1} ${p.z} minecraft:crafting_table`)
    await cmd(`/setblock ${p.x - 1} ${p.y - 1} ${p.z} minecraft:furnace`)
    await cmd(`/setblock ${p.x} ${p.y - 1} ${p.z + 1} minecraft:chest`)
    await cmd(`/setblock ${p.x} ${p.y - 1} ${p.z - 2} minecraft:white_bed`)
    await cmd(`/give fixture_setup minecraft:oak_log 4`)
    await cmd(`/give fixture_setup minecraft:coal 8`)
    await cmd(`/give fixture_setup minecraft:raw_iron 4`)
    await cmd(`/give fixture_setup minecraft:bread 4`)
    await cmd(`/give fixture_setup minecraft:stick 4`)
    // give the SAME items to the answering bot (stage2_probe) -- it's the one
    // that will actually execute the action tools via chat commands.
    await cmd(`/give stage2_probe minecraft:oak_log 4`)
    await cmd(`/give stage2_probe minecraft:coal 8`)
    await cmd(`/give stage2_probe minecraft:raw_iron 4`)
    await cmd(`/give stage2_probe minecraft:bread 4`)
    await cmd(`/give stage2_probe minecraft:stick 4`)

    console.log('FIXTURES DONE:', JSON.stringify({
      spawnRelative: p,
      crafting_table: { x: p.x + 1, y: p.y - 1, z: p.z },
      furnace: { x: p.x - 1, y: p.y - 1, z: p.z },
      chest: { x: p.x, y: p.y - 1, z: p.z + 1 },
      white_bed: { x: p.x, y: p.y - 1, z: p.z - 2 },
    }))
    bot.quit()
    process.exit(0)
  } catch (err) {
    console.error('FIXTURE SETUP FAILED:', err.message)
    process.exit(1)
  }
})

bot.on('error', (err) => { console.error('BOT ERROR:', err.message); process.exit(1) })
setTimeout(() => { console.error('TIMEOUT'); process.exit(1) }, 40000)
