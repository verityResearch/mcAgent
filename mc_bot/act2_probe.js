// Stage-4 tier-2/3 verification probe: connects as a real player, exercises
// the full dispatcher gate against the running bot -- dry-run text, confirm
// flow, deny-list, construction-level player filtering, and (via a second
// connection) the wrong-player-confirm rejection. Same pattern as
// action_probe.js / state_probe.js.
const mineflayer = require('mineflayer')

const asker = mineflayer.createBot({ host: '127.0.0.1', port: 25566, username: 'act2_probe', version: '1.21.11' })
const outsider = mineflayer.createBot({ host: '127.0.0.1', port: 25566, username: 'act2_outsider', version: '1.21.11' })

let outsiderReady = false
outsider.on('spawn', () => { outsiderReady = true })

const results = []
function log(label, msg) {
  console.log(`[${label}] ${msg}`)
  results.push({ label, msg })
}

// The bot-dev server's vanilla chat rate limit kicked BOTH the answering bot
// and this probe's own account when commands fired back-to-back (confirmed
// live, see bot.js's queuedChat() fix). Pacing sends here too, on top of
// that fix, so the probe itself doesn't trip the same limit.
const SEND_GAP_MS = 1200
function say(bot, text) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const onChat = (u, m) => {
        if (u !== bot.username && u === 'stage2_probe') {
          bot.removeListener('chat', onChat)
          resolve(m)
        }
      }
      bot.on('chat', onChat)
      bot.chat(text)
      setTimeout(() => { bot.removeListener('chat', onChat); resolve(null) }, 8000)
    }, SEND_GAP_MS)
  })
}

function extractNonce(reply) {
  const m = reply && reply.match(/!confirm ([0-9a-f]+)/)
  return m ? m[1] : null
}

async function run() {
  await new Promise((r) => setTimeout(r, 2000))

  // 1. deny-list item refused immediately, no dry-run
  let r = await say(asker, '!act2 place lava_bucket 0 0 0')
  log('deny-list place', r)

  // 2. attack a player name -> construction-level refusal, not a confirm gate
  r = await say(asker, '!act2 attack act2_outsider')
  log('attack-player-refused', r)

  // 3. unknown tool -> deny-by-default
  r = await say(asker, '!act2 not_a_real_tool')
  log('unknown-tool', r)

  // 4. real mine dry-run + confirm. Ask bot's position/nearby first via state.
  const posReply = await say(asker, '!state position')
  log('bot-position', posReply)
  const nbReply = await say(asker, '!state nearby_blocks 6')
  log('nearby-blocks(truncated)', nbReply ? nbReply.slice(0, 200) : null)

  let pos
  try { pos = JSON.parse(posReply) } catch (e) { pos = null }
  let mineX, mineY, mineZ, minedOk = false
  if (pos) {
    mineX = Math.floor(pos.x)
    mineY = Math.floor(pos.y) - 1
    mineZ = Math.floor(pos.z)
    r = await say(asker, `!act2 mine ${mineX} ${mineY} ${mineZ}`)
    log('mine-dryrun', r)
    const nonce = extractNonce(r)
    if (nonce) {
      const confirmR = await say(asker, `!confirm ${nonce}`)
      log('mine-confirm-result', confirmR)
      minedOk = confirmR && confirmR.includes('"mined":true')
      // 5. replay the SAME nonce -> must be refused (one-shot)
      const replay = await say(asker, `!confirm ${nonce}`)
      log('mine-confirm-replay(should-fail)', replay)
    } else {
      log('mine-dryrun', 'NO NONCE EXTRACTED -- cannot test confirm')
    }
  }

  // 6. real place dry-run + confirm -- place dirt back into the hole just
  // mined above: guaranteed air (we just dug it) with solid ground below it
  // (untouched), so this doesn't depend on guessing flat-terrain offsets.
  if (minedOk) {
    r = await say(asker, `!act2 place dirt ${mineX} ${mineY} ${mineZ}`)
    log('place-dryrun', r)
    const nonce = extractNonce(r)
    if (nonce) {
      const confirmR = await say(asker, `!confirm ${nonce}`)
      log('place-confirm-result', confirmR)
    } else {
      log('place-dryrun', 'NO NONCE -- likely no dirt in inventory or spot not empty, see reply above')
    }
  }

  // 7. chest_deposit dry-run + confirm
  r = await say(asker, '!act2 chest_deposit stick 1')
  log('chest_deposit-dryrun', r)
  let nonce = extractNonce(r)
  if (nonce) {
    const confirmR = await say(asker, `!confirm ${nonce}`)
    log('chest_deposit-confirm-result', confirmR)
  }

  // 8. WRONG-PLAYER CONFIRM: outsider tries to confirm asker's pending action
  r = await say(asker, '!act2 chest_withdraw stick 1')
  log('chest_withdraw-dryrun(for-wrong-player-test)', r)
  nonce = extractNonce(r)
  if (nonce && outsiderReady) {
    const wrongConfirm = await say(outsider, `!confirm ${nonce}`)
    log('wrong-player-confirm(should-fail)', wrongConfirm)
    // clean up: the real requester confirms it for real afterward
    const realConfirm = await say(asker, `!confirm ${nonce}`)
    log('correct-player-confirm-after(should-succeed)', realConfirm)
  } else {
    log('wrong-player-confirm', `SKIPPED (nonce=${nonce}, outsiderReady=${outsiderReady})`)
  }

  // 9. attack a real hostile mob if one happens to be nearby
  const nearbyReply = await say(asker, '!state nearby_entities 12')
  log('nearby-entities', nearbyReply ? nearbyReply.slice(0, 300) : null)
  let hostileName = null
  try {
    const nearby = JSON.parse(nearbyReply)
    const hit = nearby.find((e) => e.type === 'hostile')
    if (hit) hostileName = hit.name
  } catch (e) { /* ignore */ }
  if (hostileName) {
    r = await say(asker, `!act2 attack ${hostileName}`)
    log('attack-hostile-dryrun', r)
    nonce = extractNonce(r)
    if (nonce) {
      const confirmR = await say(asker, `!confirm ${nonce}`)
      log('attack-hostile-confirm-result', confirmR)
    }
  } else {
    log('attack-hostile', 'no hostile mob currently visible in nearby_entities -- skipping real-attack test (attack-player-refused above already covers the construction-level guarantee)')
  }

  console.log('\n=== SUMMARY ===')
  for (const { label, msg } of results) console.log(`${label}: ${msg}`)
  asker.quit()
  outsider.quit()
  process.exit(0)
}

asker.on('spawn', () => run())
setTimeout(() => { console.error('GLOBAL TIMEOUT'); process.exit(1) }, 120000)
