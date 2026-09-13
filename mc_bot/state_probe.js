// Stage-3 verification probe: connects as a second player, asks the running
// bot.js for each of the 8 state tools plus one !needfor composition, logs
// every reply. Confirms end-to-end from the ASKER's side (bot.js's own log
// is the other half of the confirmation, same pattern as ask_probe.js).
const mineflayer = require('mineflayer')

const asker = mineflayer.createBot({
  host: '127.0.0.1', port: 25566, username: 'state_probe', version: '1.21.11',
})

const QUERIES = [
  '!state inventory',
  '!state position',
  '!state health_and_hunger',
  '!state time_of_day',
  '!state nearby_blocks 4',
  '!state nearby_entities 16',
  '!state equipped',
  '!state biome',
  '!needfor torch',
]

let i = 0
const replies = []

asker.on('spawn', () => {
  console.log('STATE PROBE SPAWNED -- sending', QUERIES.length, 'queries, one at a time')
  setTimeout(sendNext, 1500)
})

function sendNext() {
  if (i >= QUERIES.length) {
    console.log('ALL QUERIES SENT -- waiting 3s for trailing replies then exiting')
    setTimeout(() => {
      console.log(`DONE -- got ${replies.length}/${QUERIES.length} replies (chat's line-length limit can split one long reply into two, so >= counts as success)`)
      asker.quit()
      process.exit(replies.length >= QUERIES.length ? 0 : 1)
    }, 3000)
    return
  }
  const q = QUERIES[i]
  console.log(`SENDING [${i + 1}/${QUERIES.length}]: ${q}`)
  asker.chat(q)
  i++
  setTimeout(sendNext, 2000)
}

asker.on('chat', (username, message) => {
  if (username === asker.username) return
  if (username !== 'stage2_probe') return
  console.log(`REPLY FROM [${username}]: ${message}`)
  replies.push(message)
})

setTimeout(() => {
  console.error(`TIMEOUT: only got ${replies.length}/${QUERIES.length} replies within 40s`)
  process.exit(1)
}, 40000)
