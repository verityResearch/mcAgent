// Stage-4 tier-1 verification probe: connects as a second player, sends each
// tier-1 !action command to the running bot, logs every reply. Same pattern
// as ask_probe.js / state_probe.js.
const mineflayer = require('mineflayer')

const asker = mineflayer.createBot({
  host: '127.0.0.1', port: 25566, username: 'action_probe', version: '1.21.11',
})

const QUERIES = [
  '!state inventory',
  '!action goto -28 64 -153 1',
  '!action goto_block furnace 16 1',
  '!action follow action_probe 2 3000',
  '!action craft oak_planks 4',
  '!action equip stick hand',
  '!action eat bread',
  '!action smelt raw_iron coal 15000',
  '!action chest_deposit stick 2',
  '!action chest_withdraw stick 1',
  '!action drop stick 1',
  '!action sleep',
]

let i = 0
const replies = []

asker.on('spawn', () => {
  console.log('ACTION PROBE SPAWNED -- sending', QUERIES.length, 'queries, one at a time')
  setTimeout(sendNext, 1500)
})

function sendNext() {
  if (i >= QUERIES.length) {
    console.log('ALL QUERIES SENT -- waiting 4s for trailing replies then exiting')
    setTimeout(() => {
      console.log(`DONE -- got ${replies.length}/${QUERIES.length} replies`)
      asker.quit()
      process.exit(replies.length >= QUERIES.length ? 0 : 1)
    }, 4000)
    return
  }
  const q = QUERIES[i]
  console.log(`SENDING [${i + 1}/${QUERIES.length}]: ${q}`)
  asker.chat(q)
  i++
  setTimeout(sendNext, 6000) // actions (esp. goto/smelt) take real time
}

asker.on('chat', (username, message) => {
  if (username === asker.username) return
  if (username !== 'stage2_probe') return
  console.log(`REPLY FROM [${username}]: ${message}`)
  replies.push(message)
})

setTimeout(() => {
  console.error(`TIMEOUT: only got ${replies.length}/${QUERIES.length} replies within 90s`)
  process.exit(1)
}, 90000)
