// Simulates a real player: connects, asks bot.js a question via chat, waits
// for its reply, then disconnects. End-to-end verification of the full
// stage-2 loop (player -> bot -> knowledge server -> oracle -> back to chat).
const mineflayer = require('mineflayer')

const asker = mineflayer.createBot({
  host: '127.0.0.1', port: 25566, username: 'human_probe', version: '1.21.11',
})

asker.on('spawn', () => {
  console.log('ASKER SPAWNED -- sending question')
  setTimeout(() => asker.chat('!ask What do I need to craft a torch?'), 1000)
})

asker.on('chat', (username, message) => {
  if (username === asker.username) return
  console.log(`REPLY FROM [${username}]: ${message}`)
  if (username === 'stage2_probe') {
    console.log('END-TO-END OK -- got a reply from the answering bot')
    asker.quit()
    process.exit(0)
  }
})

setTimeout(() => {
  console.error('TIMEOUT: no reply within 25s')
  process.exit(1)
}, 25000)
