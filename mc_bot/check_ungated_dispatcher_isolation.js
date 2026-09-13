// Regression guard for the gated/ungated action_dispatcher.js boundary --
// added per the stage-4/5 review reply: "The residual risk isn't a typo today, it's DRIFT --
// someone refactors in six months and the paths converge. So pin it with
// a regression test rather than a bigger boundary: a test asserting
// bot.js (and any live-serving entry point) never constructs the ungated
// dispatcher -- grep-level is fine, it just has to FAIL LOUDLY if that
// changes." Same discipline as this project's own protected-path checks
// (scripts/check_docs_integrity.py etc.), applied to a code path instead
// of a file.
//
// makeDispatcher({mode: 'ungated'}) skips the confirm/nonce gate entirely
// (see action_dispatcher.js's own header) -- it must ONLY ever be
// constructed by an OFFLINE trace-generation script running against an
// isolated server, never by anything that connects as a live,
// player-facing bot. This check does not try to be clever about what
// counts as "live-serving" -- it flips the risk the safe way: every .js
// file in this directory is FORBIDDEN from constructing the ungated
// dispatcher UNLESS explicitly allowlisted below, so a new live-serving
// file added later fails this check by default instead of silently
// inheriting an exemption.

const fs = require('fs')
const path = require('path')

// Only a real, offline, no-live-connection trace generator belongs here.
const ALLOWED_TO_USE_UNGATED_MODE = new Set(['gen_action_traces.js'])

// This checker's own file matches the pattern trivially (it's quoting the
// pattern to check for it), and action_dispatcher.js's header comment
// legitimately documents the construction syntax as API description, not
// a live construction site -- neither is a "consumer" this check is
// scoped to police. This is a real, checked distinction: EXCLUDED files
// are still readable by anyone auditing this check, just not scanned.
const EXCLUDED_FROM_CHECK = new Set(['check_ungated_dispatcher_isolation.js', 'action_dispatcher.js'])

const UNGATED_PATTERN = /makeDispatcher\s*\(\s*\{\s*mode\s*:\s*['"]ungated['"]/

function main() {
  const dir = __dirname
  const jsFiles = fs.readdirSync(dir).filter((f) => f.endsWith('.js'))

  const violations = []
  for (const file of jsFiles) {
    if (ALLOWED_TO_USE_UNGATED_MODE.has(file) || EXCLUDED_FROM_CHECK.has(file)) continue
    const contents = fs.readFileSync(path.join(dir, file), 'utf8')
    if (UNGATED_PATTERN.test(contents)) {
      violations.push(file)
    }
  }

  if (violations.length > 0) {
    console.error('UNGATED DISPATCHER ISOLATION VIOLATION')
    console.error('The following files construct the ungated action dispatcher')
    console.error("(makeDispatcher({mode: 'ungated'})) but are NOT on the allowlist")
    console.error(`(${[...ALLOWED_TO_USE_UNGATED_MODE].join(', ')}):`)
    for (const file of violations) {
      console.error(`  - ${file}`)
    }
    console.error('')
    console.error('The ungated dispatcher skips the confirm/nonce gate entirely and must')
    console.error('only ever be used by an offline, no-live-connection trace generator.')
    console.error('If this file is genuinely a new offline generator, add it to')
    console.error('ALLOWED_TO_USE_UNGATED_MODE in this script deliberately. If it is a')
    console.error('live-serving bot entry point, this is exactly the drift the review warned about')
    console.error('about -- fix the code, not this check.')
    process.exitCode = 1
    return
  }

  console.log(`OK -- ungated dispatcher isolation holds across ${jsFiles.length} files.`)
}

main()
