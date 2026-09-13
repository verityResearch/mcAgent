# Play-testing the mcAgent mod

This is the operator's own play-test — the one thing that gates every later step toward
publishing (see the embodied-agent plan in `docs/plans/` for the full sequence). Nothing here has been reviewed by a human player yet. If something in this document is
wrong, that's exactly what this is for.

Two things get tested, and they need different setups:

- **The mod itself** (`/mcagent ask`, `/mcagent builddb`, `/mcagent state`, `/mcagent action ...`)
  — a real Minecraft 26.2 client connecting to a real server running the mod. This is fully
  testable today.
- **`/mcagent ask`'s answer quality specifically** needs a running `knowledge_server.py` reachable
  at `127.0.0.1:8420` from the *server's* machine (this is hardcoded in
  `KnowledgeServerClient.DEFAULT_BASE_URL` — not yet configurable). Everything else
  (`builddb`/`state`/`action`) works without it.

## Fastest path: play against a shared test server

If you have access to a Fabric server already set up for this mod plus a running (or easily-started)
`knowledge_server.py` with the trained adapter, that is the quickest way to a real play-test —
the same kind of setup the project's own live-verification work used, just with a real client
instead of a headless bot.

1. **Get a real Minecraft Java Edition client on version 26.2.** (If 26.2 isn't your currently
   installed version, the vanilla launcher lets you select it under installations — it needs to
   match the server exactly.)
2. **Get the server's address and port**, and confirm the mod-enabled Fabric server + (if you
   want to test `/mcagent ask`) `knowledge_server.py` are both running there. If they're not
   running, they need to be started first — the mod requires no extra client-side install; Fabric
   loader 0.19.3 handles that automatically once you add the mod jar to your client's `mods/`
   folder (see below), matching the server's `gradle.properties`.
3. **Install the mod on your client**: download `mcagent-0.1.0.jar` (from
   `fact_pipeline/mc_mod/build/libs/`, built via `./gradlew build` — a real JDK 25 is required to
   build it, see the main `README.md`) into your own Fabric client's `mods/` folder, alongside
   Fabric API 0.157.0+26.2 (same version the server uses). You do **not** need the mod installed
   client-side for slash commands to work if you're only issuing them as chat commands to a
   vanilla-looking server — but installing it removes any doubt about client/server version
   matching and is the standard way to play a modded server.
4. **Join the server** with your real client, as yourself.

## Things to try

Go through these roughly in order. Each line: the command, what it *should* do, and why it's
worth trying.

1. `/mcagent ask What do I need to craft a torch?`
   Should answer correctly and immediately (a real, verified fact from the oracle, not a guess).
2. `/mcagent ask What does a carrot on a stick do?`
   Same idea, a different fact family (item-use, not recipe) — confirms the oracle isn't only
   answering recipe questions.
3. **Try to break it — this is the most important one.** Ask about something that genuinely
   isn't in the data: a modded item that doesn't exist (`/mcagent ask What do I need to craft a
   create:cogwheel?`), or a nonsense item (`/mcagent ask What does the flibbertigibbet do?`).
   **It should decline honestly** ("I don't have any data on X") rather than invent an answer.
   This — declining instead of fabricating — is the actual headline claim of the whole project.
   If it ever makes something up here, that's the single most important thing to report back.
4. `/mcagent builddb`
   Should report a summary (item/tag/recipe/loot counts) and write a JSON file to the server's
   game directory. This walks the *live* running registries — try it again after changing a
   datapack or joining a different world/modpack if you have one handy, and confirm the numbers
   change accordingly (this is the "knows your actual modpack" feature — worth specifically
   testing if you have any datapack or resource changes active).
5. `/mcagent state`
   Should report your own inventory, position, health/hunger, nearby blocks/entities, equipped
   items, biome, and time of day as a JSON blob in chat. **This command has never been tested by
   a real connected player before** — confirm every field looks right against what you can see
   in-game (your real inventory contents, your real position, the real time of day).
6. `/mcagent action goto <x> <y> <z>` (pick coordinates near you)
   Should walk you... no — should move the *player entity* toward those coordinates using
   real pathfinding. This is an admin/debug utility, not "the agent" (the actual planning agent
   is the separate Mineflayer bot, not part of this mod) — it exists so a mod-side developer can
   sanity-check the mod's own action primitives without a bot attached.
7. `/mcagent action mine <x> <y> <z>` / `/mcagent action equip <item>` / `/mcagent action attack
   <targetType>` — same idea, quick sanity checks that each primitive does what its name says.
   Not the headline feature; low priority if time is short.

## Things worth noting either way

- **Any crash, hang, or unexpected server-console error** — copy the console output.
- **What happens if you run `/mcagent builddb` on a large or heavily-modded world** — timing,
  memory, anything that looks off. Nobody's tried this at scale yet.
- **What happens if port 8421 (the HTTP oracle endpoint) is already taken** on the machine running
  the server — the mod should log a clean failure rather than crash the server, but this hasn't
  been tested either.
- **Screenshots**, if you're willing — a couple of `/mcagent ask` and `/mcagent state` in action
  are exactly what a public listing would need, and this is the natural moment to grab them.

## After this

Whatever you find — works well, needs fixes, or you don't like the direction — is the real next
input. Per the read: if it doesn't hold up, the right move is fixing it, not publishing with
a caveat. Nothing about a public listing happens until after this.
