# Play-testing the mcAgent mod

A guide for trying the mod with a real Minecraft client. It has been exercised by headless bots but has had very little testing by human players, so reports of anything that looks wrong are genuinely useful.

## What you need

- **Minecraft Java Edition 26.2.** The launcher can select 26.2 under Installations, and it must match the server exactly.
- **A Fabric 26.2 server running the mod.** Loader 0.19.3 and Fabric API 0.157.0+26.2, as pinned in `gradle.properties`.
- **For `/mcagent ask` only:** a running `knowledge_server.py` with a trained adapter, reachable at `127.0.0.1:8420` *from the server's machine*. That address is hardcoded in `KnowledgeServerClient.DEFAULT_BASE_URL`. `builddb`, `state`, and `action` work without it.

## Setup

1. **Build the mod:** `cd mc_mod && ./gradlew build` (JDK 25). The jar lands in `mc_mod/build/libs/mcagent-0.1.0.jar`.
2. **Install it on the server:** copy the jar and Fabric API into the server's `mods/` folder.
3. **Optionally install it on your client too.** Slash commands work against the server without it, but a matching client install rules out version-mismatch issues.
4. **Start the knowledge server** if you want to test `ask` (see the top-level README, step 5).
5. **Join the server.**

## Things to try

Roughly in order of importance.

1. **`/mcagent ask What do I need to craft a torch?`**
   Should answer correctly from the oracle.

2. **`/mcagent ask What does a carrot on a stick do?`**
   A different fact family (item use, not recipes), confirming the oracle isn't limited to recipes.

3. **Try to break it. This is the most important test.**
   Ask about something that isn't in the data: a modded item that isn't installed (`/mcagent ask What do I need to craft a create:cogwheel?`) or a nonsense item (`/mcagent ask What does the flibbertigibbet do?`).
   It should decline ("I don't have any data on X") rather than invent an answer. Declining instead of fabricating is the project's central claim, so a made-up answer here is the most valuable thing you can report.

4. **`/mcagent builddb`**
   Should print a summary (item, tag, recipe, and loot counts) and write a JSON file to the server's game directory. It reads the *live* registries: change a datapack or switch modpacks, run it again, and the counts should change.

5. **`/mcagent state`**
   Should report your inventory, position, health and hunger, nearby blocks and entities, equipped items, biome, and time of day as JSON in chat. Check each field against what you see in-game.

6. **`/mcagent action goto <x> <y> <z>`**
   Teleports you to the coordinates. This is an admin/debug primitive, not the agent. Real movement and planning live in the separate Mineflayer bot (`mc_bot/`).

7. **`/mcagent action mine <x> <y> <z>`, `equip <item>`, `attack <targetType>`**
   Quick checks that each primitive does what its name says. Low priority.

## Worth reporting

- **Crashes, hangs, or server-console errors.** Include the console output.
- **`/mcagent builddb` on a large or heavily modded world:** timing, memory, anything odd. It hasn't been tried at scale.
- **Port 8421 already in use** on the server machine. The mod should log a clean failure rather than crash, but this is untested.
- **Screenshots** of `ask` and `state` in action are welcome.

Open an issue with what you found. See [CONTRIBUTING.md](../CONTRIBUTING.md).
