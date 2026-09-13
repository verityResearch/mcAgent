# mcAgent (Fabric mod)

A Fabric mod for Minecraft 26.2 that serves as the live-game half of the mcAgent project (see the [top-level README](../README.md) for context).

**The Mod is the Oracle, the Bot is the Agent.**
Rather than reimplementing game physics (pathing, tool durability, mob aggro) in the mod, or guessing valid game states externally, mcAgent combines the strengths of both:
- This mod serves as an oracle, directly accessing the real running modpack's registries over localhost HTTP.
- The external Mineflayer bot (`mc_bot/`) queries this oracle to plan and execute actions.

## Features

- **`/mcagent ask <question>`**
  Relays questions to the `knowledge_server.py` HTTP service. The mod and the bot share one backend and one set of verified numbers.

- **`GET http://127.0.0.1:8421/builddb`**
  An HTTP endpoint that exposes the live server's registry and recipe data as JSON. This is exactly what the offline oracle uses, but built from the actual running server (including modpacks). It requires no connected client and does not read per-tick world state.

- **`/mcagent state [blocksRadius] [entitiesRadius]`**
  Reports the calling player's state as JSON, including:
  - Inventory and equipped items
  - Position and current biome
  - Health and hunger
  - Nearby blocks and entities (customizable radii)
  - Time of day (including exact tick times and day/night cycle calculation)

- **`/mcagent builddb`**
  An in-game command version of the HTTP endpoint that walks the running game's registries and recipe book, writing the result to a JSON file on the server.

- **`/mcagent action goto/mine/equip/attack`** *(Admin/Debug Utility)*
  A limited set of admin commands used to test basic interactions. **Note:** This is *not* the embodied agent layer. Real complex actions and pathfinding are handled by the external `mc_bot`.
  - `goto(x,y,z)`: Teleports the player.
  - `mine(x,y,z)`: Drops the block via normal loot tables.
  - `equip(item)`: Selects a hotbar slot.
  - `attack(targetType)`: Directly damages a matching hostile mob.

## What's Not Included

- Full action-tool layers (e.g., crafting, placing blocks, smelting) — these are intentionally delegated to `mc_bot`.
- Support for Minecraft versions other than 26.2, or loaders other than Fabric (NeoForge support is currently blocked upstream by Forgified Fabric API).

## Setup

This is a standard Fabric mod project.

1. Ensure you have a JDK matching the version pinned in `fabric.mod.json`/`build.gradle` (currently JDK 25).
2. Run `./gradlew build` to produce `build/libs/mcagent-<version>.jar`.
3. Drop the JAR into your server's `mods/` directory.

## License

Apache-2.0 — see [LICENSE](../LICENSE).

*This project was scaffolded from [FabricMC/fabric-example-mod](https://github.com/FabricMC/fabric-example-mod).*
