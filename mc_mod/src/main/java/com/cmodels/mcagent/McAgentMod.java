package com.cmodels.mcagent;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import com.mojang.brigadier.arguments.DoubleArgumentType;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.mojang.brigadier.exceptions.CommandSyntaxException;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.loader.api.FabricLoader;

import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerPlayer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Stage 6 (distribution), Path B from the shippable-agent plan: the same
 * tool-augmented, oracle-backed answer engine already built and verified as
 * a Mineflayer bot (fact_pipeline/mc_bot/), reimplemented as a native
 * Fabric mod so a player never needs Node.js/mineflayer to get verified,
 * non-fabricated answers in-game.
 *
 * This class wires ONLY the knowledge tool (plan section 3.1, stage 1/2's
 * scope) -- "/mcagent ask <question>" relays to the SAME knowledge_server.py
 * HTTP service the Mineflayer bot already calls (POST /ask), so this mod
 * and the bot share one backend and one set of verified numbers; nothing
 * about the oracle/model itself is reimplemented here.
 *
 * State tools (stage 3) are fully ported -- see StateTools's header.
 *
 * ARCHITECTURE, per the design-review reply: "THE MOD IS THE ORACLE. THE BOT IS THE AGENT." This mod's
 * differentiated value is the live-registry oracle (OracleDbBuilder), now
 * ALSO served over HTTP (OracleHttpServer, "/mcagent oracle over
 * http://127.0.0.1:8421/builddb") so mc_bot's real mineflayer agent
 * (real pathing/reach/tool-durability/mob-aggro, already built in stage
 * 4) can consume modpack-aware facts without either side reimplementing
 * the other's strength. "/mcagent action goto/mine/equip/attack"
 * (ActionTools.java, built under the earlier option (c) reading before
 * the reply arrived) is KEPT as a legitimate admin/debug utility, per
 * the reviewer's own words -- just not presented as "the agent". See
 * OracleHttpServer's header for the full architecture writeup.
 *
 * Ship the generator, not the database (plan section 5):
 * KnowledgeServerClient below only talks to a locally-run
 * knowledge_server.py; this mod does not embed or redistribute
 * minecraft.db.
 */
public class McAgentMod implements ModInitializer {
	public static final String MOD_ID = "mcagent";

	// This logger is used to write text to the console and the log file.
	// It is considered best practice to use your mod id as the logger's name.
	// That way, it's clear which mod wrote info, warnings, and errors.
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	private static OracleHttpServer oracleHttpServer;

	@Override
	public void onInitialize() {
		// This code runs as soon as Minecraft is in a mod-load-ready state.
		// However, some things (like resources) may still be uninitialized.
		// Proceed with mild caution.
		LOGGER.info("mcAgent initializing -- knowledge tool wired to {}", KnowledgeServerClient.DEFAULT_BASE_URL);

		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			try {
				oracleHttpServer = OracleHttpServer.start(server, LOGGER);
			} catch (IOException e) {
				// Fails open (in-game "/mcagent builddb" still works) rather
				// than crashing server startup over a convenience HTTP
				// bridge -- e.g. the port is already bound by another
				// process. Logged loudly so it isn't silently missed.
				LOGGER.error("mcAgent oracle HTTP server failed to start", e);
			}
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			if (oracleHttpServer != null) {
				oracleHttpServer.stop();
				oracleHttpServer = null;
			}
		});

		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
			dispatcher.register(
				Commands.literal("mcagent")
					.then(Commands.literal("ask")
						.then(Commands.argument("question", StringArgumentType.greedyString())
							.executes(McAgentMod::runAsk)))
					.then(Commands.literal("builddb")
						.executes(McAgentMod::runBuildDb))
					.then(Commands.literal("state")
						.executes(McAgentMod::runState)
						.then(Commands.argument("blocksRadius", IntegerArgumentType.integer(1))
							.then(Commands.argument("entitiesRadius", IntegerArgumentType.integer(1))
								.executes(McAgentMod::runStateWithRadii))))
					.then(Commands.literal("action")
						// Deny-by-default: only goto/mine/equip are reachable at
						// all -- same discipline as mc_bot's action_dispatcher.js
						// (TOOL_TIER refuses any unlisted name outright), not an
						// allowlist an added tool could accidentally bypass.
						.then(Commands.literal("goto")
							.then(Commands.argument("x", DoubleArgumentType.doubleArg())
								.then(Commands.argument("y", DoubleArgumentType.doubleArg())
									.then(Commands.argument("z", DoubleArgumentType.doubleArg())
										.executes(McAgentMod::runActionGoto)))))
						.then(Commands.literal("mine")
							.then(Commands.argument("x", IntegerArgumentType.integer())
								.then(Commands.argument("y", IntegerArgumentType.integer())
									.then(Commands.argument("z", IntegerArgumentType.integer())
										.executes(McAgentMod::runActionMine)))))
						.then(Commands.literal("equip")
							.then(Commands.argument("item", StringArgumentType.string())
								.executes(McAgentMod::runActionEquip)))
						.then(Commands.literal("attack")
							.then(Commands.argument("targetType", StringArgumentType.string())
								.executes(McAgentMod::runActionAttack)
								.then(Commands.argument("radius", DoubleArgumentType.doubleArg(1))
									.executes(McAgentMod::runActionAttackWithRadius)))))));
	}

	private static int runAsk(CommandContext<CommandSourceStack> context) {
		String question = StringArgumentType.getString(context, "question");
		CommandSourceStack source = context.getSource();

		// The HTTP call is real network I/O -- do it off the server thread so
		// a slow or unreachable knowledge server can't stall the game tick
		// loop, matching how the Mineflayer bot already treats this as an
		// async operation.
		KnowledgeServerClient.askAsync(question).whenComplete((answer, throwable) -> {
			if (throwable != null) {
				LOGGER.warn("mcAgent ask failed", throwable);
				source.sendSystemMessage(Component.literal("(couldn't reach the knowledge server: " + throwable.getMessage() + ")"));
				return;
			}
			source.sendSystemMessage(Component.literal(answer));
		});

		return 1;
	}

	/**
	 * "/mcagent builddb" -- plan section 5's "ship the generator, not the
	 * database": walk this running game's OWN registries (OracleDbBuilder)
	 * instead of shipping Mojang's minecraft.db. Items, item tags, and a
	 * partial recipe table so far; see OracleDbBuilder's header for what's
	 * not covered yet.
	 */
	private static int runBuildDb(CommandContext<CommandSourceStack> context) {
		CommandSourceStack source = context.getSource();
		Path outputPath = FabricLoader.getInstance().getGameDir().resolve("mcagent_oracle_dump.json");

		try {
			String summary = OracleDbBuilder.buildAndWrite(outputPath, source.getServer().getRecipeManager(), source.getServer().registryAccess(), source.getServer().reloadableRegistries().lookup());
			LOGGER.info("mcagent builddb: {}", summary);
			source.sendSystemMessage(Component.literal(summary));
		} catch (IOException | RuntimeException e) {
			// Catch RuntimeException too, not just IOException -- Brigadier's
			// own command-dispatch failure handler only logs a generic
			// "unexpected error" line to the server console with no stack
			// trace, which made a real runtime failure here undiagnosable
			// from server logs alone until this was widened. Log the full
			// trace ourselves instead of letting it vanish.
			LOGGER.error("mcagent builddb failed", e);
			source.sendSystemMessage(Component.literal("(builddb failed: " + e + ")"));
			return 0;
		}

		return 1;
	}

	/**
	 * "/mcagent state" -- stage 3 (state tools) port: reports the calling
	 * player's inventory, position, and health/hunger as JSON, mirroring
	 * fact_pipeline/mc_bot/state_tools.js's inventory()/position()/
	 * health_and_hunger() field shapes exactly. See StateTools's header for
	 * what's not ported yet (time_of_day, nearby_blocks, nearby_entities,
	 * equipped, biome).
	 */
	private static int runState(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		return runState(context, StateTools.nearbyBlocks(player), StateTools.nearbyEntities(player));
	}

	/**
	 * "/mcagent state &lt;blocksRadius&gt; &lt;entitiesRadius&gt;" -- caller
	 * -supplied radius variant, matching state_tools.js's own
	 * nearby_blocks(radius)/nearby_entities(radius) signatures. Both
	 * arguments are required together (Brigadier doesn't cleanly support
	 * "second optional without the first"); the no-arg "/mcagent state"
	 * above keeps the JS defaults (8, 16). Clamped inside StateTools, not
	 * here -- see StateTools's header.
	 */
	private static int runStateWithRadii(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		int blocksRadius = IntegerArgumentType.getInteger(context, "blocksRadius");
		int entitiesRadius = IntegerArgumentType.getInteger(context, "entitiesRadius");
		return runState(context, StateTools.nearbyBlocks(player, blocksRadius), StateTools.nearbyEntities(player, entitiesRadius));
	}

	/**
	 * "/mcagent state" -- stage 3 (state tools) port: reports the calling
	 * player's inventory, position, and health/hunger as JSON, mirroring
	 * fact_pipeline/mc_bot/state_tools.js's inventory()/position()/
	 * health_and_hunger() field shapes exactly. See StateTools's header for
	 * what's not ported yet (time_of_day).
	 */
	private static int runState(CommandContext<CommandSourceStack> context, JsonArray nearbyBlocks, JsonArray nearbyEntities) throws CommandSyntaxException {
		CommandSourceStack source = context.getSource();
		ServerPlayer player = source.getPlayerOrException();

		JsonObject root = new JsonObject();
		root.add("inventory", StateTools.inventory(player));
		root.add("position", StateTools.position(player));
		root.add("health_and_hunger", StateTools.healthAndHunger(player));
		root.add("nearby_blocks", nearbyBlocks);
		root.add("nearby_entities", nearbyEntities);
		root.add("equipped", StateTools.equipped(player));
		root.add("biome", StateTools.biome(player));
		root.add("time_of_day", StateTools.timeOfDay(player));

		source.sendSystemMessage(Component.literal(root.toString()));
		return 1;
	}

	/**
	 * "/mcagent action goto &lt;x&gt; &lt;y&gt; &lt;z&gt;" -- option (c)'s
	 * teleport-instead-of-pathfind. See ActionTools's header for the full
	 * "why option (c)" writeup.
	 */
	private static int runActionGoto(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		double x = DoubleArgumentType.getDouble(context, "x");
		double y = DoubleArgumentType.getDouble(context, "y");
		double z = DoubleArgumentType.getDouble(context, "z");
		context.getSource().sendSystemMessage(Component.literal(ActionTools.goto_(player, x, y, z).toString()));
		return 1;
	}

	/**
	 * "/mcagent action mine &lt;x&gt; &lt;y&gt; &lt;z&gt;" -- option (c)'s
	 * direct-block-removal-instead-of-tool-gated-breaking, using real
	 * loot-table drops (Level#destroyBlock's own dropAll=true path).
	 */
	private static int runActionMine(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		int x = IntegerArgumentType.getInteger(context, "x");
		int y = IntegerArgumentType.getInteger(context, "y");
		int z = IntegerArgumentType.getInteger(context, "z");
		context.getSource().sendSystemMessage(Component.literal(ActionTools.mine(player, x, y, z).toString()));
		return 1;
	}

	/**
	 * "/mcagent action equip &lt;item&gt;" -- hotbar reselection only; does
	 * NOT move items between inventory sections (a documented scope limit,
	 * see ActionTools's header).
	 */
	private static int runActionEquip(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		String item = StringArgumentType.getString(context, "item");
		context.getSource().sendSystemMessage(Component.literal(ActionTools.equip(player, item).toString()));
		return 1;
	}

	// Matches StateTools.NEARBY_ENTITIES_RADIUS's default -- attack()'s
	// search box is conceptually the same "how far can I see/reach a
	// target" radius as nearby_entities' own default.
	private static final double DEFAULT_ATTACK_RADIUS = 16.0;

	/**
	 * "/mcagent action attack &lt;targetType&gt;" -- option (c)'s
	 * direct-damage-instead-of-melee-swing, searching within
	 * DEFAULT_ATTACK_RADIUS. See ActionTools's header for the full
	 * construction-level target-filter writeup.
	 */
	private static int runActionAttack(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		String targetType = StringArgumentType.getString(context, "targetType");
		context.getSource().sendSystemMessage(Component.literal(ActionTools.attack(player, targetType, DEFAULT_ATTACK_RADIUS).toString()));
		return 1;
	}

	/**
	 * "/mcagent action attack &lt;targetType&gt; &lt;radius&gt;" -- caller
	 * -supplied search radius variant, clamped to [1,64] inside
	 * ActionTools.attack, same clamp discipline as StateTools.nearbyEntities.
	 */
	private static int runActionAttackWithRadius(CommandContext<CommandSourceStack> context) throws CommandSyntaxException {
		ServerPlayer player = context.getSource().getPlayerOrException();
		String targetType = StringArgumentType.getString(context, "targetType");
		double radius = DoubleArgumentType.getDouble(context, "radius");
		context.getSource().sendSystemMessage(Component.literal(ActionTools.attack(player, targetType, radius).toString()));
		return 1;
	}

	public static Identifier id(String path) {
		return Identifier.fromNamespaceAndPath(MOD_ID, path);
	}
}
