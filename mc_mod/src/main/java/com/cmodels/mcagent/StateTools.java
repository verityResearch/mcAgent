package com.cmodels.mcagent;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.clock.ClockManager;
import net.minecraft.world.clock.ClockTimeMarker;
import net.minecraft.world.clock.WorldClock;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.timeline.Timeline;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Stage 3 (state tools) port to Fabric -- mirrors fact_pipeline/mc_bot/
 * state_tools.js's field shapes exactly for inventory(), position(),
 * health_and_hunger(), nearby_blocks(), and nearby_entities() (name/
 * displayName/count/slot; x/y/z; health/food/saturation; name+position+
 * distance; type+name+username+position+distance), read-only wrappers over
 * Fabric's own tracked server-side state, same "no writes, nothing
 * invented" principle as the JS version's header comment.
 *
 * equipped() and biome() are now covered too (7 of 8 JS tools). equipped
 * mirrors the JS shape exactly: {mainhand, helmet, chestplate, leggings,
 * boots}, each null or {name, count}. biome does NOT mirror the JS shape:
 * JS returns {name, id} where id is a small mineflayer/minecraft-data
 * integer; this MC version's biome registry keys are strings like
 * "minecraft:plains" with no small-integer id exposed at this API layer,
 * so the Fabric version returns {name} only -- a real, deliberate schema
 * difference, not an oversight.
 *
 * time_of_day() IS NOW IMPLEMENTED, via the real Timeline system found by
 * disassembling vanilla's /time command (TimeCommand.class) rather than
 * guessing. The default WorldClock's raw getTotalTicks() (an earlier
 * finding) is NOT a wrapped day-length value; the actual wrapped
 * "position within the day/night cycle" comes from a Timeline, not a
 * WorldClock directly. This implementation: player.level()
 * .dimensionTypeRegistration().value() -> DimensionType; #defaultClock()
 * -> the day-night WorldClock; scan DimensionType#timelines() (a
 * HolderSet<Timeline>) for the day-night one (see the REAL BUG paragraph
 * below -- "matches the default clock" alone is NOT enough to identify
 * it); then Timeline#getCurrentTicks(ClockManager)
 * (MinecraftServer#clockManager(), which implements ClockManager) for the
 * wrapped tick position, and Timeline#periodTicks() for the day length, to
 * compute timeOfDay/day/isDay matching state_tools.js's time_of_day()
 * shape -- now all 3 fields, not partial.
 *
 * LIVE-VERIFIED, not just compiled: getCurrentTicks() vs. getTotalTicks()
 * was an inference from the API shape (Timeline exposes both as distinct
 * methods, so one had to be cumulative and one had to be
 * period-relative), not bytecode-proof the way TimeCommand's own methods
 * were. Confirmed correct on a dedicated Fabric test server:
 * "/mcagent builddb" (extended with a temporary diagnostic, since removed
 * once verified) reported timeOfDay=4878 at one moment; running vanilla's
 * own "/time query day" a few seconds later reported "Timeline
 * minecraft:day is at 5057 tick(s)" -- the same clock, same trajectory,
 * confirming getCurrentTicks() really is what "/time query <timeline>"
 * itself reports. (Also incidentally found the real default timeline id
 * is "minecraft:day", not "minecraft:daytime" -- "/time query daytime"
 * failed with "Can't find element" -- though this code discovers the
 * timeline dynamically, not via a hardcoded id.)
 *
 * REAL BUG FOUND (during isDay's implementation) AND FIXED, affecting
 * timeOfDay/day too, not just isDay: the overworld's DimensionType#timelines()
 * actually contains FOUR timelines that all share the "minecraft:overworld"
 * clock -- minecraft:day, minecraft:moon, minecraft:early_game, and
 * minecraft:villager_schedule (confirmed via the server data generator
 * output, data/minecraft/tags/timeline/in_overworld.json + each timeline's
 * own JSON) -- and only minecraft:day actually carries time_markers; the
 * other three have none. The original "pick the first clock match" logic
 * was genuinely ambiguous and, empirically, picked a marker-less timeline
 * (confirmed live: a temporary diagnostic dumping registerTimeMarkers()'s
 * output for the selected timeline came back as an empty map). Because all
 * four timelines report the SAME getCurrentTicks() early in a fresh
 * world (small elapsed-tick values haven't wrapped by any of their
 * periods yet), the earlier live cross-check above (4878 vs 5057) didn't
 * actually prove "minecraft:day" was selected -- it only proved
 * getCurrentTicks() tracks real elapsed ticks, true for ANY candidate.
 * FIXED by disambiguating generically: among clock-matching timelines,
 * prefer the one whose own registerTimeMarkers() output contains both a
 * "day" and a "night" path -- this is what "the day-night timeline"
 * actually means, not an assumption about ids or HolderSet ordering; falls
 * back to the first clock match (old behavior) only if no candidate has
 * day/night markers at all, so timeOfDay/day still report something on a
 * dimension with no such cycle.
 *
 * isDay is NOW ALSO implemented, grounded in real registered marker data,
 * not the old vanilla ticks-0-12000 convention. Timeline#registerTimeMarkers
 * (BiConsumer<ResourceKey<ClockTimeMarker>, ClockTimeMarker>) is the only
 * public way to read a Timeline's markers -- its backing timeMarkers map
 * and the Timeline$TimeMarkerInfo record type are both package-private, so
 * this had to go through the callback, not a direct getter. Grounded in
 * REAL GROUND-TRUTH DATA, not a guess: the server data generator output
 * (data/minecraft/timeline/day.json) shows minecraft:day at ticks=1000,
 * minecraft:night at ticks=13000, period_ticks=24000 -- i.e. this
 * dimension's real day/night boundary is genuinely different from
 * vanilla's old 0/12000 split. isDay is computed generically (range-check
 * timeOfDay against the "day"/"night" marker ticks() with period
 * wraparound), not by hardcoding these specific numbers -- so it stays
 * correct even if a dimension's actual marker ticks differ from the
 * overworld's. LIVE-VERIFIED both ways on the test server, after the
 * disambiguation fix above: at timeOfDay=8098 (inside [1000,13000))
 * reported isDay=true; after "/time set 15000", at timeOfDay=15034
 * (inside [13000,24000)) reported isDay=false -- both cross-checked
 * against vanilla's own "/time query day" reporting a consistent
 * trajectory at the same moments. Returns null if either marker is absent
 * (e.g. a dimension with no day/night cycle) rather than guessing.
 *
 * nearby_blocks/nearby_entities now take an optional caller-supplied
 * radius (wired via Brigadier in McAgentMod -- "/mcagent state" and
 * "/mcagent state <blocksRadius> <entitiesRadius>"), matching
 * state_tools.js's own radius(default) signature. nearby_blocks clamps
 * to [1,16] exactly like the JS version's own comment ("cap: a full-cube
 * scan is O(r^3)"). nearby_entities' JS version has NO clamp (just a
 * post-hoc distance filter), but this port adds one anyway ([1,64]) as a
 * deliberate, small divergence -- a caller-supplied radius reaching a
 * live server is a different trust boundary than a hardcoded value in a
 * bot script, and Level#getEntities' AABB query cost still scales with
 * the box even though it isn't O(r^3) the way a block cube scan is.
 *
 * Every method used (CommandSourceStack#getPlayerOrException,
 * Entity#position/level/getType/getName, Vec3's public x/y/z fields,
 * Vec3#distanceTo, Vec3#atLowerCornerOf, LivingEntity#getHealth,
 * Player#getFoodData, FoodData#getFoodLevel/getSaturationLevel,
 * Player#getInventory, Container#getContainerSize/getItem,
 * ItemStack#isEmpty/getItem/getCount/getHoverName, Registry#getKey,
 * Level#getBlockState/getEntities/getBiomeManager, BiomeManager#getBiome,
 * BlockStateBase#isAir/getBlock, AABB#inflate,
 * LivingEntity#getMainHandItem/getItemBySlot, Holder#getRegisteredName)
 * was confirmed via javap against this project's own Loom-mapped 26.2 jar
 * before being written, same discipline as OracleDbBuilder.
 *
 * NOT YET VERIFIED LIVE: written and (if the build succeeded) compiles
 * against the real API surface, but running it against a live player to
 * confirm the JSON is well-formed and matches real inventory contents
 * requires a running Minecraft client, which this development environment
 * cannot provide. Flagged honestly rather than claimed proven.
 */
final class StateTools {
	private static final int NEARBY_BLOCKS_RADIUS = 8;
	private static final double NEARBY_ENTITIES_RADIUS = 16;

	private StateTools() {
	}

	static JsonObject inventory(ServerPlayer player) {
		Inventory inv = player.getInventory();
		JsonArray items = new JsonArray();
		for (int slot = 0; slot < inv.getContainerSize(); slot++) {
			ItemStack stack = inv.getItem(slot);
			if (stack.isEmpty()) {
				continue;
			}
			JsonObject entry = new JsonObject();
			entry.addProperty("name", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
			entry.addProperty("displayName", stack.getHoverName().getString());
			entry.addProperty("count", stack.getCount());
			entry.addProperty("slot", slot);
			items.add(entry);
		}

		JsonObject result = new JsonObject();
		result.add("items", items);
		return result;
	}

	static JsonObject position(ServerPlayer player) {
		Vec3 p = player.position();
		JsonObject result = new JsonObject();
		result.addProperty("x", p.x);
		result.addProperty("y", p.y);
		result.addProperty("z", p.z);
		return result;
	}

	static JsonObject healthAndHunger(ServerPlayer player) {
		JsonObject result = new JsonObject();
		result.addProperty("health", player.getHealth());
		result.addProperty("food", player.getFoodData().getFoodLevel());
		result.addProperty("saturation", player.getFoodData().getSaturationLevel());
		return result;
	}

	static JsonArray nearbyBlocks(ServerPlayer player) {
		return nearbyBlocks(player, NEARBY_BLOCKS_RADIUS);
	}

	static JsonArray nearbyBlocks(ServerPlayer player, int radius) {
		Level level = player.level();
		Vec3 playerPos = player.position();
		BlockPos origin = player.blockPosition();

		// Same clamp as state_tools.js's nearby_blocks(): "cap: a full-cube
		// scan is O(r^3)".
		int r = Math.max(1, Math.min(radius, 16));

		List<JsonObject> found = new ArrayList<>();
		for (int dx = -r; dx <= r; dx++) {
			for (int dy = -r; dy <= r; dy++) {
				for (int dz = -r; dz <= r; dz++) {
					BlockPos pos = origin.offset(dx, dy, dz);
					BlockState state = level.getBlockState(pos);
					if (state.isAir()) {
						continue;
					}
					double distance = playerPos.distanceTo(Vec3.atLowerCornerOf(pos));
					JsonObject entry = new JsonObject();
					entry.addProperty("name", BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());
					JsonObject position = new JsonObject();
					position.addProperty("x", pos.getX());
					position.addProperty("y", pos.getY());
					position.addProperty("z", pos.getZ());
					entry.add("position", position);
					entry.addProperty("distance", distance);
					found.add(entry);
				}
			}
		}
		found.sort(Comparator.comparingDouble(e -> e.get("distance").getAsDouble()));

		JsonArray result = new JsonArray();
		found.forEach(result::add);
		return result;
	}

	static JsonArray nearbyEntities(ServerPlayer player) {
		return nearbyEntities(player, NEARBY_ENTITIES_RADIUS);
	}

	static JsonArray nearbyEntities(ServerPlayer player, double radius) {
		Level level = player.level();
		Vec3 playerPos = player.position();

		// state_tools.js's nearby_entities() has no clamp; this port adds one
		// deliberately -- see this class's header.
		double r = Math.max(1, Math.min(radius, 64));
		AABB box = new AABB(player.blockPosition()).inflate(r);

		List<JsonObject> found = new ArrayList<>();
		for (Entity entity : level.getEntities(player, box, e -> e != player)) {
			double distance = playerPos.distanceTo(entity.position());
			if (distance > r) {
				continue;
			}
			JsonObject entry = new JsonObject();
			entry.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());
			entry.addProperty("name", entity.getName().getString());
			entry.addProperty("username", entity instanceof ServerPlayer ? entity.getName().getString() : null);
			Vec3 entityPos = entity.position();
			JsonObject position = new JsonObject();
			position.addProperty("x", entityPos.x);
			position.addProperty("y", entityPos.y);
			position.addProperty("z", entityPos.z);
			entry.add("position", position);
			entry.addProperty("distance", distance);
			found.add(entry);
		}
		found.sort(Comparator.comparingDouble(e -> e.get("distance").getAsDouble()));

		JsonArray result = new JsonArray();
		found.forEach(result::add);
		return result;
	}

	static JsonObject equipped(ServerPlayer player) {
		JsonObject result = new JsonObject();
		result.add("mainhand", itemStackOrNull(player.getMainHandItem()));
		result.add("helmet", itemStackOrNull(player.getItemBySlot(EquipmentSlot.HEAD)));
		result.add("chestplate", itemStackOrNull(player.getItemBySlot(EquipmentSlot.CHEST)));
		result.add("leggings", itemStackOrNull(player.getItemBySlot(EquipmentSlot.LEGS)));
		result.add("boots", itemStackOrNull(player.getItemBySlot(EquipmentSlot.FEET)));
		return result;
	}

	static JsonObject itemStackOrNull(ItemStack stack) {
		if (stack.isEmpty()) {
			return null;
		}
		JsonObject entry = new JsonObject();
		entry.addProperty("name", BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
		entry.addProperty("count", stack.getCount());
		return entry;
	}

	static JsonObject biome(ServerPlayer player) {
		Holder<Biome> holder = player.level().getBiomeManager().getBiome(player.blockPosition());
		JsonObject result = new JsonObject();
		result.addProperty("name", holder.getRegisteredName());
		return result;
	}

	/**
	 * All of timeOfDay/day/isDay are implemented. Finds the dimension's
	 * day-night Timeline -- among the timelines sharing
	 * DimensionType#defaultClock(), the one whose own registered markers
	 * include both a "day" and a "night" path, NOT just "the first clock
	 * match" (a real bug: several timelines can share one clock and only
	 * one carries day/night markers -- see the timeOfDay(ServerLevel)
	 * overload's own comment) -- and reads Timeline#getCurrentTicks(ClockManager)
	 * -- CONFIRMED via javap -c bytecode disassembly to be
	 * getTotalTicks(ClockManager) % periodTicks (i.e. genuinely the
	 * wrapped, in-period position, not an inference from API shape) -- for
	 * timeOfDay, and Timeline#periodTicks() to derive day (total elapsed
	 * periods). isDay is computed from the same Timeline's registered
	 * "day"/"night" ClockTimeMarker ticks via
	 * Timeline#registerTimeMarkers(...), grounded in real generated
	 * ground-truth data, not the old vanilla ticks-0-12000 convention --
	 * see the timeOfDay(ServerLevel) overload's own comment for the exact
	 * values found and the live day/night cross-check.
	 */
	static JsonObject timeOfDay(ServerPlayer player) {
		return timeOfDay(player.level());
	}

	/**
	 * Overload taking a ServerLevel directly rather than a player -- time
	 * of day is world-global data, not player-specific, so it doesn't
	 * strictly need a connected player at all. Lets this be diagnosed via
	 * MinecraftServer#overworld() from a console-only command (no live
	 * client needed) instead of only through the player-gated
	 * "/mcagent state".
	 */
	static JsonObject timeOfDay(ServerLevel level) {
		DimensionType dimensionType = level.dimensionTypeRegistration().value();
		Optional<Holder<WorldClock>> defaultClock = dimensionType.defaultClock();
		if (defaultClock.isEmpty()) {
			return null;
		}
		String defaultClockName = defaultClock.get().getRegisteredName();

		// LIVE-FOUND BUG (see this method's own header comment): a dimension
		// can have SEVERAL timelines sharing the same clock -- the overworld
		// really does (day/moon/early_game/villager_schedule all use
		// minecraft:overworld) -- and only one of them actually carries
		// day/night markers. Picking "the first clock match" (the original,
		// wrong approach) is ambiguous and can silently select a marker-less
		// timeline. Disambiguate generically: among clock-matching
		// candidates, prefer the one whose own registered markers include
		// both a "day" and a "night" path -- this is what "the day-night
		// timeline" actually MEANS, not an assumption about ids or ordering.
		Timeline clockMatchFallback = null;
		Timeline dayNightTimeline = null;
		Map<String, Integer> dayNightMarkerTicks = null;
		for (Holder<Timeline> timelineHolder : dimensionType.timelines()) {
			Timeline timeline = timelineHolder.value();
			if (!timeline.clock().getRegisteredName().equals(defaultClockName)) {
				continue;
			}
			if (clockMatchFallback == null) {
				clockMatchFallback = timeline;
			}
			Map<String, Integer> markerTicksByPath = new HashMap<>();
			timeline.registerTimeMarkers((ResourceKey<ClockTimeMarker> key, ClockTimeMarker marker) ->
				markerTicksByPath.put(key.identifier().getPath(), marker.ticks()));
			if (markerTicksByPath.containsKey("day") && markerTicksByPath.containsKey("night")) {
				dayNightTimeline = timeline;
				dayNightMarkerTicks = markerTicksByPath;
				break;
			}
		}
		// No timeline on this clock carries day/night markers at all (e.g. a
		// modded/nether-like dimension) -- fall back to any clock match so
		// timeOfDay/day still report something, but isDay stays null since
		// there's genuinely no day/night boundary to compute it from.
		if (dayNightTimeline == null) {
			dayNightTimeline = clockMatchFallback;
		}
		if (dayNightTimeline == null) {
			return null;
		}

		ClockManager clockManager = level.getServer().clockManager();
		long currentTicks = dayNightTimeline.getCurrentTicks(clockManager);
		Optional<Integer> periodTicks = dayNightTimeline.periodTicks();

		JsonObject result = new JsonObject();
		result.addProperty("timeOfDay", currentTicks);
		if (periodTicks.isPresent() && periodTicks.get() > 0) {
			long totalTicks = dayNightTimeline.getTotalTicks(clockManager);
			result.addProperty("day", totalTicks / periodTicks.get());
		} else {
			result.add("day", JsonNull.INSTANCE);
		}

		Integer dayStart = dayNightMarkerTicks == null ? null : dayNightMarkerTicks.get("day");
		Integer nightStart = dayNightMarkerTicks == null ? null : dayNightMarkerTicks.get("night");
		if (dayStart != null && nightStart != null && periodTicks.isPresent() && periodTicks.get() > 0) {
			boolean isDay = dayStart <= nightStart
				? (currentTicks >= dayStart && currentTicks < nightStart)
				: (currentTicks >= dayStart || currentTicks < nightStart);
			result.addProperty("isDay", isDay);
		} else {
			result.add("isDay", JsonNull.INSTANCE);
		}
		return result;
	}
}
