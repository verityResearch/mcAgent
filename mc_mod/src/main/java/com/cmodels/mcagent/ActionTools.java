package com.cmodels.mcagent;

import com.google.gson.JsonObject;

import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.MobCategory;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

import java.util.Optional;

/**
 * UPDATE (2026-08-21, design-review reply): the review read the
 * whole a/b/c framing as the wrong question -- Path B (this mod) was never
 * meant to generate action traces at all; that's Path A's (mc_bot's) job.
 * the reviewer's actual answer is option (d): "THE MOD IS THE ORACLE. THE BOT IS
 * THE AGENT" -- see McAgentMod.java's header and the new OracleHttpServer
 * class for the real architecture. This class is KEPT, per the reviewer's own
 * words, as "a legitimate mod UTILITY... just don't let it be 'the
 * agent'" -- an admin/debug convenience for a connected player, NOT the
 * embodied-agent action layer. The writeup below (option (c), built
 * BEFORE the reply arrived) is preserved as-is for its real technical
 * content and honest history, not because it's still the governing
 * architecture decision.
 *
 * Stage 4 (action tools) port -- OPTION (c) from the architecture-fork
 * question sent for review (subject "stage 6: action-tools
 * architecture fork needs your read (a/b/c below)", sent 2026-08-20T11:23Z).
 * No reply arrived after several hours and many Stop-hook-driven rounds --
 * the review request's own next_action already named this fallback ("if no
 * reply and forward progress is wanted anyway, default to (c) ... rather
 * than committing to (b) solo"), so this is that fallback being exercised,
 * not a new decision made without review.
 *
 * OPTION (c) is explicitly "scoped-down creative-superpowers actions on
 * the player's own character": teleport instead of pathfind, direct block
 * removal instead of tool-gated breaking, direct damage instead of melee
 * swing. This is DELIBERATELY NOT trace-parity-equivalent to
 * fact_pipeline/mc_bot's action_tools.js -- mc_bot's goto() is a real
 * mineflayer-pathfinder walk that can fail on terrain, take damage, get
 * stuck; this goto() is a teleport that always succeeds if the
 * coordinates are loadable. mc_bot's attack() is a melee-range swing that
 * needs real pathing to close distance (see mc_bot's own melee-range bug
 * history); this attack() applies damage directly regardless of distance.
 * A model trained on Fabric-mod traces generated this way would NOT be
 * learning the same skill as one trained on mc_bot's traces. If the reviewer's
 * reply ever arrives with a different read, this class may need to be
 * replaced or gated behind a "cheap mode" flag rather than presented as
 * equivalent -- flagged here so that divergence is never silently
 * forgotten.
 *
 * Deliberately the SMALLEST honest slice, not a full port: goto (teleport),
 * mine (direct block removal with real loot-table drops), equip (hotbar
 * reselection only -- does NOT move items between inventory sections, a
 * real, documented scope limit, not an oversight), attack (direct damage
 * to the nearest matching hostile mob, bypassing melee range). This is
 * exactly the three examples option (c) itself named in the original mail
 * for review -- not scope creep beyond what was already proposed. place/
 * craft/smelt/chest ops are NOT covered here.
 *
 * attack()'s target filter is CONSTRUCTION-level, not confirm-gated
 * policy, matching the reviewer's own requirement from the stage-4 tier-2/3
 * design review for mc_bot's attack(): a player is never even a
 * candidate. Two independent checks enforce this: EntityType#getCategory()
 * must equal MobCategory.MONSTER (a real, principled vanilla
 * classification -- players are MobCategory.MISC, not MONSTER), AND an
 * explicit `!(entity instanceof Player)` filter as defense in depth (not
 * relying solely on an unverified assumption about MobCategory's exact
 * enum values, which javap can't reveal since they're static field
 * VALUES, not signatures).
 *
 * Every method used was confirmed via javap against this project's own
 * Loom-mapped 26.2 jar before being written, same discipline as
 * StateTools/OracleDbBuilder: ServerPlayer#teleportTo(double,double,double)
 * (a real, simple public method -- not a guess); ServerLevel#destroyBlock
 * (BlockPos, boolean dropAll, Entity, int maxUpdateDepth), where dropAll=
 * true drives the SAME internal Block#dropResources/getDrops loot-table
 * machinery a real player-caused break would use, so the item that pops
 * out is authentic, not guessed at -- confirmed via javap on
 * net.minecraft.world.level.block.Block's drop-related methods;
 * Registry#getOptional(Identifier) for item-name lookup (Optional-based,
 * not the throwing getValue(), since a caller-supplied item name can be
 * wrong); Inventory#getItem/setSelectedSlot for hotbar reselection;
 * Entity#hurtServer(ServerLevel, DamageSource, float) for direct damage
 * application (the real server-side damage entry point, distinct from the
 * client-facing hurtClient()); DamageSources#playerAttack(Player) for an
 * authentic damage-source type (not DamageSources#generic(), so death
 * messages/knockback/enchantment interactions read as a real player hit);
 * LivingEntity#getAttributeValue(Attributes.ATTACK_DAMAGE) for a real,
 * non-arbitrary damage amount (base + equipped weapon + any modifiers,
 * not a guessed constant).
 */
final class ActionTools {
	private ActionTools() {
	}

	static JsonObject goto_(ServerPlayer player, double x, double y, double z) {
		player.teleportTo(x, y, z);
		JsonObject result = new JsonObject();
		result.addProperty("moved", true);
		result.add("position", StateTools.position(player));
		return result;
	}

	static JsonObject mine(ServerPlayer player, int x, int y, int z) {
		ServerLevel level = player.level();
		BlockPos pos = new BlockPos(x, y, z);
		BlockState state = level.getBlockState(pos);
		JsonObject result = new JsonObject();
		if (state.isAir()) {
			result.addProperty("mined", false);
			result.addProperty("reason", "no block at that position");
			return result;
		}
		String blockName = BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString();
		boolean destroyed = level.destroyBlock(pos, true, player, 512);
		result.addProperty("mined", destroyed);
		result.addProperty("block", blockName);
		return result;
	}

	static JsonObject equip(ServerPlayer player, String itemName) {
		Identifier id = Identifier.parse(itemName.contains(":") ? itemName : "minecraft:" + itemName);
		Optional<Item> item = BuiltInRegistries.ITEM.getOptional(id);
		JsonObject result = new JsonObject();
		if (item.isEmpty()) {
			result.addProperty("equipped", false);
			result.addProperty("reason", "unknown item: " + itemName);
			return result;
		}

		Inventory inventory = player.getInventory();
		int hotbarSlot = -1;
		for (int i = 0; i < 9; i++) {
			ItemStack stack = inventory.getItem(i);
			if (!stack.isEmpty() && stack.getItem() == item.get()) {
				hotbarSlot = i;
				break;
			}
		}
		if (hotbarSlot < 0) {
			result.addProperty("equipped", false);
			result.addProperty("reason", itemName
				+ " not found in the hotbar (slots 0-8) -- this action only reselects an"
				+ " existing hotbar slot, it doesn't move items between inventory sections");
			return result;
		}
		inventory.setSelectedSlot(hotbarSlot);
		result.addProperty("equipped", true);
		result.add("equipped_item", StateTools.itemStackOrNull(inventory.getSelectedItem()));
		return result;
	}

	static JsonObject attack(ServerPlayer player, String targetType, double radius) {
		Identifier id = Identifier.parse(targetType.contains(":") ? targetType : "minecraft:" + targetType);
		Optional<EntityType<?>> type = BuiltInRegistries.ENTITY_TYPE.getOptional(id);
		JsonObject result = new JsonObject();
		if (type.isEmpty()) {
			result.addProperty("attacked", false);
			result.addProperty("reason", "unknown entity type: " + targetType);
			return result;
		}
		if (type.get().getCategory() != MobCategory.MONSTER) {
			result.addProperty("attacked", false);
			result.addProperty("reason", targetType
				+ " is not a hostile mob -- this action only targets MobCategory.MONSTER entities"
				+ " (a player is never a candidate, matching mc_bot's own construction-level guard)");
			return result;
		}

		ServerLevel level = player.level();
		double r = Math.max(1, Math.min(radius, 64));
		AABB box = new AABB(player.blockPosition()).inflate(r);
		Vec3 playerPos = player.position();

		LivingEntity target = null;
		double bestDistance = Double.MAX_VALUE;
		for (Entity entity : level.getEntities(player, box,
				e -> e.getType() == type.get() && e.isAlive() && !(e instanceof Player))) {
			double distance = playerPos.distanceTo(entity.position());
			if (distance < bestDistance) {
				bestDistance = distance;
				target = (LivingEntity) entity;
			}
		}
		if (target == null) {
			result.addProperty("attacked", false);
			result.addProperty("reason", "no live " + targetType + " within " + r + " blocks");
			return result;
		}

		float damage = (float) player.getAttributeValue(Attributes.ATTACK_DAMAGE);
		boolean hurt = target.hurtServer(level, level.damageSources().playerAttack(player), damage);
		result.addProperty("attacked", hurt);
		result.addProperty("target", BuiltInRegistries.ENTITY_TYPE.getKey(target.getType()).toString());
		result.addProperty("damage_dealt", damage);
		result.addProperty("target_health_after", target.getHealth());
		result.addProperty("target_alive", target.isAlive());
		return result;
	}
}
