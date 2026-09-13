"""The offline lookup tool + verifier over minecraft.db (build_db.py).

Two roles, both backed by the same DB:
  - lookup_*  : the TOOL the model queries at inference (returns ground truth).
  - verify_*  : the VERIFIER that checks a proposed item against ground truth,
                so a fabricated item (e.g. "corn" for pigs, or a real item
                for the WRONG animal) is rejected before it reaches an answer.

Lookup returns are table-grounded. The surrounding model can still skip a
lookup or add unsupported prose, so this module alone does not verify a final
free-form answer.
"""
import sqlite3
from typing import List


class OracleDB:
    def __init__(self, db_path: str):
        # check_same_thread=False: this class is read-only at inference time
        # (only build_db.py writes), so it's safe to share one connection
        # across threads -- needed by the knowledge server, whose FastAPI sync
        # routes run in a threadpool, not the thread that opened the DB.
        self.con = sqlite3.connect(db_path, check_same_thread=False)

    def _ns(self, item: str) -> str:
        return item if item.startswith(("minecraft:", "#")) else f"minecraft:{item}"

    # --- TOOL: lookups (ground truth) ---
    def breeding_foods(self, animal: str) -> List[str]:
        return [r[0] for r in self.con.execute(
            "SELECT food_item FROM breeding_food WHERE animal=? COLLATE NOCASE ORDER BY food_item",
            (animal,))]

    def recipe_ingredients(self, result_item: str) -> List[str]:
        return [r[0] for r in self.con.execute(
            "SELECT ingredient FROM recipe WHERE result_item=? COLLATE NOCASE ORDER BY ingredient",
            (self._ns(result_item),))]

    def recipe_ingredient_groups(self, result_item: str) -> List[str]:
        """Like recipe_ingredients but AND/OR-structure-preserving: one entry
        per AND-slot (a crafting-grid position), "/"-joined when that slot has
        multiple interchangeable OR-alternatives (e.g. "coal/charcoal"). Falls
        back to recipe_ingredients's flat one-slot-per-item shape when the DB
        predates recipe_slot (a test fixture, or an older build)."""
        key = self._ns(result_item)
        try:
            methods = [r[0] for r in self.con.execute(
                "SELECT DISTINCT method FROM recipe_slot WHERE result_item=? COLLATE NOCASE", (key,))]
        except sqlite3.OperationalError:
            methods = []
        if not methods:
            return self.recipe_ingredients(result_item)
        method = key if key in methods else sorted(methods)[0]
        rows = self.con.execute(
            "SELECT slot, ingredient FROM recipe_slot WHERE result_item=? COLLATE NOCASE "
            "AND method=? ORDER BY slot", (key, method)).fetchall()
        groups: dict = {}
        for slot, ingredient in rows:
            groups.setdefault(slot, []).append(ingredient)
        return ["/".join(dict.fromkeys(items)) for _, items in sorted(groups.items())]

    def loot_sources(self, drop_item: str) -> List[str]:
        """Reverse of loot_drops: which blocks/mobs drop this item -- answers
        "where do I find/get X" honestly from real loot data. Deliberately
        restricted to blocks/ and entities/ sources (mine it / kill it) -- other
        loot-pool sources (chests, pot sherds, piglin bartering) have ugly raw
        ids ("chests/village/village_toolsmith") that don't make good player-
        facing phrasing, so they're left out rather than named badly. This
        makes the answer a true but partial "where to find it", not exhaustive.

        Also drops the trivial self-source: breaking a placed block always
        gives that same block back (blocks/light_blue_carpet -> light_blue_carpet),
        which is real loot data but a useless "where do I find X" answer for
        anything player-placed/crafted (you don't find carpet in the world,
        you place it). Genuinely different sources for the same-family block
        (e.g. flowering_azalea's own leaves/potted variant) are kept."""
        key = self._ns(drop_item)
        drop_bare = key.split(":", 1)[-1]
        rows = [r[0] for r in self.con.execute(
            "SELECT DISTINCT source FROM loot WHERE drop_item=? COLLATE NOCASE ORDER BY source",
            (key,))]
        return [s for s in rows if s.startswith(("minecraft:blocks/", "minecraft:entities/"))
                and s.split("/")[-1] != drop_bare]

    def enchantment_max_level(self, name: str):
        # Case-insensitive + tolerant of an "_enchant"/"_enchantment" suffix the
        # model sometimes appends from phrasing ("the Efficiency enchant"), so a
        # minor key slip resolves instead of falsely declining.
        key = self._ns(name)
        for cand in (key, key + "_enchant", key + "_enchantment",
                     key.replace("_enchantment", "").replace("_enchant", "")):
            row = self.con.execute(
                "SELECT max_level FROM enchantment WHERE name=? COLLATE NOCASE", (cand,)).fetchone()
            if row:
                return row[0]
        return None

    def loot_drops(self, source: str) -> List[str]:
        return [r[0] for r in self.con.execute(
            "SELECT drop_item FROM loot WHERE source=? COLLATE NOCASE ORDER BY drop_item", (source,))]

    def tag_members(self, tag: str) -> List[str]:
        return [r[0] for r in self.con.execute(
            "SELECT member FROM tag WHERE tag=? COLLATE NOCASE ORDER BY member", (self._ns(tag),))]

    def trade(self, trade: str):
        """(wants_item, wants_count, gives_item, gives_count) or None."""
        return self.con.execute(
            "SELECT wants_item, wants_count, gives_item, gives_count FROM villager_trade "
            "WHERE trade=? COLLATE NOCASE", (self._ns(trade),)).fetchone()

    def jukebox_length(self, song: str):
        row = self.con.execute(
            "SELECT length_seconds FROM jukebox_song WHERE song=? COLLATE NOCASE",
            (self._ns(song),)).fetchone()
        return row[0] if row else None

    def painting_size(self, painting: str):
        """(width, height) or None."""
        return self.con.execute(
            "SELECT width, height FROM painting WHERE painting=? COLLATE NOCASE",
            (self._ns(painting),)).fetchone()

    def item_use(self, item: str):
        """(nutrition, saturation, max_damage, attack_damage, equip_slot) or None.
        Any field may be None -- an item can have some but not all (e.g. a sword
        has max_damage+attack_damage but no food/equip_slot)."""
        return self.con.execute(
            "SELECT nutrition, saturation, max_damage, attack_damage, equip_slot "
            "FROM item_use WHERE item=? COLLATE NOCASE", (self._ns(item),)).fetchone()

    def ore_depths(self, block: str):
        """(placement, dist_type, min_y, max_y, count) rows for an ore block,
        highest per-chunk attempt count first -- real vanilla worldgen Y-level
        data (see build_db.py's ore_depth table docstring for how count is
        derived; it's a comparable frequency proxy, not a true probability)."""
        return self.con.execute(
            "SELECT placement, dist_type, min_y, max_y, count FROM ore_depth "
            "WHERE block=? COLLATE NOCASE ORDER BY count DESC",
            (self._ns(block),)).fetchall()

    # --- VERIFIER: membership checks (reject fabrications) ---
    def is_breeding_food(self, item: str, animal: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM breeding_food WHERE animal=? AND food_item=?",
            (animal, self._ns(item))).fetchone() is not None

    def is_recipe_ingredient(self, item: str, result_item: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM recipe WHERE result_item=? AND ingredient=?",
            (self._ns(result_item), self._ns(item))).fetchone() is not None

    def close(self):
        self.con.close()
