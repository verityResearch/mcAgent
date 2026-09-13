from src.data_report.schema import (
    adapt_jukebox_song,
    adapt_painting_variant,
    adapt_villager_trade,
)


RABBIT_TRADE = {
    "gives": {"id": "minecraft:emerald"},
    "max_uses": 16.0,
    "reputation_discount": 0.05,
    "wants": {"count": 4.0, "id": "minecraft:rabbit"},
    "xp": 2.0,
}


def test_villager_trade_extracts_and_int_coerces():
    r = adapt_villager_trade("minecraft:butcher/1/rabbit_emerald", RABBIT_TRADE)[0]
    assert r["category"] == "villager_trade"
    assert r["fields"] == {
        "wants_item": "minecraft:rabbit", "wants_count": 4,
        "gives_item": "minecraft:emerald", "gives_count": 1,
        "max_uses": 16, "xp": 2,
    }


def test_villager_trade_requires_wants_and_gives():
    assert adapt_villager_trade("minecraft:x", {"gives": {"id": "minecraft:emerald"}}) == []
    assert adapt_villager_trade("minecraft:x", {"wants": {"id": "minecraft:rabbit"}}) == []


def test_jukebox_song_length_and_sound():
    song = {"comparator_output": 1, "length_in_seconds": 178.0, "sound_event": "minecraft:music_disc.13"}
    r = adapt_jukebox_song("minecraft:13", song)[0]
    assert r["fields"] == {
        "length_in_seconds": 178, "comparator_output": 1, "sound_event": "minecraft:music_disc.13",
    }


def test_jukebox_song_requires_length_and_sound():
    assert adapt_jukebox_song("minecraft:x", {"comparator_output": 1}) == []


def test_painting_variant_dimensions():
    r = adapt_painting_variant("minecraft:skeleton", {"width": 4, "height": 3, "asset_id": "minecraft:skeleton"})[0]
    assert r["fields"] == {"width": 4, "height": 3}


def test_painting_variant_requires_int_dims():
    assert adapt_painting_variant("minecraft:x", {"width": "4", "height": 3}) == []
