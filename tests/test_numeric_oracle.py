from src.verification.numeric_oracle import (
    check_enchantment_levels,
    check_numeric_claims,
    check_tick_time,
)


def test_tick_time_correct_passes():
    # 5 ticks / 20 = 0.25s -- the real (correct) claim from the corpus
    assert check_tick_time("stuns the player after 5 game ticks (0.25 seconds)") == []
    assert check_tick_time("100 ticks (5 seconds) of fire resistance") == []
    assert check_tick_time("1200 ticks (1 minute) later") == []


def test_tick_time_wrong_flagged():
    errs = check_tick_time("the effect lasts 5 ticks (0.5 seconds)")
    assert len(errs) == 1 and "0.25 seconds" in errs[0]
    assert check_tick_time("100 ticks (10 seconds)")  # 100/20=5, not 10


def test_tick_time_ignores_bare_numbers():
    # no tick<->time pair -> nothing to check
    assert check_tick_time("Wheat takes about 8 minutes to grow") == []
    assert check_tick_time("build a wall 2 blocks thick") == []


def test_enchantment_level_correct_passes():
    mx = {"sharpness": 5, "protection": 4, "efficiency": 5}
    assert check_enchantment_levels("Sharpness goes up to level 5", mx) == []
    assert check_enchantment_levels("Protection has a maximum level of 4", mx) == []


def test_enchantment_level_wrong_flagged():
    mx = {"sharpness": 5, "protection": 4}
    errs = check_enchantment_levels("Sharpness caps at level 6", mx)
    assert len(errs) == 1 and "sharpness" in errs[0] and "actual is 5" in errs[0]


def test_enchantment_unknown_name_ignored():
    mx = {"sharpness": 5}
    # 'bogus' is not a known enchantment -> not flagged even with a number
    assert check_enchantment_levels("Bogus goes up to level 9", mx) == []


def test_check_numeric_claims_verdict_shape():
    mx = {"sharpness": 5}
    bad = {"reasoning": "It lasts 5 ticks (0.5 seconds).", "answer": "ok"}
    v = check_numeric_claims(bad, mx)
    assert v["success"] is False and v["failure_kind"] == "numeric_inconsistent"

    good = {"reasoning": "It lasts 5 ticks (0.25 seconds).", "answer": "Sharpness maxes at level 5."}
    assert check_numeric_claims(good, mx)["success"] is True
