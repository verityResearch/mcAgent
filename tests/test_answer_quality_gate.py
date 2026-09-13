from src.verification.judge import Judge


def _gate(answer):
    return Judge._answer_quality_gate({"answer": answer})


def test_rejects_bare_number():
    v = _gate("0")
    assert v is not None and v["failure_kind"] == "degenerate_answer"


def test_rejects_empty_and_punctuation():
    assert _gate("") is not None
    assert _gate("   ") is not None
    assert _gate("---") is not None
    assert _gate("42") is not None


def test_rejects_non_string():
    assert _gate(None) is not None
    assert _gate(["a"]) is not None


def test_accepts_short_real_answer():
    # a terse but genuine prose answer must still pass (near-zero false positive)
    assert _gate("Use a bed.") is None
    assert _gate("Feed them carrots.") is None


def test_accepts_normal_answer():
    assert _gate("The most efficient early-game food source is wheat, farmed into bread.") is None


def test_single_word_alpha_rejected():
    # one lone word is not a substantive free-recall answer
    assert _gate("Wheat") is not None
