"""Numeric-claim oracle -- mechanical checks for quantitative claims in prose.

Free-recall answers embed specific numbers ("5 game ticks (0.25 seconds)",
"Sharpness goes up to level 5"). Most numbers in a strategy answer are
subjective (wall thickness, tunnel spacing) or lack any report ground truth
(crop growth is probabilistic, not a fixed duration) -- checking those would
be high false-positive, so they are deliberately left to the judge.

This oracle checks only the two numeric classes that ARE mechanically
ground-truthable with near-zero false positives:

  1. tick <-> time arithmetic. Minecraft runs at a hard 20 ticks/second, so a
     stated "N ticks (M seconds)" (or minutes) pair is verifiable by pure
     arithmetic -- no lookup, no ambiguity. A mismatch is a definite error.

  2. enchantment max-level claims, checked against the pinned data report's
     enchantment facts (the same ground truth the fact-seeded oracle uses).

Both were false-positive tested against real generated free-recall answers.
"""
import re
from typing import Any, Dict, List, Optional

TICKS_PER_SECOND = 20

# "N tick(s) (M second(s))" or "N tick(s) (M minute(s))" -- a tick count with a
# parenthetical real-time equivalent the model stated itself.
_TICK_TIME_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:game\s+)?ticks?\b\s*\(\s*(\d+(?:\.\d+)?)\s*(seconds?|minutes?)\b",
    re.IGNORECASE,
)

# "<Enchantment> ... (max|maximum) level ... N"  and the reverse
# "(max|maximum) level ... of <Enchantment> ... N". Kept tight: the enchantment
# name and the number must sit close together with a max-level phrase between.
_ENCH_LEVEL_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+(?:of|the)\s+[A-Z][a-z]+)?)\b[^.]{0,40}?"
    r"\b(?:max(?:imum)?\s+level|maximum\s+enchantment\s+level|goes\s+up\s+to\s+level|"
    r"caps?\s+at\s+level|up\s+to\s+level)\b[^.]{0,20}?\b(\d+)\b",
    re.IGNORECASE,
)


def check_tick_time(text: str) -> List[str]:
    """Return descriptions of any tick<->time pair whose arithmetic is wrong.

    ``N ticks`` should equal ``N / 20`` seconds. Allows a small tolerance for
    the model rounding (e.g. 0.25 stated for 5 ticks is exact; 0.3 would be
    flagged). Empty list = all stated conversions are internally consistent.
    """
    errors: List[str] = []
    for m in _TICK_TIME_RE.finditer(text or ""):
        ticks = float(m.group(1))
        stated = float(m.group(2))
        unit = m.group(3).lower()
        stated_seconds = stated * 60 if unit.startswith("minute") else stated
        expected = ticks / TICKS_PER_SECOND
        if abs(stated_seconds - expected) > max(0.05, expected * 0.05):
            errors.append(
                f"{int(ticks) if ticks.is_integer() else ticks} ticks stated as "
                f"{m.group(2)} {unit} (should be {expected:g} seconds at 20 ticks/s)")
    return errors


def _ench_max_levels(enchantment_facts: List[Dict[str, Any]]) -> Dict[str, int]:
    """name (lowercased, no namespace) -> max_level, from report enchantment facts."""
    out: Dict[str, int] = {}
    for f in enchantment_facts:
        name = f["subject_id"].split(":", 1)[-1].replace("_", " ").lower()
        ml = f["fields"].get("max_level")
        if isinstance(ml, int):
            out[name] = ml
    return out


def check_enchantment_levels(text: str, max_levels: Dict[str, int]) -> List[str]:
    """Return descriptions of stated enchantment max-levels that disagree with
    the report. ``max_levels`` maps enchantment name -> true max level.

    Conservative: only fires when the matched name is a known enchantment, so a
    stray "level 5" about something else is ignored.
    """
    errors: List[str] = []
    for m in _ENCH_LEVEL_RE.finditer(text or ""):
        name = m.group(1).strip().lower()
        stated = int(m.group(2))
        true = max_levels.get(name)
        if true is not None and stated != true:
            errors.append(f"{name!r} max level stated as {stated}, actual is {true}")
    return errors


def check_numeric_claims(candidate: Dict[str, Any],
                         max_levels: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Verifier-shaped numeric check over a candidate's reasoning+answer text.

    Returns the standard ``{success, errors, failure_kind}``. ``failure_kind``
    is ``"numeric_inconsistent"`` on a violation. ``max_levels`` (from
    ``_ench_max_levels``) enables the enchantment check; omit it to run only the
    lookup-free tick/time arithmetic check.
    """
    text = " ".join(
        v for v in (candidate.get("reasoning"), candidate.get("answer")) if isinstance(v, str))
    errors = check_tick_time(text)
    if max_levels:
        errors += check_enchantment_levels(text, max_levels)
    if errors:
        return {"success": False, "errors": errors, "failure_kind": "numeric_inconsistent"}
    return {"success": True, "errors": [], "failure_kind": None}
