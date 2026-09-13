"""Mechanical static oracle for fact-seeded candidates.

Diffs a fact-seeded candidate's ``structured_claim`` against the ground-truth
``FactRecord`` it was seeded from. This is a pure data comparison with no LLM
call: it is the load-bearing ground-truth check that lets a fact-seeded sample
be called ``verified``.
"""
import re
from typing import Any, Dict

# Refusal / non-answer phrasings the teacher emits when the archetype question
# reads as ambiguous (e.g. asking what a single ingredient of a multi-ingredient
# recipe "produces"). The structured_claim still restates the given fact, so the
# field diff passes -- but the answer text is useless-to-harmful supervision
# ("the fact does not provide information..."). Deliberately anchored to the
# refusal's OBJECT (information / what / the fact) so ordinary prose that merely
# contains "do not provide <benefits>" isn't flagged -- verified against real
# generated answers ("wild wolves do not provide these benefits" must NOT match).
_REFUSAL_RE = re.compile(
    # negated "provide/specify/..." whose object is information / what / details
    r"\b(?:does not|doesn't|do not|don't|cannot|can't|could not|couldn't|unable to)\s+"
    r"(?:provide|specify|state|contain|mention|determine|tell(?:\s+\w+)?|say|indicate|give|include)\s+"
    r"(?:any\s+|us\s+|you\s+|me\s+)?(?:information|details|what|which|whether|the\s+(?:answer|result|recipe))"
    # ... or an explicit "no information about" ...
    r"|\bno information\b"
    # ... or the fact/answer explicitly disclaiming itself
    r"|\bthe\s+(?:given\s+|provided\s+)?fact\b[^.]*\b(?:does not|doesn't|do not|don't)\s+"
    r"(?:provide|specify|state|contain|mention|indicate)",
    re.IGNORECASE,
)


def _is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def _values_match(expected: Any, actual: Any) -> bool:
    """Return whether ``actual`` matches ``expected``.

    List-valued fields are compared as order-insensitive multisets so that a
    restated claim need not preserve the FactRecord's element ordering; every
    other value is compared for equality.
    """
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        try:
            return sorted(expected) == sorted(actual)
        except TypeError:
            return False
    return expected == actual


def check_fact_seeded(candidate: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, Any]:
    """Mechanically verify a fact-seeded ``candidate`` against ``fact``.

    Compares ``candidate["structured_claim"]`` field-by-field against
    ``fact["fields"]`` and returns a verdict dict with ``success`` (bool),
    ``errors`` (list of human-readable mismatch descriptions), and
    ``failure_kind`` (``"malformed_answer"`` when ``answer`` is missing, empty,
    or not a string, ``"malformed_claim"`` when the claim is missing or not an
    object, ``"fact_mismatch"`` when one or more fields disagree, otherwise
    ``None``). No LLM is consulted; the check is a deterministic data diff.

    The ``answer`` check exists because ``structured_claim`` matching alone
    previously admitted a candidate whose ``answer`` field was a raw JSON
    array (identical in content to ``structured_claim``'s own list field)
    instead of the natural-language sentence the schema requires -- found via
    a live base-vs-fine-tuned model comparison (see project docs); this
    verifier had no way to catch it because it never looked at ``answer`` at
    all.
    """
    answer = candidate.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {"success": False, "errors": [f"answer must be a non-empty string, got {answer!r}"],
                "failure_kind": "malformed_answer"}
    if _is_refusal(answer):
        return {"success": False,
                "errors": [f"answer is a refusal/non-answer, not a real answer: {answer.strip()[:120]!r}"],
                "failure_kind": "refusal_answer"}
    claim = candidate.get("structured_claim")
    if not isinstance(claim, dict):
        return {"success": False, "errors": ["structured_claim missing or not an object"],
                "failure_kind": "malformed_claim"}
    errors = []
    for key, expected in fact["fields"].items():
        actual = claim.get(key)
        if not _values_match(expected, actual):
            errors.append(f"field {key!r}: expected {expected!r}, got {actual!r}")
    if errors:
        return {"success": False, "errors": errors, "failure_kind": "fact_mismatch"}
    return {"success": True, "errors": [], "failure_kind": None}
