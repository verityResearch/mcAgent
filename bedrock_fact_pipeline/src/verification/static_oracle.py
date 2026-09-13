from typing import Any, Dict


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        try:
            return sorted(expected) == sorted(actual)
        except TypeError:
            return False
    return expected == actual


def check_fact_seeded(candidate: Dict[str, Any], fact: Dict[str, Any]) -> Dict[str, Any]:
    # structured_claim matching alone previously admitted a candidate whose
    # answer field was a raw JSON array instead of a natural-language
    # sentence -- found via a live base-vs-fine-tuned model comparison on
    # the sibling fact_pipeline (identical oracle code, same gap). Ported
    # here since bedrock_fact_pipeline is exposed to the same risk.
    answer = candidate.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {"success": False, "errors": [f"answer must be a non-empty string, got {answer!r}"],
                "failure_kind": "malformed_answer"}
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
