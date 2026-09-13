"""Minecraft registry-existence oracle -- layer 0 of the domain verifier.

The strict-C flagship's power is that GCC mechanically rejects a program that
references an undeclared identifier. This is the Minecraft analog: the pinned
data report's ``reports/registries.json`` is the canonical list of every id
that actually exists in the game (items, blocks, entities, enchantments,
biomes, ... -- 95 registries). Any claim that names a ``minecraft:*`` id, or a
plain noun that resolves to no real id, is mechanically false -- no LLM needed.

This catches the exact failure class the LLM judge misses: a fabricated entity
(e.g. "feed pigs corn" -- there is no ``minecraft:corn``) or a hallucinated
recipe ingredient. Purely mechanical, ground-truthed against the same pinned
report the static fact oracle uses.

Scope: existence only. It does NOT check that a real id has the claimed
property/relationship -- that's the structured-fact oracle's job (static_oracle
+ its expansions). Existence is the cheap, high-precision first gate.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

# A conservative id token: minecraft:foo or minecraft:foo/bar. Lowercase,
# digits, underscore, slash -- matches the data report's id grammar and won't
# gobble surrounding prose.
_ID_RE = re.compile(r"\bminecraft:[a-z0-9_]+(?:/[a-z0-9_]+)*\b")


class MinecraftRegistry:
    """Loaded set of every real id, with a fuzzy plain-noun resolver."""

    def __init__(self, ids_by_registry: Dict[str, Set[str]]):
        self.ids_by_registry = ids_by_registry
        self.all_ids: Set[str] = set()
        for s in ids_by_registry.values():
            self.all_ids |= s
        # name index: bare name (no namespace, underscores->spaces) -> id, for
        # resolving prose nouns like "carrot" / "iron ingot" back to an id.
        self._by_name: Dict[str, str] = {}
        for i in self.all_ids:
            bare = i.split(":", 1)[-1].split("/")[-1]
            self._by_name.setdefault(bare, i)
            self._by_name.setdefault(bare.replace("_", " "), i)

    def exists(self, identifier: str, registry: Optional[str] = None) -> bool:
        if registry is not None:
            return identifier in self.ids_by_registry.get(registry, set())
        return identifier in self.all_ids

    def resolve_name(self, name: str) -> Optional[str]:
        """Best-effort map a plain noun to an id, or None if nothing matches.

        None means "no game object by this name exists" -- a strong signal the
        noun is fabricated (e.g. 'corn'). A hit does not prove the surrounding
        claim is true, only that the referenced object is real.
        """
        return self._by_name.get(name.strip().lower())


def load_registry(data_report_dir: str) -> MinecraftRegistry:
    """Load the registry from a pinned data report's ``reports/registries.json``.

    ``data_report_dir`` is the same root the fact sampler consumes (the parent
    of ``data/``); the registries live under its ``reports/`` sibling.
    """
    path = os.path.join(data_report_dir, "reports", "registries.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ids_by_registry: Dict[str, Set[str]] = {}
    for reg_name, body in raw.items():
        entries = body.get("entries", {}) if isinstance(body, dict) else {}
        ids_by_registry[reg_name] = set(entries.keys())
    return MinecraftRegistry(ids_by_registry)


def find_nonexistent_ids(text: str, registry: MinecraftRegistry) -> List[str]:
    """Return every ``minecraft:*`` id token in ``text`` that is NOT in the
    registry (deduped, in first-seen order). Empty list = all referenced ids
    are real.
    """
    seen: List[str] = []
    for tok in _ID_RE.findall(text or ""):
        if tok not in registry.all_ids and tok not in seen:
            seen.append(tok)
    return seen


# High-precision prose contexts where a noun is unambiguously asserted to be a
# real game object. Deliberately narrow -- we would rather miss a fabrication
# than flag a legitimate answer, because a false reject silently drops a good
# training sample. Each pattern captures the claimed object token.
_OBJECT_TYPE_SUFFIX = re.compile(r"\b([a-z][a-z]+(?:[ _][a-z]+)?)\s+(ore|ingot|nugget)\b", re.I)
_FEED_BREED = re.compile(r"\b(?:feed(?:ing)?|breed(?:ing)?)\b[^.]*?\bwith\s+([a-z][a-z]+(?:\s+[a-z]+)?)\b", re.I)
_FEED_DIRECT = re.compile(r"\bfeed(?:ing)?\s+(?:them|it|the\s+\w+|a\s+\w+|your\s+\w+)?\s*([a-z][a-z]+(?:\s+[a-z]+)?)\b", re.I)

# Words that show up in the captured slot but are grammar, not game objects.
_STOP = {
    "them", "it", "the", "a", "an", "with", "and", "or", "to", "your", "their",
    "this", "that", "some", "any", "using", "from", "these", "those", "each",
    "food", "items", "item", "block", "blocks", "player", "players",
}


def find_fabricated_objects(text: str, registry: MinecraftRegistry) -> List[str]:
    """Return nouns asserted (in prose) to be real game objects that resolve to
    NO registry id -- likely fabrications (e.g. "corn"). Combines definitive
    ``minecraft:*`` id misses with a few narrow, high-precision prose patterns.

    Conservative by design: only nouns in strong object-defining contexts are
    considered, and only a resolve MISS is reported. Empirically FP-tested
    against real generated free-recall answers before wiring into admission.
    """
    text = text or ""
    bad: List[str] = list(find_nonexistent_ids(text, registry))
    candidates: List[str] = []
    for m in _OBJECT_TYPE_SUFFIX.finditer(text):
        candidates.append(m.group(1))
    for rx in (_FEED_BREED, _FEED_DIRECT):
        for m in rx.finditer(text):
            candidates.append(m.group(1))
    for raw in candidates:
        noun = raw.strip().lower()
        if noun in _STOP or len(noun) < 3:
            continue
        # try the noun and its last word (e.g. "raw beef" -> "beef")
        if registry.resolve_name(noun) is None and registry.resolve_name(noun.split()[-1]) is None:
            label = f"(prose) {noun}"
            if label not in bad:
                bad.append(label)
    return bad


def check_registry(candidate: Dict[str, Any], registry: MinecraftRegistry) -> Dict[str, Any]:
    """Verifier-shaped existence gate for a candidate.

    Scans the candidate's ``structured_claim`` (fact-seeded) and/or ``answer``
    text for ``minecraft:*`` ids that don't exist. Returns the same
    ``{success, errors, failure_kind}`` shape as the static oracle so it can be
    chained ahead of it. ``failure_kind = "nonexistent_id"`` on a violation.
    """
    haystacks: List[str] = []
    claim = candidate.get("structured_claim")
    if claim is not None:
        haystacks.append(json.dumps(claim))
    for key in ("answer", "reasoning", "instruction"):
        v = candidate.get(key)
        if isinstance(v, str):
            haystacks.append(v)

    bad: List[str] = []
    for h in haystacks:
        for tok in find_nonexistent_ids(h, registry):
            if tok not in bad:
                bad.append(tok)
    if bad:
        return {"success": False,
                "errors": [f"references id(s) that do not exist in the pinned registry: {bad}"],
                "failure_kind": "nonexistent_id"}
    return {"success": True, "errors": [], "failure_kind": None}


def make_layered_fact_check(registry, structured_check):
    """Compose the registry existence gate (Layer 0) ahead of a structured-fact
    oracle (Layer 1), returning one ``(candidate, fact) -> verdict`` callable
    with the exact signature the orchestrator expects.

    Layer 0 rejects fabricated ids cheaply and mechanically before Layer 1
    checks the (now known-real) ids against the ground-truth fields. A Layer 0
    reject short-circuits -- there's no point diffing fields of a nonexistent
    id. This mirrors strict-C's "undeclared identifier fails before the type
    checker runs" ordering.
    """
    def _check(candidate, fact):
        existence = check_registry(candidate, registry)
        if not existence["success"]:
            return existence
        return structured_check(candidate, fact)
    return _check
