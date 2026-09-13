"""LLM judge for the free-recall lane.

Free-recall candidates have no ground-truth ``FactRecord`` to diff against, so
they cannot use the mechanical static oracle. Instead this judge runs two LLM
checks in sequence: an internal-consistency check followed by an adversarial
critique. It subclasses ``ChatClient`` so the HTTP call and JSON-envelope
handling stay in one place, and it emits the same verdict shape as the static
oracle so the orchestrator can treat both lanes uniformly.
"""
import asyncio
import re
from typing import Any, Dict, Optional

from src.llm_client import ChatClient
from src.tools.minecraft_wiki import MINECRAFT_WIKI_TOOL_SCHEMA, execute_minecraft_wiki_tool, fetch_minecraft_wiki_page
from src.verification.numeric_oracle import check_numeric_claims

_CRITIQUE_VOTES = 3
_CRITIQUE_VOTE_TEMPERATURE = 0.5

# Cheap local pre-filter for _check_mechanism_claims: most answers don't describe
# one block/item acting on another at all, and the LLM-based check has a real,
# measured cost per candidate -- at full diagnostic batch concurrency, adding it
# unconditionally overloaded a local single-instance mlx_lm.server (21/24 admitted
# dropped to 4/24, mostly timeouts; see project memory). Skipping the LLM call
# entirely when no verb from this allowlist appears trades a small, disclosed risk
# of missing an unusually-phrased claim for a large cut in the common case's added
# load. False negatives here just mean that candidate skips this one extra check,
# not that it skips the consistency check or critique ensemble -- the existing
# safety nets are unaffected.
_MECHANISM_VERB_PATTERN = re.compile(
    r"\b(open|opens|opened|close|closes|closed|power|powers|powered|trigger|triggers|triggered|"
    r"activate|activates|activated|control|controls|controlled|toggle|toggles|toggled|"
    r"extend|extends|extended|retract|retracts|retracted|push|pushes|pushed|pull|pulls|pulled|"
    r"ignite|ignites|ignited|unlock|unlocks|unlocked|lock|locks|locked)\b",
    re.IGNORECASE,
)

# Same pre-filter rationale as _MECHANISM_VERB_PATTERN, for a different claim
# shape: one block/item/mechanic having a specific capability or purpose (e.g.
# "droppers can be used to sort items by type", "a locator map is used to find
# a shipwreck") rather than one block acting on another. This is a fuzzier
# category than mechanism verbs -- capability claims don't share one small verb
# set -- so this pattern is intentionally broader and expected to over-trigger
# more often; the extraction call below is the real precision filter, and an
# over-triggered prefilter only costs one wasted extraction call, not a false
# rejection (extraction correctly reports claim_found: false for non-claims).
_CAPABILITY_CLAIM_PATTERN = re.compile(
    r"\b(can be used to|can be used for|is used to|is used for|used to|used for|"
    r"allows?|lets?|enables?|"
    r"locates?|locating|finds?|finding|tracks?|tracking|sorts?|sorting|"
    r"detects?|detecting|identifies?|identifying|filters?|filtering|"
    r"stores?|storing|converts?|converting|collects?|collecting)\b",
    re.IGNORECASE,
)

# Same pre-filter rationale again, for a third claim shape: comparing two things
# along some dimension (e.g. "deepslate diamond ore is more common than its
# regular variant" -- false, both yield the same amount; "an enchanted golden
# apple is stronger than a regular one" -- true). "\w+er than" is a deliberately
# broad catch-all for comparative adjectives (faster/rarer/stronger/cheaper/...
# than); "more/less [up to 4 words] than" catches multi-word noun-phrase
# comparisons ("deals more attack damage than", not just single-word "more
# common than" -- found live testing that a single-\w+ gap silently missed
# these); the bare "more/less X" forms catch the common case where a claim
# drops the explicit "than Y" and leaves the comparison target implicit from
# context (confirmed live: the deepslate-ore fabrication was phrased this way).
_COMPARATIVE_CLAIM_PATTERN = re.compile(
    r"\b(\w+er than|more (?:\w+\s+){0,3}\w+ than|less (?:\w+\s+){0,3}\w+ than|"
    r"more common|less common|more rare|rarer|more abundant|less abundant|"
    r"more durable|less durable|more effective|less effective|more efficient|less efficient|"
    r"better than|worse than|compared to|"
    r"contains? more|contains? less|"
    r"most common|most rare|most abundant)\b",
    re.IGNORECASE,
)


class Judge(ChatClient):
    """LLM judge that verifies free-recall candidates.

    The constructor is inherited unchanged from ``ChatClient``. ``judge`` is the
    public entry point; ``_check_consistency`` and ``_adversarial_critique`` are
    the two underlying LLM checks it sequences.

    ``enchantment_max_levels`` (a ``name -> max_level`` dict) is optional
    mechanical ground truth for the numeric pre-check; main.py sets it from the
    data report after construction. When unset, the numeric check still runs its
    lookup-free tick<->time arithmetic verification.
    """

    enchantment_max_levels = None

    async def _check_consistency(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the model whether the candidate is internally consistent.

        Returns the parsed judge response with ``consistent`` (bool) and
        ``issues`` (list of str) keys.
        """
        system = (
            "You check whether a Minecraft Q&A sample is internally consistent: does the reasoning "
            "actually support the answer, and does the answer address the question? "
            'Respond with JSON: {"consistent": bool, "issues": [str]}.'
        )
        user = (
            f"Question: {candidate['instruction']}\nReasoning: {candidate['reasoning']}\n"
            f"Answer: {candidate['answer']}"
        )
        return await self.call_chat(system, user, ["consistent", "issues"], temperature=0.0)

    async def _cast_critique_vote(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Ask a single adversarial critic to find a flaw in the candidate's answer.

        One independent vote in the ensemble ``_adversarial_critique`` runs. Returns
        the parsed judge response with ``critique_passed`` (bool) and ``issues``
        (list of str) keys.
        """
        system = (
            "You are reviewing a Minecraft answer for factual or logical errors.\n"
            "Before you flag any specific numeric or mechanical claim (a depth, a tool tier, an "
            "enchantment's behavior, a trade ratio, a crafting recipe, and similar details) as wrong, "
            "pause and honestly ask yourself: am I actually certain this is wrong and certain what the "
            "correct value is, or am I assuming -- or worse, guessing -- and it merely sounds like a "
            "confident correction? If there is any real doubt about either the claim or your own "
            "correction, use the search_minecraft_wiki tool to check before flagging it. Never state a "
            "'correction' you have not actually verified. Only flag an issue if you are confident "
            "(ideally wiki-verified) a specific claim in the answer is factually wrong or a step in the "
            "reasoning does not follow -- name the exact claim and state concretely why it is "
            "incorrect. Do not flag an answer merely because it is incomplete, omits alternative "
            "approaches, or states a subjective 'best' recommendation -- those are not errors. This "
            "does not apply to uncertainty about whether a specific claim is factually correct: that "
            "uncertainty is exactly what the search_minecraft_wiki tool is for, and must be resolved "
            "by checking, not by defaulting to pass. Only pass a claim without checking it if you are "
            "genuinely certain it is correct. "
            'Respond with JSON: {"critique_passed": bool, "issues": [str]}.'
        )
        user = f"Question: {candidate['instruction']}\nAnswer: {candidate['answer']}"
        return await self.call_chat_with_tools(
            system, user, ["critique_passed", "issues"],
            tools=[MINECRAFT_WIKI_TOOL_SCHEMA], tool_executor=execute_minecraft_wiki_tool,
            temperature=_CRITIQUE_VOTE_TEMPERATURE)

    async def _check_mechanism_claims(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated check for claims that one block/item mechanically acts on another.

        Two things were tried and failed before this design, both worth recording
        (see project memory for the full postmortem): folding mechanism-specific
        guidance into the general adversarial critique prompt reproducibly regressed
        an unrelated, previously-reliable fact -- appending more instructions to an
        already-long prompt destabilized behavior it wasn't meant to touch. A
        follow-up dedicated tool-calling check (this method's first version) avoided
        that, but traced evidence showed the model looked up the block/item it was
        already confident about (e.g. "Piston") rather than the one actually in
        dispute (e.g. "Door") even when explicitly told to look up the latter --
        tool-*selection* isn't reliably steerable by instruction on this model.

        This version removes that choice from the model entirely: a first plain call
        extracts ``actor``/``acted_upon`` as structured fields (a much easier task
        than picking a tool target while reasoning about correctness at the same
        time), the code fetches the wiki pages for both ``acted_upon`` and ``actor``
        directly (best-effort for ``actor`` -- see below), and a second plain call
        verifies the claim using only that fetched text -- the model is never given
        a choice of what to look up.

        Fetching only ``acted_upon`` was tried first and found, via live testing, to
        under-cover: a page about the thing being acted on doesn't always spell out
        every way other things interact with it (Minecraft Wiki's "Door" article
        never mentions pistons at all, in an extract long enough to include unrelated
        detail like door hinge-placement rules), so a claim like "a piston opens the
        door" had no evidence to contradict it even though it's wrong. The actor's
        own page usually documents what it actually does -- "Piston" explicitly
        describes pushing entities and blocks, nothing about opening doors -- which
        is real disambiguating signal the acted_upon page alone was missing.

        Returns a dict with ``mechanism_claim_found`` (bool), ``verified_correct``
        (bool, meaningless if no claim was found), and ``issues`` (list of str).
        """
        if not _MECHANISM_VERB_PATTERN.search(candidate["answer"]):
            return {"mechanism_claim_found": False, "verified_correct": True, "issues": []}

        extract_system = (
            "Read this Minecraft answer and look for a specific kind of claim: one block or item "
            "mechanically causing, controlling, opening, powering, or otherwise acting on another "
            "(e.g. 'a piston opens the door', 'a torch powers the rail'). Ignore every other kind of "
            "claim -- numeric depths, tool tiers, trade ratios, and similar are not this check's job. "
            'If the answer contains no such claim, respond with {"claim_found": false, "actor": "", '
            '"acted_upon": "", "claim": ""}. If it does, respond with JSON: {"claim_found": bool, '
            '"actor": str, "acted_upon": str, "claim": str}, where ``actor`` is the block/item doing '
            'the acting (e.g. "Piston"), ``acted_upon`` is the block/item it claims to act on (e.g. '
            '"Door") -- these must name two different things -- and ``claim`` is a one-sentence '
            "restatement of the specific mechanism claimed."
        )
        extract_user = f"Answer: {candidate['answer']}"
        extraction = await self.call_chat(
            extract_system, extract_user, ["claim_found", "actor", "acted_upon", "claim"], temperature=0.0)
        if not extraction["claim_found"] or not extraction.get("acted_upon"):
            return {"mechanism_claim_found": False, "verified_correct": True, "issues": []}

        reference = await fetch_minecraft_wiki_page(extraction["acted_upon"])
        if reference.startswith(("Error:", "No wiki page found", "No content found")):
            # A failed lookup (network error, encoding issue, malformed or missing page
            # title -- observed live: a compound title like "Lever or button" tripped a
            # wiki-side response mlx couldn't decode) is uncertainty about the claim, not
            # evidence against it. The same calibration principle used everywhere else in
            # this judge applies here too: don't reject on failure to verify, only on an
            # actually confirmed problem. Skip the verify call entirely rather than handing
            # an error string to the model as if it were real reference material and
            # trusting it to recognize that on its own.
            return {"mechanism_claim_found": True, "verified_correct": True,
                    "issues": [f"Could not verify (wiki lookup for {extraction['acted_upon']!r} failed): "
                               f"{reference}"]}

        # Also fetch the actor's own page. A wiki article about the thing being acted
        # on doesn't always spell out every way other blocks interact with it (e.g.
        # "Door" never mentions pistons at all), but the actor's own page usually does
        # describe what it acts on -- "Piston" explicitly says it pushes entities and
        # blocks, nothing about opening doors -- which is real disambiguating signal
        # the acted_upon page alone can miss. Best-effort: acted_upon succeeding is
        # the only hard precondition, so a failed actor fetch just means verifying
        # with less context, not aborting the check.
        actor_reference = await fetch_minecraft_wiki_page(extraction["actor"])
        if actor_reference.startswith(("Error:", "No wiki page found", "No content found")):
            actor_reference = "(unavailable)"

        verify_system = (
            "You are given a claimed Minecraft mechanism and reference material about both the block/item "
            "doing the acting and the one being acted on. Decide, using ONLY the reference material below "
            "-- not your own recollection -- whether the claim is accurate.\n"
            "There are two different ways a reference can fail to support a claim, and they call for "
            "different verdicts. (1) The reference is silent: it's about something else, too sparse, or "
            "doesn't describe any mechanism for the actor at all -- this is a failed verification attempt, "
            "not a contradiction, so set verified_correct to true (the claim passes, unconfirmed) and note "
            "in issues that it could not be checked. (2) The reference DOES describe how the actor actually "
            "behaves or what it actually does -- and that described mechanism is a different one than the "
            "claim describes. That is a real contradiction even without an explicit denial: a description "
            "of what something does implies it doesn't also do an unlisted, unrelated thing, the same way "
            "a page that says a lever is flipped by hand contradicts a claim that a lever opens by itself. "
            "In that case set verified_correct to false. "
            "Being unable to verify a claim is never grounds by itself to treat it as wrong -- only case "
            "(2), an actually described but different mechanism, is. "
            'Respond with JSON: {"verified_correct": bool, "issues": [str]}.'
        )
        verify_user = (
            f"Claim: {extraction['claim']}\n\n"
            f"Reference material about {extraction['acted_upon']!r} (the thing acted on):\n{reference}\n\n"
            f"Reference material about {extraction['actor']!r} (the thing doing the acting):\n{actor_reference}"
        )
        verdict = await self.call_chat(
            verify_system, verify_user, ["verified_correct", "issues"], temperature=0.0)
        return {"mechanism_claim_found": True, "verified_correct": verdict["verified_correct"],
                "issues": verdict["issues"]}

    async def _check_capability_claims(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated check for claims that a single block/item/mechanic has a specific
        capability or purpose (e.g. "droppers sort items by type", "a locator map finds
        shipwrecks") -- a different claim shape than ``_check_mechanism_claims``, which
        is about one block acting on *another*. Found live, reviewing a broadened
        topic-bank diagnostic: two confirmed fabrications of exactly this shape slipped
        through uncaught because the mechanism check structurally can't see them (no
        second block/item is claimed to be acted on) -- droppers don't have any
        inherent type-sorting logic (they just eject/move items; real sorting needs
        hopper+comparator filtering), and locator maps track terrain and players, not
        specific structures (that's a different item, the explorer map).

        Reuses the design that worked for the mechanism check after its own iteration:
        the model only extracts the subject and the claimed capability as structured
        fields (no tool choice), the code fetches the subject's own wiki page directly,
        and a plain call verifies using only that text, with the same "silent vs.
        actually-describes-something-different" distinction that mechanism-claim
        verification needed to get right (see project memory for that full postmortem
        -- an earlier, more ambiguous version of that distinction caused real false
        rejections before being reworded explicitly).

        Only one page is fetched here, unlike the two-page mechanism check: a
        capability claim is about what the subject itself can do, and the subject's
        own page is where the Wiki documents that -- there's no second "actor" party
        whose page might hold the disambiguating fact instead.

        Returns a dict with ``capability_claim_found`` (bool), ``verified_correct``
        (bool, meaningless if no claim was found), and ``issues`` (list of str).
        """
        if not _CAPABILITY_CLAIM_PATTERN.search(candidate["answer"]):
            return {"capability_claim_found": False, "verified_correct": True, "issues": []}

        extract_system = (
            "Read this Minecraft answer and look for a specific kind of claim: a block, item, or "
            "mechanic having a specific capability, function, or purpose (e.g. 'droppers can sort "
            "items by type', 'a locator map is used to find a shipwreck', 'bone meal instantly grows "
            "crops'). Ignore every other kind of claim -- numeric depths, tool tiers, trade ratios, "
            "and one block directly acting on another block (a different check handles that) are not "
            "this check's job. If the answer contains no such claim, respond with "
            '{"claim_found": false, "subject": "", "claim": ""}. If it does, respond with JSON: '
            '{"claim_found": bool, "subject": str, "claim": str}, where ``subject`` is the block, '
            'item, or mechanic the capability is claimed about (e.g. "Dropper", "Locator Map") and '
            "``claim`` is a one-sentence restatement of the specific capability or purpose claimed."
        )
        extract_user = f"Answer: {candidate['answer']}"
        extraction = await self.call_chat(
            extract_system, extract_user, ["claim_found", "subject", "claim"], temperature=0.0)
        if not extraction["claim_found"] or not extraction.get("subject"):
            return {"capability_claim_found": False, "verified_correct": True, "issues": []}

        reference = await fetch_minecraft_wiki_page(extraction["subject"])
        if reference.startswith(("Error:", "No wiki page found", "No content found")):
            # Same calibration principle as everywhere else in this judge: a failed
            # lookup is uncertainty about the claim, not evidence against it.
            return {"capability_claim_found": True, "verified_correct": True,
                    "issues": [f"Could not verify (wiki lookup for {extraction['subject']!r} failed): "
                               f"{reference}"]}

        verify_system = (
            "You are given a claimed Minecraft capability or purpose and reference material about the "
            "specific block, item, or mechanic it's claimed about. Decide, using ONLY the reference "
            "material below -- not your own recollection -- whether the claim is accurate.\n"
            "There are two different ways the reference can fail to support a claim, and they call for "
            "different verdicts. (1) The reference is silent: it's about something else, too sparse, or "
            "doesn't describe the subject's capabilities or purpose at all -- this is a failed "
            "verification attempt, not a contradiction, so set verified_correct to true (the claim "
            "passes, unconfirmed) and note in issues that it could not be checked. (2) The reference "
            "DOES describe what the subject actually does or is for -- and that description doesn't "
            "include the claimed capability. That is a real contradiction even without an explicit "
            "denial: a description of what something does and is for implies it doesn't also do an "
            "unlisted, unrelated thing. In that case set verified_correct to false. "
            "Being unable to verify a claim is never grounds by itself to treat it as wrong -- only "
            "case (2), an actually described but different capability, is. "
            'Respond with JSON: {"verified_correct": bool, "issues": [str]}.'
        )
        verify_user = (
            f"Claim: {extraction['claim']}\n\n"
            f"Reference material about {extraction['subject']!r}:\n{reference}"
        )
        verdict = await self.call_chat(
            verify_system, verify_user, ["verified_correct", "issues"], temperature=0.0)
        return {"capability_claim_found": True, "verified_correct": verdict["verified_correct"],
                "issues": verdict["issues"]}

    async def _check_comparative_claims(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated check for claims comparing two things along some dimension (e.g.
        "deepslate diamond ore is more common than the regular variant" -- false, both
        yield the same amount; the regularity is about the depth range, not the ore
        block itself). A third claim shape distinct from mechanism ("X acts on Y") and
        capability ("X can do Y"): confirmed live, reviewing a broadened diagnostic,
        as a recurring fabrication pattern the other two checks structurally can't see.

        Same converged design as the other two checks: the model only extracts the two
        things being compared plus a one-sentence restatement of the claim (no tool
        choice), the code fetches wiki pages directly, and a plain call verifies with
        the same silent-vs-actually-describes-something-different distinction (see
        project memory for why that distinction has to be explicit, not implied).

        The comparison target is often left implicit in practice ("deepslate diamond
        ore... is more common" never names what it's more common *than* in the same
        clause) -- extraction may legitimately return an empty ``compared_to``.
        ``subject``'s page is always fetched; ``compared_to``'s page is fetched too
        only when it names something specific, best-effort like the mechanism check's
        actor fetch (only ``subject`` succeeding is a hard precondition).

        Returns a dict with ``comparative_claim_found`` (bool), ``verified_correct``
        (bool, meaningless if no claim was found), and ``issues`` (list of str).
        """
        if not _COMPARATIVE_CLAIM_PATTERN.search(candidate["answer"]):
            return {"comparative_claim_found": False, "verified_correct": True, "issues": []}

        extract_system = (
            "Read this Minecraft answer and look for a specific kind of claim: comparing two things "
            "along some dimension -- which is more common, more durable, faster, stronger, more "
            "valuable, and similar (e.g. 'deepslate diamond ore is more common than regular diamond "
            "ore', 'a diamond sword is stronger than an iron sword'). Ignore every other kind of "
            "claim -- a single fact stated on its own with nothing being compared to it is not this "
            "check's job. If the answer contains no such claim, respond with {\"claim_found\": false, "
            '"subject": "", "compared_to": "", "claim": ""}. If it does, respond with JSON: '
            '{"claim_found": bool, "subject": str, "compared_to": str, "claim": str}, where '
            "``subject`` is the thing the claim is actually about (e.g. \"Deepslate Diamond Ore\"), "
            "``compared_to`` is the specific thing it's being compared against if one is actually "
            'named (e.g. "Diamond Ore") -- leave this "" if the answer only implies a comparison '
            "without naming a specific target -- and ``claim`` is a one-sentence restatement of the "
            "specific comparison made."
        )
        extract_user = f"Answer: {candidate['answer']}"
        extraction = await self.call_chat(
            extract_system, extract_user, ["claim_found", "subject", "compared_to", "claim"],
            temperature=0.0)
        if not extraction["claim_found"] or not extraction.get("subject"):
            return {"comparative_claim_found": False, "verified_correct": True, "issues": []}

        reference = await fetch_minecraft_wiki_page(extraction["subject"])
        if reference.startswith(("Error:", "No wiki page found", "No content found")):
            return {"comparative_claim_found": True, "verified_correct": True,
                    "issues": [f"Could not verify (wiki lookup for {extraction['subject']!r} failed): "
                               f"{reference}"]}

        compared_reference = "(no specific comparison target was named)"
        if extraction.get("compared_to"):
            compared_reference = await fetch_minecraft_wiki_page(extraction["compared_to"])
            if compared_reference.startswith(("Error:", "No wiki page found", "No content found")):
                compared_reference = "(unavailable)"

        verify_system = (
            "You are given a claimed Minecraft comparison and reference material about the thing the "
            "claim is about, and (if named) the thing it's compared against. Decide, using ONLY the "
            "reference material below -- not your own recollection -- whether the claim is accurate.\n"
            "There are two different ways the reference can fail to support a claim, and they call for "
            "different verdicts. (1) The reference is silent: it's about something else, too sparse, "
            "or doesn't describe the specific property being compared at all -- this is a failed "
            "verification attempt, not a contradiction, so set verified_correct to true (the claim "
            "passes, unconfirmed) and note in issues that it could not be checked. (2) The reference "
            "DOES describe the actual property being compared -- e.g. it states both things have the "
            "same value, or states the opposite relationship -- and that contradicts the claim. That "
            "is a real contradiction even without an explicit denial. In that case set verified_correct "
            "to false. Being unable to verify a claim is never grounds by itself to treat it as wrong "
            "-- only case (2), an actually described but different relationship, is. "
            'Respond with JSON: {"verified_correct": bool, "issues": [str]}.'
        )
        verify_user = (
            f"Claim: {extraction['claim']}\n\n"
            f"Reference material about {extraction['subject']!r}:\n{reference}\n\n"
            f"Reference material about {extraction['compared_to'] or '(not named)'!r}:\n{compared_reference}"
        )
        verdict = await self.call_chat(
            verify_system, verify_user, ["verified_correct", "issues"], temperature=0.0)
        return {"comparative_claim_found": True, "verified_correct": verdict["verified_correct"],
                "issues": verdict["issues"]}

    async def _adversarial_critique(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Run an ensemble of independent adversarial-critique votes and majority-decide.

        A single low-temperature critique call proved unreliable in practice: on this
        model, tuning the prompt alone traded false rejections against false
        admits with no wording found that avoided both. Sampling several
        independent votes at a non-zero temperature and taking the majority
        filters one-off fabricated critiques (which don't reliably repeat across
        independent calls) without needing the prompt itself to be perfectly
        calibrated. Returns the same ``critique_passed``/``issues`` shape as a
        single vote; on a majority rejection, ``issues`` is the union of the
        rejecting votes' stated issues.
        """
        votes = await asyncio.gather(*[
            self._cast_critique_vote(candidate) for _ in range(_CRITIQUE_VOTES)
        ])
        reject_votes = [v for v in votes if not v["critique_passed"]]
        if len(reject_votes) * 2 <= len(votes):
            return {"critique_passed": True, "issues": []}
        issues = [issue for vote in reject_votes for issue in vote["issues"]]
        return {"critique_passed": False, "issues": issues}

    @staticmethod
    def _answer_quality_gate(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mechanical, LLM-free reject of a degenerate free-recall answer.

        The LLM judge admitted a literal ``"0"`` as the answer to an open
        strategy question; a cheap structural gate catches that class before
        any model call. Deliberately near-zero false-positive: it only rejects
        answers with essentially no linguistic content (fewer than 3 alphabetic
        characters, or fewer than 2 word-tokens containing a letter), so a
        short-but-real prose answer still passes. Returns a reject verdict, or
        ``None`` to continue.
        """
        answer = candidate.get("answer")
        if not isinstance(answer, str):
            return {"success": False, "errors": [f"answer is not a string: {answer!r}"],
                    "failure_kind": "degenerate_answer"}
        stripped = answer.strip()
        alpha = sum(c.isalpha() for c in stripped)
        word_tokens = [t for t in stripped.split() if any(c.isalpha() for c in t)]
        if alpha < 3 or len(word_tokens) < 2:
            return {"success": False,
                    "errors": [f"degenerate answer (not a substantive response): {stripped!r}"],
                    "failure_kind": "degenerate_answer"}
        return None

    async def judge(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a free-recall ``candidate`` and return a verdict dict.

        Two mechanical, LLM-free gates run first: the answer-quality gate
        (catches degenerate answers) and the numeric-claim check (tick<->time
        arithmetic, and enchantment max-levels when ground truth is set). Then
        the consistency check short-circuits on failure (saving further LLM
        calls), then the adversarial critique, then the three dedicated
        claim-shape checks (mechanism, capability, comparative). Returns the same
        ``{"success": bool, "errors": list[str], "failure_kind": Optional[str]}``
        shape as the static oracle, with ``failure_kind`` set to
        ``"degenerate_answer"``, ``"numeric_inconsistent"``,
        ``"logical_inconsistency"``, ``"adversarial_critique_failed"``,
        ``"mechanism_claim_incorrect"``, ``"capability_claim_incorrect"``, or
        ``"comparative_claim_incorrect"`` on rejection, otherwise ``None``.
        """
        degenerate = self._answer_quality_gate(candidate)
        if degenerate is not None:
            return degenerate
        numeric = check_numeric_claims(candidate, self.enchantment_max_levels)
        if not numeric["success"]:
            return numeric
        consistency = await self._check_consistency(candidate)
        if not consistency["consistent"]:
            return {"success": False, "errors": consistency["issues"] or ["reasoning inconsistent with answer"],
                    "failure_kind": "logical_inconsistency"}
        critique = await self._adversarial_critique(candidate)
        if not critique["critique_passed"]:
            return {"success": False, "errors": critique["issues"] or ["adversarial critique failed"],
                    "failure_kind": "adversarial_critique_failed"}
        mechanism = await self._check_mechanism_claims(candidate)
        if mechanism["mechanism_claim_found"] and not mechanism["verified_correct"]:
            return {"success": False, "errors": mechanism["issues"] or ["mechanism claim failed verification"],
                    "failure_kind": "mechanism_claim_incorrect"}
        capability = await self._check_capability_claims(candidate)
        if capability["capability_claim_found"] and not capability["verified_correct"]:
            return {"success": False, "errors": capability["issues"] or ["capability claim failed verification"],
                    "failure_kind": "capability_claim_incorrect"}
        comparative = await self._check_comparative_claims(candidate)
        if comparative["comparative_claim_found"] and not comparative["verified_correct"]:
            return {"success": False, "errors": comparative["issues"] or ["comparative claim failed verification"],
                    "failure_kind": "comparative_claim_incorrect"}
        return {"success": True, "errors": [], "failure_kind": None}
