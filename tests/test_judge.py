import asyncio
import json
from unittest.mock import patch

from src.verification.judge import Judge


class _FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)


class _FakeSession:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        return self._response


def _chat_payload(content_obj):
    return {"choices": [{"message": {"content": json.dumps(content_obj)}}]}


_CANDIDATE = {
    "instruction": "What's the fastest early-game route to obsidian?",
    "reasoning": "Water source plus a lava pool creates obsidian on contact.",
    "answer": "Pour water on a lava source block.",
}


def _no_mechanism_claim_session():
    response = _FakeResponse(
        200, _chat_payload({"claim_found": False, "actor": "", "acted_upon": "", "claim": ""}))
    return _FakeSession(response)


def test_judge_admits_when_consistent_and_critique_passes():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        # _CANDIDATE's answer has no mechanism verb, so the pre-filter skips the
        # mechanism check's LLM call entirely -- no extra session needed here.
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_rejects_on_inconsistency_without_calling_critique():
    consistency = _FakeResponse(200, _chat_payload({"consistent": False, "issues": ["answer contradicts reasoning"]}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession",
                    side_effect=[_FakeSession(consistency)]):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "logical_inconsistency"
    assert "answer contradicts reasoning" in result["errors"]


def test_judge_rejects_on_failed_critique():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": False, "issues": ["unverifiable claim"]}))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "adversarial_critique_failed"


def test_judge_critique_majority_passes_despite_one_dissenting_reject():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    pass_vote = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    reject_vote = _FakeResponse(200, _chat_payload({"critique_passed": False, "issues": ["one dissenting critic"]}))

    async def _run():
        # _CANDIDATE's answer has no mechanism verb, so the pre-filter skips the
        # mechanism check's LLM call entirely -- no extra session needed here.
        sessions = [_FakeSession(consistency), _FakeSession(pass_vote),
                    _FakeSession(pass_vote), _FakeSession(reject_vote)]
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_critique_majority_rejects_and_collects_dissenting_issues():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    pass_vote = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    reject_vote_a = _FakeResponse(200, _chat_payload({"critique_passed": False, "issues": ["issue A"]}))
    reject_vote_b = _FakeResponse(200, _chat_payload({"critique_passed": False, "issues": ["issue B"]}))

    async def _run():
        sessions = [_FakeSession(consistency), _FakeSession(pass_vote),
                    _FakeSession(reject_vote_a), _FakeSession(reject_vote_b)]
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "adversarial_critique_failed"
    assert "issue A" in result["errors"]
    assert "issue B" in result["errors"]


def test_judge_admits_when_extraction_finds_no_mechanism_claim():
    """A mechanism verb is present (passes the pre-filter) but the extraction call
    itself legitimately finds no actor/acted_upon pair -- a different case from the
    pre-filter's own skip above, since here the LLM call does happen."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3 + [_no_mechanism_claim_session()]
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_MECHANISM_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


_MECHANISM_CANDIDATE = {
    "instruction": "How do I build a basic automatic door?",
    "reasoning": "A redstone torch inverts a signal to drive a piston.",
    "answer": "Place a redstone torch on the wall, attach a piston to the door, and the piston "
              "opens the door when the torch is powered.",
}


def test_judge_mechanism_prefilter_skips_llm_call_when_no_action_verb_present():
    """_CANDIDATE's answer names no block/item mechanism verb, so the cheap local
    pre-filter should skip the LLM-based mechanism check entirely -- confirmed by
    only providing sessions for consistency + the 3 critique votes; if the check
    fell through to an LLM call anyway, side_effect would raise StopIteration."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_rejects_on_incorrect_mechanism_claim():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "actor": "Piston", "acted_upon": "Door",
        "claim": "a piston opens and closes the door",
    }))
    verify = _FakeResponse(200, _chat_payload({
        "verified_correct": False,
        "issues": ["doors are toggled directly by redstone signal, not by a piston"],
    }))

    fetched_titles = []

    async def fake_fetch(page_title, *args, **kwargs):
        fetched_titles.append(page_title)
        if page_title == "Door":
            return "A door is toggled directly by a redstone signal."
        if page_title == "Piston":
            return "A piston pushes entities and blocks when given a redstone signal."
        raise AssertionError(f"unexpected page title: {page_title!r}")

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page", side_effect=fake_fetch):
            judge = Judge(api_key="test-key")
            return await judge.judge(_MECHANISM_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "mechanism_claim_incorrect"
    assert "doors are toggled directly by redstone signal, not by a piston" in result["errors"]
    # Both the acted-on page and the actor's own page must be fetched -- fetching
    # only "Door" was found, via live testing, to miss real disambiguating signal
    # that only "Piston"'s own page states (see _check_mechanism_claims docstring).
    assert fetched_titles == ["Door", "Piston"]


def test_judge_admits_when_mechanism_claim_found_but_verified_correct():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "actor": "Redstone Torch", "acted_upon": "Rail",
        "claim": "a redstone torch powers an adjacent powered rail",
    }))
    verify = _FakeResponse(200, _chat_payload({"verified_correct": True, "issues": []}))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="A powered rail accelerates minecarts when powered by redstone."):
            judge = Judge(api_key="test-key")
            return await judge.judge(_MECHANISM_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_still_verifies_when_only_actor_page_fetch_fails():
    """The acted_upon fetch succeeding is the only hard precondition -- a failed
    actor fetch degrades to verifying with less context, not to skipping the check
    the way a failed acted_upon fetch does."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "actor": "Piston", "acted_upon": "Door",
        "claim": "a piston opens and closes the door",
    }))
    verify = _FakeResponse(200, _chat_payload({
        "verified_correct": False,
        "issues": ["doors are toggled directly by redstone signal, not by a piston"],
    }))

    async def fake_fetch(page_title, *args, **kwargs):
        if page_title == "Door":
            return "A door is toggled directly by a redstone signal."
        return "Error: Minecraft Wiki lookup failed: 500, message: server error"

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page", side_effect=fake_fetch):
            judge = Judge(api_key="test-key")
            return await judge.judge(_MECHANISM_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "mechanism_claim_incorrect"


def test_judge_admits_when_wiki_lookup_fails_instead_of_rejecting():
    """A failed lookup (network error, malformed title, whatever) is uncertainty about
    the claim, not evidence against it -- observed live: a compound extracted title
    ("Lever or button") tripped a wiki-side response mlx couldn't decode, and the old
    behavior handed that error string to the verify call as if it were real reference
    material, which read as "unaddressed claim" and defaulted to rejecting. Only one
    extraction session is needed here since a failed fetch skips the verify call
    entirely -- if it fell through anyway, side_effect would raise StopIteration."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "actor": "Redstone Torch", "acted_upon": "Lever or button",
        "claim": "a redstone torch powers a lever or button",
    }))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3 + [_FakeSession(extraction)]
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="Error: Minecraft Wiki lookup failed: 400, message:\n  Can not decode "
                                 "content-encoding: br"):
            judge = Judge(api_key="test-key")
            return await judge.judge(_MECHANISM_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


# --- _check_capability_claims: a block/item having a capability or purpose,
# distinct from _check_mechanism_claims' "one block acts on another." Found live
# reviewing a broadened diagnostic: "droppers sort items by type" and "a locator
# map finds a shipwreck" both slipped through uncaught because neither claims one
# block acting on a second one, so the mechanism check structurally can't see them.

_CAPABILITY_CANDIDATE = {
    "instruction": "How do I build an efficient item storage and sorting system?",
    "reasoning": "Droppers eject items on a redstone pulse.",
    "answer": "Droppers can be used to sort items into specific chests based on their type, "
              "creating a fully automated and organized system.",
}


def _no_capability_claim_session():
    response = _FakeResponse(200, _chat_payload({"claim_found": False, "subject": "", "claim": ""}))
    return _FakeSession(response)


def test_judge_capability_prefilter_skips_llm_call_when_no_capability_phrase_present():
    """_CANDIDATE's answer has no capability-claim phrase, so the pre-filter should
    skip the LLM call entirely -- confirmed by only providing sessions for
    consistency + the 3 critique votes; if it fell through anyway, side_effect
    would raise StopIteration."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_rejects_on_incorrect_capability_claim():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Dropper",
        "claim": "droppers sort items into specific chests based on their type",
    }))
    verify = _FakeResponse(200, _chat_payload({
        "verified_correct": False,
        "issues": ["droppers only eject items on a redstone pulse; they have no type-sorting logic"],
    }))

    fetched_titles = []

    async def fake_fetch(page_title, *args, **kwargs):
        fetched_titles.append(page_title)
        return "A dropper is a low-capacity storage block that can eject its contents into the " \
               "world or into other containers when given a redstone signal."

    async def _run():
        # _CAPABILITY_CANDIDATE's answer has no mechanism verb, so the mechanism
        # check's pre-filter skips its own LLM call entirely -- no session needed
        # for it here, only for the capability-check calls.
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page", side_effect=fake_fetch):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CAPABILITY_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "capability_claim_incorrect"
    assert "droppers only eject items on a redstone pulse; they have no type-sorting logic" \
        in result["errors"]
    assert fetched_titles == ["Dropper"]


def test_judge_admits_when_capability_claim_found_but_verified_correct():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Hopper",
        "claim": "hoppers can be used to move items between containers automatically",
    }))
    verify = _FakeResponse(200, _chat_payload({"verified_correct": True, "issues": []}))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="A hopper is used to move items into and out of containers."):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CAPABILITY_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_admits_when_capability_extraction_finds_no_claim():
    """A capability-pattern phrase is present (passes the pre-filter) but the
    extraction call itself legitimately finds no real capability claim."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_no_capability_claim_session()])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CAPABILITY_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_admits_when_capability_wiki_lookup_fails_instead_of_rejecting():
    """Same calibration principle as the mechanism check's equivalent test: a failed
    lookup is uncertainty about the claim, not evidence against it."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Dropper",
        "claim": "droppers sort items into specific chests based on their type",
    }))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="No wiki page found for 'Dropper'."):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CAPABILITY_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


# --- _check_comparative_claims: two things compared along some dimension, a
# third claim shape distinct from mechanism ("X acts on Y") and capability
# ("X can do Y"). Found live reviewing a broadened diagnostic: "deepslate diamond
# ore is more common ... than regular diamond ore" slipped through uncaught --
# both variants yield the same amount; the depth range is what's more/less common.

_COMPARATIVE_CANDIDATE = {
    "instruction": "How do I strip-mine efficiently for diamonds?",
    "reasoning": "Diamond ore generates in a range of depths, mostly as deepslate near the bottom.",
    "answer": "Prioritize deepslate diamond ore, which is more common and more abundant than "
              "regular diamond ore, as you dig near the bottom of the world.",
}


def _no_comparative_claim_session():
    response = _FakeResponse(
        200, _chat_payload({"claim_found": False, "subject": "", "compared_to": "", "claim": ""}))
    return _FakeSession(response)


def test_judge_comparative_prefilter_skips_llm_call_when_no_comparative_phrase_present():
    """_CANDIDATE's answer has no comparative phrase, so the pre-filter should skip
    the LLM call entirely -- confirmed by only providing sessions for consistency +
    the 3 critique votes; if it fell through anyway, side_effect would raise
    StopIteration."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = [_FakeSession(consistency)] + [_FakeSession(critique)] * 3
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_rejects_on_incorrect_comparative_claim():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Deepslate Diamond Ore", "compared_to": "Diamond Ore",
        "claim": "deepslate diamond ore is more common and yields more diamonds than regular diamond ore",
    }))
    verify = _FakeResponse(200, _chat_payload({
        "verified_correct": False,
        "issues": ["both variants drop the same amount; only the generation depth differs, not the ore itself"],
    }))

    fetched_titles = []

    async def fake_fetch(page_title, *args, **kwargs):
        fetched_titles.append(page_title)
        if page_title == "Deepslate Diamond Ore":
            return "Deepslate diamond ore is a variant of diamond ore found below Y=0. It drops 1 " \
                   "diamond, the same as regular diamond ore, when mined without Silk Touch."
        if page_title == "Diamond Ore":
            return "Diamond ore drops 1 diamond when mined without Silk Touch."
        raise AssertionError(f"unexpected page title: {page_title!r}")

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page", side_effect=fake_fetch):
            judge = Judge(api_key="test-key")
            return await judge.judge(_COMPARATIVE_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "comparative_claim_incorrect"
    assert "both variants drop the same amount; only the generation depth differs, not the ore itself" \
        in result["errors"]
    assert fetched_titles == ["Deepslate Diamond Ore", "Diamond Ore"]


def test_judge_admits_when_comparative_claim_found_but_verified_correct():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Netherite Sword", "compared_to": "Diamond Sword",
        "claim": "a netherite sword deals more damage than a diamond sword",
    }))
    verify = _FakeResponse(200, _chat_payload({"verified_correct": True, "issues": []}))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="A netherite sword deals more attack damage than a diamond sword."):
            judge = Judge(api_key="test-key")
            return await judge.judge(_COMPARATIVE_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_admits_when_comparative_extraction_finds_no_claim():
    """A comparative-pattern phrase is present (passes the pre-filter) but the
    extraction call itself legitimately finds no real comparison being made."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_no_comparative_claim_session()])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions):
            judge = Judge(api_key="test-key")
            return await judge.judge(_COMPARATIVE_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_admits_comparative_claim_with_no_named_comparison_target():
    """extraction may legitimately leave compared_to empty when the answer implies
    a comparison without naming a specific target -- only subject's page should be
    fetched in that case (confirmed by the session list length: one fetch call, one
    verify call, and fetch_minecraft_wiki_page is only patched to accept "Deepslate
    Diamond Ore" -- any other title raises)."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Deepslate Diamond Ore", "compared_to": "",
        "claim": "deepslate diamond ore is more common",
    }))
    verify = _FakeResponse(200, _chat_payload({"verified_correct": True, "issues": []}))

    async def fake_fetch(page_title, *args, **kwargs):
        if page_title == "Deepslate Diamond Ore":
            return "Deepslate diamond ore generates below Y=0."
        raise AssertionError(f"unexpected page title: {page_title!r}")

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction), _FakeSession(verify)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page", side_effect=fake_fetch):
            judge = Judge(api_key="test-key")
            return await judge.judge(_COMPARATIVE_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True


def test_judge_admits_when_comparative_wiki_lookup_fails_instead_of_rejecting():
    """Same calibration principle as the other two checks' equivalent tests: a
    failed lookup is uncertainty about the claim, not evidence against it."""
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))
    extraction = _FakeResponse(200, _chat_payload({
        "claim_found": True, "subject": "Deepslate Diamond Ore", "compared_to": "Diamond Ore",
        "claim": "deepslate diamond ore is more common than regular diamond ore",
    }))

    async def _run():
        sessions = ([_FakeSession(consistency)] + [_FakeSession(critique)] * 3
                    + [_FakeSession(extraction)])
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=sessions), \
             patch("src.verification.judge.fetch_minecraft_wiki_page",
                   return_value="No wiki page found for 'Deepslate Diamond Ore'."):
            judge = Judge(api_key="test-key")
            return await judge.judge(_COMPARATIVE_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is True
