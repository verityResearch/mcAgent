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
    "instruction": "What's the fastest early-game route to obsidian in Bedrock Edition?",
    "reasoning": "Water source plus a lava pool creates obsidian on contact.",
    "answer": "Pour water on a lava source block.",
}


def test_judge_admits_when_consistent_and_critique_passes():
    consistency = _FakeResponse(200, _chat_payload({"consistent": True, "issues": []}))
    critique = _FakeResponse(200, _chat_payload({"critique_passed": True, "issues": []}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession",
                    side_effect=[_FakeSession(consistency), _FakeSession(critique)]):
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
        with patch("src.llm_client.aiohttp.ClientSession",
                    side_effect=[_FakeSession(consistency), _FakeSession(critique)]):
            judge = Judge(api_key="test-key")
            return await judge.judge(_CANDIDATE)

    result = asyncio.run(_run())
    assert result["success"] is False
    assert result["failure_kind"] == "adversarial_critique_failed"
