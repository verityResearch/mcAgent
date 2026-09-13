import asyncio
import json
from unittest.mock import patch

import pytest

from src.generation.teacher import TeacherModel


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


def test_generate_fact_seeded_returns_parsed_candidate():
    response = _FakeResponse(200, _chat_payload({
        "instruction": "What do you need to craft an acacia boat?",
        "reasoning": "Acacia boats use planks in a shaped pattern.",
        "answer": "Acacia planks.",
        "structured_claim": {"result_item": "minecraft:acacia_boat"},
    }))
    fact = {"category": "recipe", "subject_id": "minecraft:acacia_boat",
            "fields": {"result_item": "minecraft:acacia_boat"}}

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            teacher = TeacherModel(api_key="test-key")
            return await teacher.generate_fact_seeded(fact, "What do you need to craft an acacia boat?")

    candidate = asyncio.run(_run())
    assert candidate["structured_claim"] == {"result_item": "minecraft:acacia_boat"}


def test_generate_fact_seeded_raises_on_missing_keys():
    response = _FakeResponse(200, _chat_payload({"instruction": "x"}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            teacher = TeacherModel(api_key="test-key")
            await teacher.generate_fact_seeded({"category": "recipe", "subject_id": "x", "fields": {}}, "q")

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


def test_generate_free_recall_returns_parsed_candidate():
    response = _FakeResponse(200, _chat_payload({
        "instruction": "What's the fastest early-game route to obsidian in Bedrock Edition?",
        "reasoning": "Water source plus lava pool, or a diamond pickaxe and a lava lake.",
        "answer": "Pour water on a lava source block, or mine it directly with a diamond pickaxe.",
    }))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            teacher = TeacherModel(api_key="test-key")
            return await teacher.generate_free_recall("Early-game strategy: fastest route to obsidian in Bedrock Edition")

    candidate = asyncio.run(_run())
    assert "structured_claim" not in candidate
    assert candidate["instruction"]


def test_generate_free_recall_raises_on_missing_keys():
    response = _FakeResponse(200, _chat_payload({"instruction": "x"}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            teacher = TeacherModel(api_key="test-key")
            await teacher.generate_free_recall("some topic")

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


def test_generate_fact_seeded_raises_on_http_error():
    response = _FakeResponse(500, {})

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            teacher = TeacherModel(api_key="test-key")
            await teacher.generate_fact_seeded({"category": "recipe", "subject_id": "x", "fields": {}}, "q")

    with pytest.raises(RuntimeError):
        asyncio.run(_run())
