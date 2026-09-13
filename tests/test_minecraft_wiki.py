import asyncio
import json
from unittest.mock import patch

from src.tools.minecraft_wiki import execute_minecraft_wiki_tool, fetch_minecraft_wiki_page


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


class _FakeSession:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, *args, **kwargs):
        return self._response


def _wiki_payload(page_id, title, extract):
    return {"query": {"pages": {page_id: {"pageid": int(page_id), "title": title, "extract": extract}}}}


def test_fetch_minecraft_wiki_page_returns_extract():
    response = _FakeResponse(200, _wiki_payload("123", "Diamond Ore", "Diamond ore generates from Y=16 to Y=-63."))

    async def _run():
        with patch("src.tools.minecraft_wiki.aiohttp.ClientSession", return_value=_FakeSession(response)):
            return await fetch_minecraft_wiki_page("Diamond Ore")

    result = asyncio.run(_run())
    assert "Y=16 to Y=-63" in result


def test_fetch_minecraft_wiki_page_truncates_to_max_chars():
    long_text = "x" * 10000
    response = _FakeResponse(200, _wiki_payload("123", "Long Page", long_text))

    async def _run():
        with patch("src.tools.minecraft_wiki.aiohttp.ClientSession", return_value=_FakeSession(response)):
            return await fetch_minecraft_wiki_page("Long Page", max_chars=500)

    result = asyncio.run(_run())
    assert len(result) <= 500


def test_fetch_minecraft_wiki_page_missing_page_returns_message():
    response = _FakeResponse(200, {"query": {"pages": {"-1": {"missing": "", "title": "Nonexistent Page"}}}})

    async def _run():
        with patch("src.tools.minecraft_wiki.aiohttp.ClientSession", return_value=_FakeSession(response)):
            return await fetch_minecraft_wiki_page("Nonexistent Page")

    result = asyncio.run(_run())
    assert "no wiki page found" in result.lower()


def test_fetch_minecraft_wiki_page_http_error_returns_message_not_raise():
    response = _FakeResponse(500, {})

    async def _run():
        with patch("src.tools.minecraft_wiki.aiohttp.ClientSession", return_value=_FakeSession(response)):
            return await fetch_minecraft_wiki_page("Diamond Ore")

    result = asyncio.run(_run())
    assert "error" in result.lower()


def test_execute_minecraft_wiki_tool_calls_fetch_with_page_title():
    response = _FakeResponse(200, _wiki_payload("123", "Diamond Ore", "Diamond ore generates from Y=16 to Y=-63."))

    async def _run():
        with patch("src.tools.minecraft_wiki.aiohttp.ClientSession", return_value=_FakeSession(response)):
            return await execute_minecraft_wiki_tool("search_minecraft_wiki", {"page_title": "Diamond Ore"})

    result = asyncio.run(_run())
    assert "Y=16 to Y=-63" in result


def test_execute_minecraft_wiki_tool_rejects_unknown_tool_name():
    async def _run():
        return await execute_minecraft_wiki_tool("some_other_tool", {})

    result = asyncio.run(_run())
    assert "unknown tool" in result.lower()


def test_execute_minecraft_wiki_tool_requires_page_title_argument():
    async def _run():
        return await execute_minecraft_wiki_tool("search_minecraft_wiki", {})

    result = asyncio.run(_run())
    assert "page_title" in result.lower()
