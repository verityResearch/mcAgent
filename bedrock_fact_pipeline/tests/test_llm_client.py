import asyncio
import json
from unittest.mock import patch

import pytest

from src.llm_client import ChatClient


class _CapturingSession:
    """Records the JSON payload of every POST for assertions on request shape."""

    def __init__(self, response):
        self._response = response
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        self.payloads.append(kwargs.get("json"))
        return self._response


class _TimeoutResponse:
    async def __aenter__(self):
        raise asyncio.TimeoutError()

    async def __aexit__(self, *exc):
        return False


class _TimeoutSession:
    """Simulates a server that never responds -- ``post`` returns something whose
    ``__aenter__`` raises, matching aiohttp's shape when a request times out."""

    def __init__(self):
        self.post_call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        self.post_call_count += 1
        return _TimeoutResponse()


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


def test_call_chat_sends_a_generous_default_max_tokens():
    session = _CapturingSession(_FakeResponse(200, _chat_payload({"a": 1})))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a"])

    asyncio.run(_run())
    assert session.payloads[0]["max_tokens"] >= 1024


def test_call_chat_with_tools_sends_a_generous_default_max_tokens():
    session = _CapturingSession(_FakeResponse(200, _chat_payload({"answer": "x"})))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")

            async def executor(name, args):
                raise AssertionError("not expected")

            await client.call_chat_with_tools(
                "system", "user", ["answer"], tools=[{"type": "function", "function": {"name": "noop"}}],
                tool_executor=executor)

    asyncio.run(_run())
    assert session.payloads[0]["max_tokens"] >= 1024


def test_call_chat_returns_parsed_json():
    response = _FakeResponse(200, _chat_payload({"a": 1, "b": 2}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")
            return await client.call_chat("system", "user", ["a", "b"])

    assert asyncio.run(_run()) == {"a": 1, "b": 2}


def test_call_chat_raises_on_missing_keys():
    response = _FakeResponse(200, _chat_payload({"a": 1}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a", "b"])

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


def test_call_chat_wraps_timeout_in_runtime_error_without_retrying():
    session = _TimeoutSession()

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a"])

    with pytest.raises(RuntimeError, match="did not complete within"):
        asyncio.run(_run())
    # A hung server isn't fixed by retrying inside the same short window -- unlike
    # the JSON-decode-corruption retry, a timeout should fail after one attempt.
    assert session.post_call_count == 1


def test_call_chat_raises_on_http_error():
    response = _FakeResponse(500, {})

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a"])

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


def test_call_chat_raises_on_non_json_content():
    response = _FakeResponse(200, {"choices": [{"message": {"content": "not json"}}]})

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a"])

    with pytest.raises(RuntimeError):
        asyncio.run(_run())


def test_call_chat_retries_once_on_bad_json_then_succeeds():
    """The existing test_call_chat_raises_on_non_json_content only proves 'always
    bad JSON eventually raises', which is true whether or not the retry loop
    exists at all. This proves the actual value of the retry: one bad-JSON
    response followed by a good one succeeds, using exactly 2 POST calls."""
    responses = [
        _FakeResponse(200, {"choices": [{"message": {"content": "not json"}}]}),
        _FakeResponse(200, _chat_payload({"a": 1})),
    ]

    class _SequentialSession:
        def __init__(self, responses):
            self._responses = list(responses)
            self.post_call_count = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            self.post_call_count += 1
            return self._responses.pop(0)

    session = _SequentialSession(responses)

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            return await client.call_chat("system", "user", ["a"])

    result = asyncio.run(_run())
    assert result == {"a": 1}
    assert session.post_call_count == 2


def _tool_call_payload(name, arguments, call_id="call-1"):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {"name": name, "arguments": json.dumps(arguments)}}],
            }
        }]
    }


def test_call_chat_with_tools_returns_direct_answer_when_no_tool_call_requested():
    response = _FakeResponse(200, _chat_payload({"answer": "42"}))

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")

            async def executor(name, args):
                raise AssertionError("tool executor should not be called")

            return await client.call_chat_with_tools(
                "system", "user", ["answer"], tools=[{"type": "function", "function": {"name": "noop"}}],
                tool_executor=executor)

    assert asyncio.run(_run()) == {"answer": "42"}


def test_call_chat_with_tools_executes_tool_then_returns_final_answer():
    responses = [
        _FakeResponse(200, _tool_call_payload("search_minecraft_wiki", {"page_title": "Diamond Ore"}, call_id="call-1")),
        _FakeResponse(200, _chat_payload({"answer": "Y=-58 to Y=-64"})),
    ]

    class _SequentialSession:
        """Like _CapturingSession, but returns each response in order rather than
        the same one every time -- records every POST payload too, so the round-2
        request's message-history bookkeeping (assistant turn with tool_calls, then
        the matching tool-role response) can be asserted, not just the final result.

        call_chat_with_tools builds one `messages` list and appends to it in place
        across rounds (never replacing or mutating an already-appended entry), so a
        shallow copy of the list at capture time is enough to freeze what was
        actually sent for *this* round -- storing the bare reference would mean
        every captured payload ends up pointing at the same, fully-mutated final
        list once the whole call completes.
        """

        def __init__(self, responses):
            self._responses = list(responses)
            self.payloads = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            payload = kwargs.get("json")
            self.payloads.append({**payload, "messages": list(payload["messages"])})
            return self._responses.pop(0)

    calls_seen = []

    async def executor(name, args):
        calls_seen.append((name, args))
        return "Diamond ore generates from Y=16 to Y=-64."

    session = _SequentialSession(responses)

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            return await client.call_chat_with_tools(
                "system", "user", ["answer"],
                tools=[{"type": "function", "function": {"name": "search_minecraft_wiki"}}],
                tool_executor=executor)

    result = asyncio.run(_run())
    assert result == {"answer": "Y=-58 to Y=-64"}
    assert calls_seen == [("search_minecraft_wiki", {"page_title": "Diamond Ore"})]

    assert len(session.payloads) == 2
    round1_messages = session.payloads[0]["messages"]
    assert round1_messages == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]

    round2_messages = session.payloads[1]["messages"]
    assert round2_messages[:2] == round1_messages
    assistant_turn = round2_messages[2]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["tool_calls"][0]["id"] == "call-1"
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "search_minecraft_wiki"
    tool_turn = round2_messages[3]
    assert tool_turn == {"role": "tool", "tool_call_id": "call-1",
                          "content": "Diamond ore generates from Y=16 to Y=-64."}
    assert len(round2_messages) == 4


class _CapturingSequentialSession:
    """Like the inline ``_SequentialSession`` in the tool-call test above, but
    reusable across the tests below: returns each response in order and
    snapshots every POST's ``messages`` list (a shallow copy, since
    ``call_chat_with_tools`` mutates its own ``messages`` list in place across
    rounds -- capturing the bare reference would mean every stored payload
    ends up pointing at the same, fully-mutated final list)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        payload = kwargs.get("json")
        self.payloads.append({**payload, "messages": list(payload["messages"])})
        return self._responses.pop(0)


def test_call_chat_tolerates_trailing_data_after_valid_json():
    """A server that doesn't strictly enforce response_format=json_object has
    been observed to emit a valid JSON object followed by extra trailing text
    (e.g. repeated commentary) in the same response. That must not be treated
    as a parse failure -- the intended object is still right there at the
    start of the content."""
    content = json.dumps({"a": 1, "b": 2}) + "\nHere is some extra unrequested commentary."
    response = _FakeResponse(200, {"choices": [{"message": {"content": content}}]})

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=_FakeSession(response)):
            client = ChatClient(api_key="test-key")
            return await client.call_chat("system", "user", ["a", "b"])

    assert asyncio.run(_run()) == {"a": 1, "b": 2}


def test_call_chat_sends_corrective_message_then_succeeds_on_prose_only_response():
    """The other observed failure mode: the model ignores response_format
    entirely and returns plain prose with no JSON at all. Unlike trailing-data
    corruption, reposting the identical request wouldn't help a model that
    deterministically ignores the instruction -- so the retry must append the
    bad reply plus a corrective instruction, not just try again blind."""
    responses = [
        _FakeResponse(200, {"choices": [{"message": {"content": "Sure! Here's my answer as plain text."}}]}),
        _FakeResponse(200, _chat_payload({"a": 1})),
    ]
    session = _CapturingSequentialSession(responses)

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            return await client.call_chat("system", "user", ["a"])

    result = asyncio.run(_run())
    assert result == {"a": 1}
    assert len(session.payloads) == 2

    retry_messages = session.payloads[1]["messages"]
    assert retry_messages[:2] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert retry_messages[2] == {"role": "assistant", "content": "Sure! Here's my answer as plain text."}
    assert retry_messages[3]["role"] == "user"
    assert "valid JSON" in retry_messages[3]["content"]
    assert len(retry_messages) == 4


def test_call_chat_raises_with_bounded_attempts_when_every_retry_stays_prose():
    """Exhausting all retries on a model that never produces JSON must still
    fail with a clear count -- not retry forever, and not silently succeed
    with prose mistaken for JSON."""
    session = _CapturingSequentialSession([
        _FakeResponse(200, {"choices": [{"message": {"content": "still not json"}}]}) for _ in range(5)
    ])

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            await client.call_chat("system", "user", ["a"])

    with pytest.raises(RuntimeError, match="non-JSON content"):
        asyncio.run(_run())
    assert len(session.payloads) == 3  # _MAX_TRANSIENT_RETRIES=2 extra attempts + the first


def test_call_chat_with_tools_corrective_retry_does_not_pollute_later_rounds():
    """A corrective retry inside one round must be local to that round's
    request -- once that round eventually produces a tool-calls response and
    the outer loop moves on to the next round, the next round must not see
    the earlier round's bad reply or correction. This guards the
    ``messages = messages + [...]`` design (a new list per retry) against a
    regression to in-place mutation of the shared conversation.

    Three POSTs happen here, all inside call_chat_with_tools's round 0: attempt
    1 ("not json yet") fails to parse and triggers an internal corrective
    retry; attempt 2 (with the correction appended) gets a tool-calls
    response, so _post_chat_round returns it immediately -- ending round 0.
    The outer loop then appends the real assistant/tool turns and starts
    round 1, which is the third POST."""
    responses = [
        _FakeResponse(200, {"choices": [{"message": {"content": "not json yet"}}]}),
        _FakeResponse(200, _tool_call_payload("search_minecraft_wiki", {"page_title": "Diamond Ore"})),
        _FakeResponse(200, _chat_payload({"answer": "final"})),
    ]
    session = _CapturingSequentialSession(responses)

    async def executor(name, args):
        return "some wiki text"

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", return_value=session):
            client = ChatClient(api_key="test-key")
            return await client.call_chat_with_tools(
                "system", "user", ["answer"],
                tools=[{"type": "function", "function": {"name": "search_minecraft_wiki"}}],
                tool_executor=executor)

    result = asyncio.run(_run())
    assert result == {"answer": "final"}
    assert len(session.payloads) == 3

    # payloads[1] is round 0's own corrective retry -- it SHOULD carry the
    # correction, that's the fix working as designed.
    assert session.payloads[1]["messages"][2] == {"role": "assistant", "content": "not json yet"}

    # payloads[2] is genuinely the next round (after a real tool-calls
    # response), and must start from the real assistant/tool turns only --
    # no leftover "not json yet" or corrective instruction from round 0.
    round1_messages = session.payloads[2]["messages"]
    assert round1_messages[:2] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert round1_messages[2]["role"] == "assistant"
    assert round1_messages[2]["content"] is None
    assert round1_messages[2]["tool_calls"][0]["function"]["name"] == "search_minecraft_wiki"
    assert round1_messages[3] == {"role": "tool", "tool_call_id": "call-1", "content": "some wiki text"}
    assert len(round1_messages) == 4


def test_call_chat_with_tools_raises_after_exceeding_max_rounds():
    call_payload = _tool_call_payload("search_minecraft_wiki", {"page_title": "Diamond Ore"})

    class _AlwaysToolCallSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *args, **kwargs):
            return _FakeResponse(200, call_payload)

    async def executor(name, args):
        return "some wiki text"

    async def _run():
        with patch("src.llm_client.aiohttp.ClientSession", side_effect=lambda: _AlwaysToolCallSession()):
            client = ChatClient(api_key="test-key")
            await client.call_chat_with_tools(
                "system", "user", ["answer"],
                tools=[{"type": "function", "function": {"name": "search_minecraft_wiki"}}],
                tool_executor=executor, max_tool_rounds=2)

    with pytest.raises(RuntimeError):
        asyncio.run(_run())
