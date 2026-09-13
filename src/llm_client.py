import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List

import aiohttp

# Concurrent requests against a local mlx_lm.server were observed to
# occasionally corrupt one response in a batch (empty or truncated content)
# while an identical retry against the same server succeeded immediately --
# see docs on the free-recall diagnostic. Retrying a few times absorbs that
# without masking a genuinely broken prompt/model (which fails the same way
# every time, retries included).
_MAX_TRANSIENT_RETRIES = 2

# A local single-instance server that stops responding under load was
# observed to stay unresponsive well after the load that caused it stopped
# (still failing a trivial 10-token request 2+ minutes later) -- a genuine
# wedge, not a queue that drains given more time. aiohttp's default total
# timeout (300s) would let that cost five minutes per attempt; failing
# faster surfaces the problem sooner without pretending a retry would help
# a server that is actually stuck (so, unlike JSON-decode corruption above,
# timeouts are not retried here -- they get one clearly-labeled failure).
_REQUEST_TIMEOUT_SECONDS = 60


class ChatAPITimeoutError(RuntimeError):
    """The chat API didn't respond within ``_REQUEST_TIMEOUT_SECONDS``.

    A distinct subclass (rather than a plain ``RuntimeError``, which every other
    failure in this module also raises) so a caller running many calls in a batch
    -- e.g. ``PipelineOrchestrator`` -- can tell "the server appears to be wedged"
    apart from "this one candidate was malformed" without parsing error text, and
    react differently (stop launching new work rather than retrying each item).
    """


def _parse_json_tolerating_trailing_data(content: str) -> Dict[str, Any]:
    """Parse ``content`` as a JSON object, tolerating trailing data after it.

    A server that doesn't strictly enforce ``response_format: json_object`` has
    been observed to occasionally emit a valid JSON object followed by extra
    text (e.g. a repeated answer or stray commentary) in the same response. A
    bare ``json.loads`` rejects the whole response over that trailing text even
    though the intended JSON object parsed fine; ``raw_decode`` reads only the
    first JSON value and reports where it stopped, so trailing text no longer
    fails an otherwise-valid response. ``lstrip`` matters because ``raw_decode``
    (unlike ``loads``/``decode``) does not skip leading whitespace on its own.
    """
    return json.JSONDecoder().raw_decode(content.lstrip())[0]


class ChatClient:
    """Shared OpenAI-compatible chat client returning parsed JSON responses.

    Subclassed by both the teacher (generation) and the judge (verification)
    so the HTTP call and JSON-envelope handling live in one place.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o", base_url: str = "https://api.openai.com/v1",
                 max_tokens: int = 3072):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        # Some servers (e.g. mlx_lm.server) default max_tokens to a small value
        # (512) when a request doesn't specify one, which silently truncates a
        # full instruction/reasoning/answer JSON response -- especially after a
        # tool-call round-trip adds context. Always send an explicit, generous
        # value rather than relying on the server's default.
        self.max_tokens = max_tokens

    async def _post_chat_round(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST one chat round and return the raw response message.

        A ``tool_calls`` response is returned as-is (no content to parse). A
        final-answer response whose content isn't valid JSON is retried up to
        ``_MAX_TRANSIENT_RETRIES`` extra times -- some servers don't reliably
        enforce ``response_format: json_object`` and instead return plain prose
        on a final turn. Each retry appends the bad reply plus a corrective
        instruction to the conversation before asking again, rather than
        reposting the identical request and hoping for a different transient
        result (which alone only helps the subset of failures that really are
        transient, e.g. valid JSON with stray trailing text -- see
        ``_parse_json_tolerating_trailing_data``). Raises ``RuntimeError`` on an
        HTTP error or if every attempt's content still fails to parse.
        """
        messages = payload["messages"]
        last_error: Any = None
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            if attempt:
                await asyncio.sleep(0.5 * attempt)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions", headers=headers,
                        json={**payload, "messages": messages},
                        timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"Chat API failed ({resp.status}): {text[:500]}")
                        data = await resp.json()
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                raise ChatAPITimeoutError(
                    f"Chat API request did not complete within {_REQUEST_TIMEOUT_SECONDS}s: {exc!r}"
                ) from exc
            message = data["choices"][0]["message"]
            if message.get("tool_calls"):
                return message
            content = message.get("content") or ""
            try:
                _parse_json_tolerating_trailing_data(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "That response was not a valid JSON object. "
                                                 "Respond again with ONLY the JSON object -- no other text."},
                ]
                continue
            return message
        raise RuntimeError(
            f"Chat API returned non-JSON content after {_MAX_TRANSIENT_RETRIES + 1} attempts: {last_error}"
        ) from last_error

    async def call_chat(self, system: str, user: str, required_keys: List[str],
                        temperature: float = 0.7) -> Dict[str, Any]:
        """Call the chat-completions endpoint and return the parsed JSON object.

        Requests a JSON-object response, parses the message content, and checks
        that every name in ``required_keys`` is present. Raises ``RuntimeError``
        on any HTTP error, non-JSON content, or missing required key.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        message = await self._post_chat_round(headers, payload)
        parsed = _parse_json_tolerating_trailing_data(message.get("content") or "")
        missing = [k for k in required_keys if k not in parsed]
        if missing:
            raise RuntimeError(f"Chat API response missing required keys: {missing}")
        return parsed

    async def call_chat_with_tools(
        self, system: str, user: str, required_keys: List[str], tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, Dict[str, Any]], Awaitable[str]],
        temperature: float = 0.7, max_tool_rounds: int = 12,
    ) -> Dict[str, Any]:
        """Like ``call_chat``, but lets the model request tool calls before answering.

        ``tool_executor(name, arguments)`` is awaited for each tool call the model
        requests, and its return value is fed back as that call's tool-response
        message. Loops until the model returns a direct JSON answer (parsed and
        validated exactly like ``call_chat``) or ``max_tool_rounds`` is exceeded,
        at which point it raises rather than silently returning an ungrounded
        answer. The default was raised from an earlier 6: the judge's adversarial
        critique prompt (see ``Judge._cast_critique_vote``) deliberately asks the
        model to verify every claim it has real doubt about, and a real answer can
        carry several distinct checkable claims -- measured against 5 real
        previously-failing topics, the actual rounds needed ranged 3-8 (one
        verified claim per round plus one final round), so 6 was cutting off
        genuinely thorough, still-converging critiques rather than only catching
        runaway loops.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for _round in range(max_tool_rounds):
            payload = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "response_format": {"type": "json_object"},
                "temperature": temperature,
                "max_tokens": self.max_tokens,
            }
            message = await self._post_chat_round(headers, payload)
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                parsed = _parse_json_tolerating_trailing_data(message.get("content") or "")
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    raise RuntimeError(f"Chat API response missing required keys: {missing}")
                return parsed

            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": tool_calls})
            for call in tool_calls:
                func = call.get("function", {})
                name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result_text = await tool_executor(name, arguments)
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result_text})

        raise RuntimeError(f"Chat API did not produce a final answer within {max_tool_rounds} tool-call rounds")
