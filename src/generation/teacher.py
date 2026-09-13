import json
from typing import Any, Dict

from src.llm_client import ChatClient
from src.tools.minecraft_wiki import MINECRAFT_WIKI_TOOL_SCHEMA, execute_minecraft_wiki_tool


class TeacherModel(ChatClient):
    """Teacher that turns a ground-truth fact into a training candidate."""

    async def generate_fact_seeded(self, fact: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Generate a fact-seeded candidate for ``question`` about ``fact``.

        Returns a dict with ``instruction``, ``reasoning``, ``answer``, and
        ``structured_claim`` keys. Raises ``RuntimeError`` if the LLM call fails
        or the response is missing any of those keys.
        """
        system = (
            "You write Minecraft trivia training samples. You are given a GROUND-TRUTH fact and a "
            "question about it. Put that question itself into 'instruction' -- verbatim, or lightly "
            "rephrased into a natural standalone question, but it must still ask specifically about "
            "this fact's subject, not describe the exercise you're doing. Write a natural-language "
            "reasoning trace and answer that use ONLY the given fact -- never invent details. Also "
            "restate the fact's fields exactly as given, under 'structured_claim', so it can be "
            "mechanically checked: structured_claim must be the SAME shape as the fact's own fields "
            "object (same keys, same values, copied verbatim from what you were given), regardless of "
            "what the question asks -- it is a restatement of the given fact, not an encoding of the "
            "question or of your answer. "
            'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str, '
            '"structured_claim": object}.'
        )
        user = f"Fact ({fact['category']} {fact['subject_id']}): {json.dumps(fact['fields'])}\nQuestion: {question}"
        return await self.call_chat(system, user, ["instruction", "reasoning", "answer", "structured_claim"])

    async def generate_free_recall(self, topic: str) -> Dict[str, Any]:
        """Generate a free-recall candidate for ``topic`` from general knowledge.

        Returns a dict with ``instruction``, ``reasoning``, and ``answer`` keys
        (no ``structured_claim``, since nothing is mechanically checkable here).
        Before answering, the model may call the ``search_minecraft_wiki`` tool
        to verify specific facts (ore generation depths, enchantment behavior,
        trade ratios, etc.) against the live wiki rather than from memory alone.
        Raises ``RuntimeError`` if the LLM call fails, the response is missing
        any of those keys, or ``instruction``/``reasoning``/``answer`` are not
        plain strings (observed in practice: the model sometimes answers with
        a nested object of key-value pairs instead of prose, which the key
        presence check alone doesn't catch).
        """
        system = (
            "You write Minecraft strategy training samples. Given a topic, write a natural-language "
            "question, a reasoning trace, and an answer, drawing on general Minecraft knowledge.\n"
            "Before you write each specific numeric or mechanical claim (a depth, a tool tier, an "
            "enchantment's behavior, a trade ratio, a crafting recipe, and similar details), pause and "
            "honestly ask yourself: do I actually know this precisely, or am I estimating or guessing "
            "and it merely sounds confident? Recalled numbers and mechanics are exactly the kind of "
            "thing that feels certain while being wrong. If there is any real doubt, use the "
            "search_minecraft_wiki tool to check before writing the claim, rather than after. If you "
            "are genuinely certain (e.g. you just verified it, or it's basic/stable game structure), "
            "you do not need to look it up. "
            'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str}.'
        )
        user = f"Topic: {topic}"
        candidate = await self.call_chat_with_tools(
            system, user, ["instruction", "reasoning", "answer"],
            tools=[MINECRAFT_WIKI_TOOL_SCHEMA], tool_executor=execute_minecraft_wiki_tool)
        non_string = [k for k in ("instruction", "reasoning", "answer") if not isinstance(candidate[k], str)]
        if non_string:
            raise RuntimeError(f"Chat API response fields must be strings, got non-string: {non_string}")
        return candidate
