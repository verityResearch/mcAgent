import json
from typing import Any, Dict

from src.llm_client import ChatClient


class TeacherModel(ChatClient):
    async def generate_fact_seeded(self, fact: Dict[str, Any], question: str) -> Dict[str, Any]:
        system = (
            "You write Minecraft: Bedrock Edition trivia training samples. You are given a GROUND-TRUTH "
            "fact and a question about it. Write a natural-language reasoning trace and answer that use "
            "ONLY the given fact -- never invent details, and never draw on Java Edition specifics that "
            "may differ. Also restate the fact's fields exactly as given, under 'structured_claim', so it "
            "can be mechanically checked. "
            'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str, '
            '"structured_claim": object}.'
        )
        user = f"Fact ({fact['category']} {fact['subject_id']}): {json.dumps(fact['fields'])}\nQuestion: {question}"
        return await self.call_chat(system, user, ["instruction", "reasoning", "answer", "structured_claim"])

    async def generate_free_recall(self, topic: str) -> Dict[str, Any]:
        system = (
            "You write Minecraft: Bedrock Edition strategy training samples. Given a topic, write a "
            "natural-language question, a reasoning trace, and an answer, drawing on general Bedrock "
            "Edition knowledge -- never Java Edition specifics that may differ. "
            'Respond with JSON: {"instruction": str, "reasoning": str, "answer": str}.'
        )
        user = f"Topic: {topic}"
        return await self.call_chat(system, user, ["instruction", "reasoning", "answer"])
