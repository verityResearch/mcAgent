from typing import Any, Dict

from src.llm_client import ChatClient


class Judge(ChatClient):
    async def _check_consistency(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You check whether a Minecraft: Bedrock Edition Q&A sample is internally consistent: does "
            "the reasoning actually support the answer, and does the answer address the question? "
            'Respond with JSON: {"consistent": bool, "issues": [str]}.'
        )
        user = (
            f"Question: {candidate['instruction']}\nReasoning: {candidate['reasoning']}\n"
            f"Answer: {candidate['answer']}"
        )
        return await self.call_chat(system, user, ["consistent", "issues"], temperature=0.0)

    async def _adversarial_critique(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are an adversarial critic. Try to find a factual or logical flaw in this Minecraft: "
            "Bedrock Edition answer -- including any claim that is actually true only for Java Edition. "
            "Be skeptical; default to flagging anything you cannot verify. "
            'Respond with JSON: {"critique_passed": bool, "issues": [str]}.'
        )
        user = f"Question: {candidate['instruction']}\nAnswer: {candidate['answer']}"
        return await self.call_chat(system, user, ["critique_passed", "issues"], temperature=0.0)

    async def judge(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        consistency = await self._check_consistency(candidate)
        if not consistency["consistent"]:
            return {"success": False, "errors": consistency["issues"] or ["reasoning inconsistent with answer"],
                    "failure_kind": "logical_inconsistency"}
        critique = await self._adversarial_critique(candidate)
        if not critique["critique_passed"]:
            return {"success": False, "errors": critique["issues"] or ["adversarial critique failed"],
                    "failure_kind": "adversarial_critique_failed"}
        return {"success": True, "errors": [], "failure_kind": None}
