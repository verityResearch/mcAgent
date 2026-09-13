import asyncio
from typing import Any, Callable, Dict, List, Tuple


class PipelineOrchestrator:
    def __init__(self, teacher: Any, static_oracle_check: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
                 judge: Any, concurrency_limit: int = 5):
        self.teacher = teacher
        self.static_oracle_check = static_oracle_check
        self.judge = judge
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def _process_fact_seeded(self, fact: Dict[str, Any], question: str) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                candidate = await self.teacher.generate_fact_seeded(fact, question)
            except Exception as exc:
                return {"sample": None, "lane": "fact_seeded",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "generation_error"}}
            try:
                loop = asyncio.get_event_loop()
                verification = await loop.run_in_executor(None, self.static_oracle_check, candidate, fact)
            except Exception as exc:
                return {"sample": candidate, "lane": "fact_seeded",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "oracle_error"}}
            return {"sample": candidate, "lane": "fact_seeded", "verification": verification}

    async def _process_free_recall(self, topic: str) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                candidate = await self.teacher.generate_free_recall(topic)
            except Exception as exc:
                return {"sample": None, "lane": "free_recall",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "generation_error"}}
            try:
                verification = await self.judge.judge(candidate)
            except Exception as exc:
                return {"sample": candidate, "lane": "free_recall",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "judge_error"}}
            return {"sample": candidate, "lane": "free_recall", "verification": verification}

    async def run_batch(self, fact_seeded_items: List[Tuple[Dict[str, Any], str]],
                        free_recall_topics: List[str]) -> List[Dict[str, Any]]:
        tasks = [self._process_fact_seeded(fact, question) for fact, question in fact_seeded_items]
        tasks += [self._process_free_recall(topic) for topic in free_recall_topics]
        return list(await asyncio.gather(*tasks))

    def log_yield(self, results: List[Dict[str, Any]]) -> None:
        admitted = sum(1 for r in results if r["verification"]["success"])
        print(f"Batch: {admitted}/{len(results)} admitted.")
