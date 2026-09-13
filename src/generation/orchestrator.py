import asyncio
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from src.llm_client import ChatAPITimeoutError

# A wedged local mlx_lm.server was observed to stay unresponsive well after the
# load that caused it stopped -- every remaining item in a batch would otherwise
# each individually wait out their own request timeout before failing, one at a
# time, for no benefit (a genuinely stuck server isn't going to answer the next
# call either). After this many CONSECUTIVE ChatAPITimeoutErrors, stop launching
# new LLM calls for the rest of this batch and fail them immediately instead.
_CIRCUIT_BREAKER_THRESHOLD = 3


class _CircuitBreaker:
    """Tracks consecutive ``ChatAPITimeoutError``s for one backend and trips after
    ``_CIRCUIT_BREAKER_THRESHOLD`` in a row. Any non-timeout outcome (success, or a
    different kind of failure like a malformed response) resets the streak, since
    that indicates the server is still actually responding.
    """

    def __init__(self):
        self.consecutive_timeouts = 0
        self.open = False

    def reset(self) -> None:
        self.consecutive_timeouts = 0
        self.open = False

    def record(self, timed_out: bool) -> None:
        if timed_out:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts >= _CIRCUIT_BREAKER_THRESHOLD:
                self.open = True
        else:
            self.consecutive_timeouts = 0


class PipelineOrchestrator:
    def __init__(self, teacher: Any, static_oracle_check: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
                 judge: Any, concurrency_limit: int = 5):
        self.teacher = teacher
        self.static_oracle_check = static_oracle_check
        self.judge = judge
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        # main.py allows the teacher and judge to point at entirely different
        # backends (--teacher-base-url / --judge-base-url), so a wedged judge
        # backend must not stop the fact-seeded lane, which never calls the judge
        # at all -- each gets its own breaker rather than sharing one flag.
        self._teacher_breaker = _CircuitBreaker()
        self._judge_breaker = _CircuitBreaker()

    async def _process_fact_seeded(self, fact: Dict[str, Any], question: str) -> Dict[str, Any]:
        async with self.semaphore:
            if self._teacher_breaker.open:
                return {"sample": None, "lane": "fact_seeded", "verification": {
                    "success": False, "errors": ["circuit breaker open: too many consecutive teacher API timeouts"],
                    "failure_kind": "circuit_open"}}
            try:
                candidate = await self.teacher.generate_fact_seeded(fact, question)
            except Exception as exc:
                self._teacher_breaker.record(timed_out=isinstance(exc, ChatAPITimeoutError))
                return {"sample": None, "lane": "fact_seeded",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "generation_error"}}
            self._teacher_breaker.record(timed_out=False)
            try:
                loop = asyncio.get_event_loop()
                verification = await loop.run_in_executor(None, self.static_oracle_check, candidate, fact)
            except Exception as exc:
                return {"sample": candidate, "lane": "fact_seeded",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "oracle_error"}}
            return {"sample": candidate, "lane": "fact_seeded", "verification": verification}

    async def _process_free_recall(self, topic: str) -> Dict[str, Any]:
        async with self.semaphore:
            if self._teacher_breaker.open:
                return {"sample": None, "lane": "free_recall", "verification": {
                    "success": False, "errors": ["circuit breaker open: too many consecutive teacher API timeouts"],
                    "failure_kind": "circuit_open"}}
            try:
                candidate = await self.teacher.generate_free_recall(topic)
            except Exception as exc:
                self._teacher_breaker.record(timed_out=isinstance(exc, ChatAPITimeoutError))
                return {"sample": None, "lane": "free_recall",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "generation_error"}}
            self._teacher_breaker.record(timed_out=False)
            if self._judge_breaker.open:
                return {"sample": candidate, "lane": "free_recall", "verification": {
                    "success": False, "errors": ["circuit breaker open: too many consecutive judge API timeouts"],
                    "failure_kind": "circuit_open"}}
            try:
                verification = await self.judge.judge(candidate)
            except Exception as exc:
                self._judge_breaker.record(timed_out=isinstance(exc, ChatAPITimeoutError))
                return {"sample": candidate, "lane": "free_recall",
                        "verification": {"success": False, "errors": [str(exc)], "failure_kind": "judge_error"}}
            self._judge_breaker.record(timed_out=False)
            return {"sample": candidate, "lane": "free_recall", "verification": verification}

    async def run_batch(self, fact_seeded_items: List[Tuple[Dict[str, Any], str]],
                        free_recall_topics: List[str]) -> List[Dict[str, Any]]:
        """Run one batch of fact-seeded and free-recall items concurrently.

        Resets both circuit breakers at the start of each call: a backend that was
        wedged in an earlier batch (main.py's normal usage calls this repeatedly in
        a loop until --target is reached) may have recovered since -- e.g. after a
        manual server restart -- and a breaker that never reset would otherwise
        silently fail every future batch for the rest of the run.
        """
        self._teacher_breaker.reset()
        self._judge_breaker.reset()
        tasks = [self._process_fact_seeded(fact, question) for fact, question in fact_seeded_items]
        tasks += [self._process_free_recall(topic) for topic in free_recall_topics]
        return list(await asyncio.gather(*tasks))

    def log_yield(self, results: List[Dict[str, Any]]) -> None:
        """Print this batch's admit count, and a failure_kind breakdown if anything
        didn't succeed. A real run has no equivalent of the diagnostic script's
        per-item JSON dump -- without this, "Batch: 4/5 admitted" repeated across a
        long run gives no way to tell a legitimate rejection (adversarial_critique_
        failed, mechanism_claim_incorrect, ...) from an infrastructure problem
        (generation_error, judge_error, circuit_open) that actually needs attention.
        """
        admitted = sum(1 for r in results if r["verification"]["success"])
        print(f"Batch: {admitted}/{len(results)} admitted.")
        failure_kinds = Counter(r["verification"]["failure_kind"] for r in results if not r["verification"]["success"])
        if failure_kinds:
            breakdown = ", ".join(f"{kind}: {count}" for kind, count in failure_kinds.most_common())
            print(f"  Failures: {breakdown}")
