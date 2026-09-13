import asyncio

from src.generation.orchestrator import PipelineOrchestrator
from src.llm_client import ChatAPITimeoutError


class _FakeTeacher:
    def __init__(self, fact_seeded_result=None, fact_seeded_error=None,
                 free_recall_result=None, free_recall_error=None):
        self._fact_seeded_result = fact_seeded_result
        self._fact_seeded_error = fact_seeded_error
        self._free_recall_result = free_recall_result
        self._free_recall_error = free_recall_error

    async def generate_fact_seeded(self, fact, question):
        if self._fact_seeded_error:
            raise self._fact_seeded_error
        return self._fact_seeded_result

    async def generate_free_recall(self, topic):
        if self._free_recall_error:
            raise self._free_recall_error
        return self._free_recall_result


class _FakeJudge:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def judge(self, candidate):
        if self._error:
            raise self._error
        return self._result


def _static_check_always_admits(candidate, fact):
    return {"success": True, "errors": [], "failure_kind": None}


class _CountingTeacher:
    """Raises the given error every call and counts how many times it was invoked."""

    def __init__(self, error):
        self._error = error
        self.call_count = 0

    async def generate_free_recall(self, topic):
        self.call_count += 1
        raise self._error


class _CountingJudge:
    """Raises the given error every call and counts how many times it was invoked."""

    def __init__(self, error):
        self._error = error
        self.call_count = 0

    async def judge(self, candidate):
        self.call_count += 1
        raise self._error


def test_run_batch_routes_fact_seeded_to_static_oracle():
    teacher = _FakeTeacher(fact_seeded_result={"instruction": "q", "structured_claim": {}})
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, _FakeJudge())
    fact = {"category": "recipe", "subject_id": "x", "fields": {}}

    results = asyncio.run(orchestrator.run_batch([(fact, "q")], []))
    assert len(results) == 1
    assert results[0]["lane"] == "fact_seeded"
    assert results[0]["verification"]["success"] is True


def test_run_batch_routes_free_recall_to_judge():
    teacher = _FakeTeacher(free_recall_result={"instruction": "q", "reasoning": "r", "answer": "a"})
    judge = _FakeJudge(result={"success": True, "errors": [], "failure_kind": None})
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, judge)

    results = asyncio.run(orchestrator.run_batch([], ["some topic"]))
    assert len(results) == 1
    assert results[0]["lane"] == "free_recall"
    assert results[0]["verification"]["success"] is True


def test_run_batch_isolates_generation_failure():
    teacher = _FakeTeacher(fact_seeded_error=RuntimeError("boom"),
                            free_recall_result={"instruction": "q", "reasoning": "r", "answer": "a"})
    judge = _FakeJudge(result={"success": True, "errors": [], "failure_kind": None})
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, judge)
    fact = {"category": "recipe", "subject_id": "x", "fields": {}}

    results = asyncio.run(orchestrator.run_batch([(fact, "q")], ["some topic"]))
    assert len(results) == 2
    fact_seeded_result = next(r for r in results if r["lane"] == "fact_seeded")
    free_recall_result = next(r for r in results if r["lane"] == "free_recall")
    assert fact_seeded_result["verification"]["success"] is False
    assert fact_seeded_result["verification"]["failure_kind"] == "generation_error"
    assert free_recall_result["verification"]["success"] is True


def test_run_batch_isolates_oracle_failure():
    def _static_check_raises(candidate, fact):
        raise RuntimeError("oracle boom")

    teacher = _FakeTeacher(fact_seeded_result={"instruction": "q", "structured_claim": {}})
    orchestrator = PipelineOrchestrator(teacher, _static_check_raises, _FakeJudge())
    fact = {"category": "recipe", "subject_id": "x", "fields": {}}

    results = asyncio.run(orchestrator.run_batch([(fact, "q")], []))
    assert len(results) == 1
    assert results[0]["lane"] == "fact_seeded"
    assert results[0]["verification"]["success"] is False
    assert results[0]["verification"]["failure_kind"] == "oracle_error"


def test_circuit_breaker_opens_after_consecutive_timeouts_and_stops_calling_teacher():
    teacher = _CountingTeacher(ChatAPITimeoutError("server wedged"))
    # concurrency_limit=1 makes calls run strictly one at a time, so "consecutive"
    # matches call order deterministically.
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, _FakeJudge(), concurrency_limit=1)

    async def _run():
        results = []
        for _ in range(5):
            results.append(await orchestrator._process_free_recall("some topic"))
        return results

    results = asyncio.run(_run())
    failure_kinds = [r["verification"]["failure_kind"] for r in results]
    # First 3 calls actually hit the (timing-out) teacher and are reported as
    # generation_error; only once the streak reaches the threshold does the
    # breaker open and short-circuit the remaining calls without invoking it.
    assert failure_kinds == ["generation_error", "generation_error", "generation_error",
                              "circuit_open", "circuit_open"]
    assert teacher.call_count == 3


def test_circuit_breaker_does_not_open_on_non_timeout_errors():
    teacher = _CountingTeacher(RuntimeError("model returned garbage"))
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, _FakeJudge(), concurrency_limit=1)

    async def _run():
        results = []
        for _ in range(5):
            results.append(await orchestrator._process_free_recall("some topic"))
        return results

    results = asyncio.run(_run())
    assert all(r["verification"]["failure_kind"] == "generation_error" for r in results)
    assert teacher.call_count == 5


class _AlwaysSucceedsTeacher:
    """A teacher whose calls always succeed, counting fact-seeded invocations."""

    def __init__(self):
        self.fact_seeded_call_count = 0

    async def generate_free_recall(self, topic):
        return {"instruction": "q", "reasoning": "r", "answer": "a"}

    async def generate_fact_seeded(self, fact, question):
        self.fact_seeded_call_count += 1
        return {"instruction": "q", "structured_claim": {}}


def test_judge_circuit_breaker_does_not_stop_fact_seeded_lane():
    """The fact-seeded lane never calls the judge at all (main.py allows the teacher
    and judge to point at entirely different backends via --teacher-base-url /
    --judge-base-url), so a wedged judge must not block fact-seeded generation."""
    teacher = _AlwaysSucceedsTeacher()
    judge = _CountingJudge(ChatAPITimeoutError("judge backend wedged"))
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, judge, concurrency_limit=1)
    fact = {"category": "recipe", "subject_id": "x", "fields": {}}

    async def _run():
        # Trip the judge breaker via free_recall calls (teacher succeeds each time,
        # only the judge call times out).
        for _ in range(3):
            await orchestrator._process_free_recall("some topic")
        # The judge breaker should now be open, but fact_seeded never touches the
        # judge and must still succeed.
        return await orchestrator._process_fact_seeded(fact, "q")

    result = asyncio.run(_run())
    assert result["verification"]["success"] is True
    assert result["verification"]["failure_kind"] is None
    assert judge.call_count == 3
    assert teacher.fact_seeded_call_count == 1


def test_run_batch_resets_circuit_breakers_so_a_later_batch_can_recover():
    """main.py's normal usage calls run_batch() repeatedly in a loop until --target
    is reached. A backend that was wedged in one batch (e.g. before a manual server
    restart) may have recovered by the next one -- the breaker must not stay open
    forever once tripped."""
    teacher = _CountingTeacher(ChatAPITimeoutError("server wedged"))
    orchestrator = PipelineOrchestrator(teacher, _static_check_always_admits, _FakeJudge(), concurrency_limit=1)

    async def _run():
        # First batch: 3 free-recall items, all time out -> breaker trips.
        first_batch = await orchestrator.run_batch([], ["t1", "t2", "t3"])
        # Second batch: same teacher (still erroring, simulating a still-wedged
        # server) -- but since run_batch resets the breaker, this batch's items
        # should actually attempt the call again rather than short-circuiting.
        second_batch = await orchestrator.run_batch([], ["t4"])
        return first_batch, second_batch

    first_batch, second_batch = asyncio.run(_run())
    assert [r["verification"]["failure_kind"] for r in first_batch] == \
        ["generation_error", "generation_error", "generation_error"]
    # If the breaker had NOT reset, this would be "circuit_open" without a new call.
    assert second_batch[0]["verification"]["failure_kind"] == "generation_error"
    assert teacher.call_count == 4


def test_log_yield_reports_only_admit_count_when_everything_succeeds(capsys):
    orchestrator = PipelineOrchestrator(_FakeTeacher(), _static_check_always_admits, _FakeJudge())
    results = [{"verification": {"success": True, "failure_kind": None}}] * 3

    orchestrator.log_yield(results)

    out = capsys.readouterr().out
    assert out == "Batch: 3/3 admitted.\n"


def test_log_yield_reports_failure_kind_breakdown(capsys):
    """A real run has no per-item JSON dump like the diagnostic script's -- without
    this breakdown, "Batch: 4/5 admitted" gives no way to tell a legitimate
    rejection from an infrastructure problem that actually needs attention."""
    orchestrator = PipelineOrchestrator(_FakeTeacher(), _static_check_always_admits, _FakeJudge())
    results = [
        {"verification": {"success": True, "failure_kind": None}},
        {"verification": {"success": False, "failure_kind": "generation_error"}},
        {"verification": {"success": False, "failure_kind": "generation_error"}},
        {"verification": {"success": False, "failure_kind": "mechanism_claim_incorrect"}},
    ]

    orchestrator.log_yield(results)

    out = capsys.readouterr().out
    assert "Batch: 1/4 admitted." in out
    assert "generation_error: 2" in out
    assert "mechanism_claim_incorrect: 1" in out
