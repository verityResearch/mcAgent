import asyncio

from src.generation.orchestrator import PipelineOrchestrator


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
