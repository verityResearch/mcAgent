import json
from unittest.mock import patch

from main import main


_RECIPE = {
    "minecraft:recipe_shaped": {
        "description": {"identifier": "minecraft:acacia_boat"},
        "key": {"#": {"item": "minecraft:acacia_planks"}},
        "result": {"item": "minecraft:acacia_boat"},
    }
}


class _StubTeacher:
    def __init__(self, *args, **kwargs):
        pass

    async def generate_fact_seeded(self, fact, question):
        return {
            "instruction": question,
            "reasoning": "The recipe lists these exact ingredients and result.",
            "answer": "Acacia planks craft into an acacia boat.",
            "structured_claim": dict(fact["fields"]),
        }

    async def generate_free_recall(self, topic):
        return {
            "instruction": f"Tell me about: {topic}",
            "reasoning": "General strategy knowledge applies here.",
            "answer": "Use a water bucket on a lava source block.",
        }


class _StubJudge:
    def __init__(self, *args, **kwargs):
        pass

    async def judge(self, candidate):
        return {"success": True, "errors": [], "failure_kind": None}


def test_end_to_end_smoke(tmp_path):
    data_report_dir = tmp_path / "extracted"
    recipe_path = data_report_dir / "bedrock-samples-1.26.30.5/behavior_pack/recipes/acacia_boat.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(_RECIPE))

    output_path = tmp_path / "out.jsonl"

    with patch("main.TeacherModel", _StubTeacher), patch("main.Judge", _StubJudge):
        main([
            "--data-report-dir", str(data_report_dir),
            "--output", str(output_path),
            "--api-key", "test-key",
            "--target", "3",
            "--concurrency", "3",
            "--fact-seeded-ratio", "0.5",
        ])

    lines = [json.loads(line) for line in open(output_path, encoding="utf-8")]
    assert len(lines) >= 3
    for row in lines:
        assert row["instruction"]
        assert "<thought>" in row["response"]
        assert row["verification"]["method"] in ("static_oracle", "llm_judge")
    methods = {row["verification"]["method"] for row in lines}
    assert methods == {"static_oracle", "llm_judge"}


class _RecordingTeacher(_StubTeacher):
    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingTeacher.last_kwargs = kwargs


class _RecordingJudge(_StubJudge):
    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingJudge.last_kwargs = kwargs


def test_main_applies_role_specific_overrides_with_shared_fallback(tmp_path):
    data_report_dir = tmp_path / "extracted"
    recipe_path = data_report_dir / "bedrock-samples-1.26.30.5/behavior_pack/recipes/acacia_boat.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(_RECIPE))
    output_path = tmp_path / "out.jsonl"

    with patch("main.TeacherModel", _RecordingTeacher), patch("main.Judge", _RecordingJudge):
        main([
            "--data-report-dir", str(data_report_dir),
            "--output", str(output_path),
            "--api-key", "shared-key",
            "--model", "shared-model",
            "--base-url", "https://shared.example/v1",
            "--teacher-model", "teacher-only-model",
            "--judge-api-key", "judge-only-key",
            "--target", "1",
            "--concurrency", "1",
        ])

    assert _RecordingTeacher.last_kwargs["model_name"] == "teacher-only-model"
    assert _RecordingTeacher.last_kwargs["base_url"] == "https://shared.example/v1"
    assert _RecordingTeacher.last_kwargs["api_key"] == "shared-key"

    assert _RecordingJudge.last_kwargs["model_name"] == "shared-model"
    assert _RecordingJudge.last_kwargs["base_url"] == "https://shared.example/v1"
    assert _RecordingJudge.last_kwargs["api_key"] == "judge-only-key"


def test_main_succeeds_with_only_role_specific_api_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data_report_dir = tmp_path / "extracted"
    recipe_path = data_report_dir / "bedrock-samples-1.26.30.5/behavior_pack/recipes/acacia_boat.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps(_RECIPE))
    output_path = tmp_path / "out.jsonl"

    with patch("main.TeacherModel", _StubTeacher), patch("main.Judge", _StubJudge):
        main([
            "--data-report-dir", str(data_report_dir),
            "--output", str(output_path),
            "--teacher-api-key", "teacher-key",
            "--judge-api-key", "judge-key",
            "--target", "1",
            "--concurrency", "1",
        ])

    assert output_path.exists()
    assert output_path.read_text().strip() != ""
