import pytest

from main import _parse_args, main


def test_parse_args_defaults():
    args = _parse_args(["--data-report-dir", "/tmp/data"])
    assert args.target == 10
    assert args.concurrency == 5
    assert args.fact_seeded_ratio == 0.8
    assert args.data_report_dir == "/tmp/data"


def test_parse_args_requires_data_report_dir():
    with pytest.raises(SystemExit):
        _parse_args([])


def test_main_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        main(["--data-report-dir", str(tmp_path), "--api-key", ""])
    assert "API key" in str(exc_info.value)


def test_main_refuses_to_overwrite_nonempty_output(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"instruction": "x"}\n')
    with pytest.raises(SystemExit) as exc_info:
        main(["--data-report-dir", str(tmp_path), "--output", str(output_path)])
    assert "already exists" in str(exc_info.value)


def test_parse_args_role_specific_overrides_default_to_none():
    args = _parse_args(["--data-report-dir", "/tmp/data"])
    assert args.teacher_model is None
    assert args.teacher_base_url is None
    assert args.teacher_api_key is None
    assert args.judge_model is None
    assert args.judge_base_url is None
    assert args.judge_api_key is None


def test_main_requires_teacher_api_key_when_missing_everywhere(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        main(["--data-report-dir", str(tmp_path), "--judge-api-key", "judge-key"])
    message = str(exc_info.value).lower()
    assert "teacher" in message
    assert "api key" in message


def test_main_requires_judge_api_key_when_missing_everywhere(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        main(["--data-report-dir", str(tmp_path), "--teacher-api-key", "teacher-key"])
    message = str(exc_info.value).lower()
    assert "judge" in message
    assert "api key" in message
