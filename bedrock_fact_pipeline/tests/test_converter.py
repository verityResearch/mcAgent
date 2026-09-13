import json
import os

from src.generation.converter import convert_to_jsonl


def test_convert_to_jsonl_writes_only_admitted_rows(tmp_path):
    results = [
        {"lane": "fact_seeded", "sample": {"instruction": "q1", "reasoning": "r1", "answer": "a1"},
         "verification": {"success": True, "errors": [], "failure_kind": None}},
        {"lane": "free_recall", "sample": {"instruction": "q2", "reasoning": "r2", "answer": "a2"},
         "verification": {"success": True, "errors": [], "failure_kind": None}},
        {"lane": "fact_seeded", "sample": {"instruction": "q3", "reasoning": "r3", "answer": "a3"},
         "verification": {"success": False, "errors": ["mismatch"], "failure_kind": "fact_mismatch"}},
    ]
    output_path = str(tmp_path / "out.jsonl")

    count = convert_to_jsonl(results, output_path)

    assert count == 2
    lines = [json.loads(line) for line in open(output_path, encoding="utf-8")]
    assert len(lines) == 2
    assert lines[0]["instruction"] == "q1"
    assert lines[0]["verification"]["method"] == "static_oracle"
    assert lines[1]["verification"]["method"] == "llm_judge"
    assert "<thought>" in lines[0]["response"]
    assert not os.path.exists(output_path + ".tmp")


def test_convert_to_jsonl_empty_admitted_writes_empty_file(tmp_path):
    results = [{"lane": "fact_seeded", "sample": None,
                "verification": {"success": False, "errors": ["x"], "failure_kind": "generation_error"}}]
    output_path = str(tmp_path / "out.jsonl")

    count = convert_to_jsonl(results, output_path)

    assert count == 0
    assert os.path.exists(output_path)
    assert open(output_path, encoding="utf-8").read() == ""
