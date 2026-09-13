import os
from unittest.mock import MagicMock, patch

from src.data_report.extract import extract_pinned_version, find_version_entry, run_data_generator
from src.generation.fact_sampler import discover_fact_records


def test_find_version_entry_found():
    manifest = {"versions": [{"id": "1.21", "url": "https://example/1.21.json"}]}
    assert find_version_entry(manifest, "1.21") == manifest["versions"][0]


def test_find_version_entry_missing():
    assert find_version_entry({"versions": []}, "1.21") is None


@patch("src.data_report.extract.subprocess.run")
def test_run_data_generator_raises_on_nonzero_exit(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
    try:
        run_data_generator(str(tmp_path / "server.jar"), str(tmp_path / "out"))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "boom" in str(exc)


@patch("src.data_report.extract.subprocess.run")
def test_run_data_generator_raises_when_data_dir_missing(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    output_dir = tmp_path / "out"
    try:
        run_data_generator(str(tmp_path / "server.jar"), str(output_dir))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "data" in str(exc)


@patch("src.data_report.extract.subprocess.run")
def test_run_data_generator_succeeds_when_data_dir_created(mock_run, tmp_path):
    output_dir = tmp_path / "out"

    def _fake_run(*args, **kwargs):
        (output_dir / "data").mkdir(parents=True)
        return MagicMock(returncode=0, stdout="ok", stderr="")

    mock_run.side_effect = _fake_run
    run_data_generator(str(tmp_path / "server.jar"), str(output_dir))


@patch("src.data_report.extract.run_data_generator")
@patch("src.data_report.extract.download_file")
@patch("src.data_report.extract.server_jar_url")
def test_extract_pinned_version_return_value_feeds_discover_fact_records(
    mock_server_jar_url, mock_download_file, mock_run_data_generator, tmp_path
):
    # Regression test: extract_pinned_version's return value must be exactly
    # what discover_fact_records/_subject_id expect as data_report_dir (the
    # parent of the generator's "data" directory). These two modules'
    # contracts silently diverged with no test connecting them -- a real
    # extraction produced only garbled "minecraft/recipe/x"-style subject_ids
    # (missing the "minecraft:x" colon form) until this was caught.
    mock_server_jar_url.return_value = "https://example.invalid/server.jar"

    def _fake_generate(jar_path, output_dir):
        recipe_dir = os.path.join(output_dir, "data", "minecraft", "recipe")
        os.makedirs(recipe_dir, exist_ok=True)
        with open(os.path.join(recipe_dir, "stick.json"), "w", encoding="utf-8") as f:
            f.write('{"type": "minecraft:crafting_shapeless", '
                    '"ingredients": [{"item": "minecraft:planks"}], '
                    '"result": {"id": "minecraft:stick", "count": 4}}')

    mock_run_data_generator.side_effect = _fake_generate

    data_report_dir = extract_pinned_version("26.2", str(tmp_path))
    records = discover_fact_records(data_report_dir)

    assert len(records) == 1
    assert records[0]["subject_id"] == "minecraft:stick"
