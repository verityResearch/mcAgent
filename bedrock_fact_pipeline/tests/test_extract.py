import os
import tarfile
from unittest.mock import patch

from src.data_report.extract import extract_pinned_version, release_tarball_url


def test_release_tarball_url():
    assert release_tarball_url("v1.26.30.5") == (
        "https://github.com/Mojang/bedrock-samples/archive/refs/tags/v1.26.30.5.tar.gz"
    )


def _make_fake_tarball(tar_path, has_behavior_pack):
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        import io
        payload = b"{}"
        info = tarfile.TarInfo(
            name="bedrock-samples-1.26.30.5/behavior_pack/recipes/x.json"
            if has_behavior_pack else "bedrock-samples-1.26.30.5/documentation/x.md"
        )
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


@patch("src.data_report.extract.download_file")
def test_extract_pinned_version_succeeds_when_behavior_pack_present(mock_download, tmp_path):
    tarball_path = tmp_path / "bedrock-samples-v1.26.30.5.tar.gz"

    def _fake_download(url, dest_path):
        _make_fake_tarball(dest_path, has_behavior_pack=True)

    mock_download.side_effect = _fake_download
    result = extract_pinned_version("v1.26.30.5", str(tmp_path))
    assert os.path.isdir(result)
    assert os.path.exists(os.path.join(tmp_path, "PINNED_VERSION.txt"))


@patch("src.data_report.extract.download_file")
def test_extract_pinned_version_raises_when_behavior_pack_missing(mock_download, tmp_path):
    def _fake_download(url, dest_path):
        _make_fake_tarball(dest_path, has_behavior_pack=False)

    mock_download.side_effect = _fake_download
    try:
        extract_pinned_version("v1.26.30.5", str(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "behavior_pack" in str(exc)
