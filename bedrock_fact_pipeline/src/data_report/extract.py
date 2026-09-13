import os
import tarfile
import urllib.request


def release_tarball_url(tag: str) -> str:
    return f"https://github.com/Mojang/bedrock-samples/archive/refs/tags/{tag}.tar.gz"


def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".tmp"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)


def _contains_behavior_pack(root: str) -> bool:
    for dirpath, dirnames, _filenames in os.walk(root):
        if "behavior_pack" in dirnames:
            return True
    return False


def extract_pinned_version(tag: str, workdir: str) -> str:
    tarball_path = os.path.join(workdir, f"bedrock-samples-{tag}.tar.gz")
    if not os.path.exists(tarball_path):
        download_file(release_tarball_url(tag), tarball_path)
    output_dir = os.path.join(workdir, f"extracted-{tag}")
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(output_dir, filter="data")
    if not _contains_behavior_pack(output_dir):
        raise RuntimeError(
            f"Extracted {tarball_path} but found no behavior_pack/ directory under {output_dir}; "
            f"the tarball layout may not match what this extractor expects."
        )
    with open(os.path.join(workdir, "PINNED_VERSION.txt"), "w", encoding="utf-8") as f:
        f.write(tag + "\n")
    return output_dir
