import json
import os
import subprocess
import urllib.request
from typing import Any, Dict, Optional

VERSION_MANIFEST_URL = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"


def find_version_entry(manifest: Dict[str, Any], version_id: str) -> Optional[Dict[str, Any]]:
    for entry in manifest.get("versions", []):
        if entry.get("id") == version_id:
            return entry
    return None


def fetch_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def server_jar_url(version_id: str) -> str:
    manifest = fetch_json(VERSION_MANIFEST_URL)
    entry = find_version_entry(manifest, version_id)
    if entry is None:
        raise RuntimeError(f"Version {version_id!r} not found in Mojang version manifest")
    version_meta = fetch_json(entry["url"])
    server = version_meta.get("downloads", {}).get("server")
    if not server or "url" not in server:
        raise RuntimeError(f"Version {version_id!r} has no server download listed")
    return server["url"]


def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".tmp"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)


def run_data_generator(server_jar_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["java", "-DbundlerMainClass=net.minecraft.data.Main", "-jar", server_jar_path,
         "--all", "--output", output_dir],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Data generator exited {result.returncode}. stderr:\n{result.stderr[-4000:]}")
    data_dir = os.path.join(output_dir, "data")
    if not os.path.isdir(data_dir):
        raise RuntimeError(
            f"Data generator reported success but {data_dir} was not created; the invocation "
            f"flags may not match this server jar version. stdout:\n{result.stdout[-2000:]}"
        )


def extract_pinned_version(version_id: str, workdir: str) -> str:
    jar_path = os.path.join(workdir, f"server-{version_id}.jar")
    if not os.path.exists(jar_path):
        download_file(server_jar_url(version_id), jar_path)
    output_dir = os.path.join(workdir, f"generated-{version_id}")
    run_data_generator(jar_path, output_dir)
    with open(os.path.join(workdir, "PINNED_VERSION.txt"), "w", encoding="utf-8") as f:
        f.write(version_id + "\n")
    # Return the report root (parent of the generator's own "data" dir), not
    # "data" itself: fact_sampler._subject_id expects data_report_dir to be
    # one level above "data" so its "data/<namespace>/<registry>/<name>"
    # relative-path parsing lines up (see test_extract_pinned_version_return_
    # value_feeds_discover_fact_records for the regression this guards).
    return output_dir
