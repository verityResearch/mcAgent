"""Convert admitted pipeline results into training-ready JSONL.

Takes the orchestrator's per-candidate result list and writes only the admitted
samples as JSONL rows of shape
``{"instruction": str, "response": str, "verification": {"method": str}}``. The
``response`` folds the sample's reasoning and answer into a single
``<thought>``-tagged string, and ``method`` records which oracle admitted the
sample (``"static_oracle"`` for the fact-seeded lane, ``"llm_judge"`` for the
free-recall lane). The write is atomic: rows go to a temporary file that is
renamed into place only after every admitted row is written, so a partially
written file never appears at ``output_path``.
"""
import json
import os
from typing import Any, Dict, List


def convert_to_jsonl(results: List[Dict[str, Any]], output_path: str) -> int:
    """Write admitted samples from ``results`` to ``output_path`` as JSONL.

    ``results`` is the orchestrator's result list; each entry has a ``lane``, an
    optional ``sample`` dict, and a ``verification`` verdict. A result is
    admitted when ``verification["success"]`` is true and ``sample`` is present.
    Each admitted sample becomes one JSONL row with the training instruction, a
    ``<thought>``-tagged response, and the admitting oracle's ``method``. The
    file is written atomically via a ``.tmp`` sidecar and ``os.replace``; on any
    error the sidecar is removed and the exception re-raised so ``output_path``
    is never left partially written. Returns the number of admitted rows written
    (zero still produces an empty file).
    """
    admitted = [r for r in results if r["verification"]["success"] and r["sample"]]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for result in admitted:
                sample = result["sample"]
                method = "static_oracle" if result["lane"] == "fact_seeded" else "llm_judge"
                row = {
                    "instruction": sample["instruction"],
                    "response": f"<thought>\n{sample['reasoning']}\n</thought>\n{sample['answer']}",
                    "verification": {"method": method},
                }
                f.write(json.dumps(row) + "\n")
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return len(admitted)
