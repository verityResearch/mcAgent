import json
import os
from typing import Any, Dict, List


def convert_to_jsonl(results: List[Dict[str, Any]], output_path: str) -> int:
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
