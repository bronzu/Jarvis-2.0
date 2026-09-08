"""
Performance regression comparison.

Compares the current benchmark run against the documented Phase 2A baseline
recorded in the `baseline_benchmark.json` file (if present).

If the baseline is not trustworthy (outdated, missing, or measured on different
hardware), the tool declares BASELINE NOT TRUSTWORTHY and records a NEW
baseline from the current run.
"""

import json
import platform
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


BASELINE_PATH = Path("neural_core/data/baseline_benchmark.json")


def _load_baseline() -> Optional[Dict[str, Any]]:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _save_baseline(report: Dict[str, Any]) -> Path:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return BASELINE_PATH


def _is_trustworthy_baseline(baseline: Optional[Dict[str, Any]], current: Dict[str, Any]) -> bool:
    if not baseline:
        return False
    base_hw = baseline.get("hardware", {})
    cur_hw = current.get("hardware", {})
    if base_hw.get("platform") != cur_hw.get("platform"):
        return False
    if base_hw.get("python_version") != cur_hw.get("python_version"):
        return False
    if base_hw.get("machine") != cur_hw.get("machine"):
        return False
    if "inference" not in baseline or "training" not in baseline:
        return False
    return True


def compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"deltas": {}}
    for key in ("inference", "training", "startup", "model_loading"):
        b = baseline.get(key, {})
        c = current.get(key, {})
        if not b or not c:
            continue
        b_mean = b.get("mean")
        c_mean = c.get("mean")
        if not b_mean:
            continue
        delta_pct = (c_mean - b_mean) / b_mean * 100
        out["deltas"][key] = {
            "baseline_mean_seconds": b_mean,
            "current_mean_seconds": c_mean,
            "delta_pct": round(delta_pct, 2),
        }
    return out


def reconcile(current: Dict[str, Any], force_new: bool = False) -> Dict[str, Any]:
    baseline = _load_baseline()
    trustworthy = _is_trustworthy_baseline(baseline, current)
    if not trustworthy or force_new:
        path = _save_baseline(current)
        return {
            "baseline_status": "BASELINE NOT TRUSTWORTHY" if not trustworthy and baseline else "FRESH BASELINE",
            "baseline_saved_to": str(path),
            "comparison": None,
        }
    comparison = compare(baseline, current)
    return {
        "baseline_status": "TRUSTWORTHY",
        "baseline_path": str(BASELINE_PATH),
        "comparison": comparison,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--new-baseline":
        current = json.loads(Path("FINAL_BENCHMARK.json").read_text())
        print(json.dumps(reconcile(current, force_new=True), indent=2, ensure_ascii=False))
    else:
        current = json.loads(Path("FINAL_BENCHMARK.json").read_text())
        print(json.dumps(reconcile(current), indent=2, ensure_ascii=False))
