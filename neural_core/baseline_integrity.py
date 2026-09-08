"""
Phase 2B — Baseline integrity verification.

The protected CPU/NumPy/float64 baseline lives in:

    neural_core/data/backup_20260826_103727/

Its SHA256 digests are recorded in ``neural_core/data/baseline_manifest.json``.
This module recomputes the digests and compares them to the manifest. The
baseline MUST remain byte-identical; any mismatch is a hard stop condition.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from neural_core.dataset_verification import sha256_of_file


def _default_backup_dir() -> Path:
    repo = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return repo / "neural_core" / "data" / "backup_20260826_103727"


def _default_manifest() -> Path:
    repo = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return repo / "neural_core" / "data" / "baseline_manifest.json"


def verify_baseline_integrity(backup_dir=None, manifest_path=None) -> Dict[str, any]:
    """
    Recompute SHA256 of baseline artifacts and compare to the manifest.

    Returns a dict with per-artifact results and an overall ``all_ok`` flag.
    """
    backup_dir = Path(backup_dir) if backup_dir else _default_backup_dir()
    manifest_path = Path(manifest_path) if manifest_path else _default_manifest()

    if not manifest_path.exists():
        return {"error": "manifest not found", "all_ok": False}
    if not backup_dir.exists():
        return {"error": "backup dir not found", "all_ok": False}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})

    results: List[Dict] = []
    for name, meta in artifacts.items():
        p = backup_dir / name
        if not p.exists():
            results.append({"artifact": name, "exists": False, "match": False})
            continue
        actual = sha256_of_file(p)
        expected = meta.get("sha256")
        size = p.stat().st_size
        results.append({
            "artifact": name,
            "exists": True,
            "size_bytes": size,
            "expected_size_bytes": meta.get("size_bytes"),
            "size_matches": size == meta.get("size_bytes"),
            "sha256": actual,
            "expected_sha256": expected,
            "match": actual == expected,
        })

    all_ok = all(r["match"] for r in results) and len(results) > 0
    return {
        "backup_dir": str(backup_dir),
        "manifest": str(manifest_path),
        "baseline_id": manifest.get("baseline_id"),
        "artifacts": results,
        "all_ok": all_ok,
    }
