"""
Extended dataset audit combining:
  - existing line/token/overlap statistics (neural_core.dataset_audit)
  - provenance manifest (source_id / template_id / generation_group / split)
  - MinHash + LSH near-duplicate detection (full dataset, no sampling)
  - leakage detection (source, template, group, sentence_id)

Produces a single JSON report consumable by tests and the third-pass final report.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from neural_core.dataset_audit import audit_dataset
from neural_core.leakage_audit import audit_leakage
from neural_core.minhash_lsh import audit_dataset_near_duplicates
from neural_core.provenance_pipeline import PROVENANCE_FILE


def audit_dataset_full(
    train_path: Path = Path("neural_core/data/train.txt"),
    val_path: Path = Path("neural_core/data/validation.txt"),
    test_path: Path = Path("neural_core/data/test.txt"),
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
) -> Dict[str, Any]:
    base = audit_dataset()
    provenance = audit_leakage() if PROVENANCE_FILE.exists() else {
        "error": f"provenance file not found: {PROVENANCE_FILE}",
    }
    near_dups = audit_dataset_near_duplicates(
        train_path=train_path, val_path=val_path, test_path=test_path,
        threshold=threshold, num_perm=num_perm, ngram=ngram,
    )

    base["provenance"] = {
        "manifest_path": str(PROVENANCE_FILE),
        "template_leakage_measured": provenance != {"error": f"provenance file not found: {PROVENANCE_FILE}"},
        "source_leakage_measured": provenance != {"error": f"provenance file not found: {PROVENANCE_FILE}"},
        "group_leakage_measured": provenance != {"error": f"provenance file not found: {PROVENANCE_FILE}"},
        "details": provenance,
    }
    base["near_duplicates"] = {
        "method": near_dups["params"]["method"],
        "threshold": near_dups["params"]["threshold"],
        "num_perm": near_dups["params"]["num_perm"],
        "ngram": near_dups["params"]["ngram"],
        "bands": near_dups["params"]["bands"],
        "rows": near_dups["params"]["rows"],
        "num_documents": near_dups["totals"]["num_documents"],
        "num_candidate_pairs": near_dups["totals"]["num_candidate_pairs"],
        "num_verified_pairs": near_dups["totals"]["num_verified_pairs"],
        "cross_split_count": near_dups["cross_split_count"],
        "histogram": near_dups["similarity_histogram"],
    }
    base["valid"] = (
        base["valid"]
        and base["provenance"]["source_leakage_measured"]
        and base["provenance"]["template_leakage_measured"]
    )
    return base


if __name__ == "__main__":
    print(json.dumps(audit_dataset_full(), indent=2, ensure_ascii=False))
