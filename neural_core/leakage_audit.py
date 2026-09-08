"""
Comprehensive leakage and integrity audit over the provenance manifest.

Measures:
  - source leakage   : sentences with same source_id across multiple splits
  - template leakage : sentences with same template_id across multiple splits
  - group leakage    : sentences with same generation_group across multiple splits
  - split integrity  : every split non-empty, no duplicate sentence_ids across splits
  - global statistics: per-split counts, per-source counts, per-template counts

Returns a structured report consumable by the dataset_audit and the
third-pass security/perf reports.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from neural_core.provenance_pipeline import PROVENANCE_FILE, _load_jsonl


def _index_by(records: List[dict], key: str) -> Dict[str, List[dict]]:
    idx: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        idx[r[key]].append(r)
    return idx


def audit_leakage(path: Path = PROVENANCE_FILE) -> Dict[str, object]:
    records = _load_jsonl(path)
    splits = {"train", "validation", "test"}

    by_source = _index_by(records, "source_id")
    by_template = _index_by(records, "template_id")
    by_group = _index_by(records, "generation_group")
    by_split: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_split[r.get("split") or "unassigned"].append(r)

    sent_ids_per_split: Dict[str, set] = {s: {r["sentence_id"] for r in by_split[s]} for s in splits}

    def _leakage(by_key: Dict[str, List[dict]], label: str) -> Dict[str, object]:
        leaked = {}
        per_split_presence = defaultdict(lambda: defaultdict(int))
        for k, recs in by_key.items():
            splits_for_key = {r.get("split") for r in recs}
            splits_for_key.discard(None)
            for s in splits_for_key:
                per_split_presence[s][k] += 1
            if len(splits_for_key) > 1:
                leaked[k] = sorted(splits_for_key)
        return {
            "label": label,
            "total_keys": len(by_key),
            "leaked_keys": len(leaked),
            "leakage_ratio": len(leaked) / max(1, len(by_key)),
            "examples": list(leaked.items())[:10],
        }

    source_leakage = _leakage(by_source, "source_id")
    template_leakage = _leakage(by_template, "template_id")
    group_leakage = _leakage(by_group, "generation_group")

    overlaps = {
        f"{a}_vs_{b}": len(sent_ids_per_split[a] & sent_ids_per_split[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }

    return {
        "total_records": len(records),
        "per_split_counts": {s: len(by_split[s]) for s in splits},
        "overlap_sentence_ids": overlaps,
        "source_leakage": source_leakage,
        "template_leakage": template_leakage,
        "group_leakage": group_leakage,
        "split_integrity": {
            "all_three_splits_non_empty": all(len(by_split[s]) > 0 for s in splits),
            "no_sentence_id_overlap": all(v == 0 for v in overlaps.values()),
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit_leakage(), indent=2, ensure_ascii=False))
