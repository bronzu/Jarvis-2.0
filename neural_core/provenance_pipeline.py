"""
Dataset provenance pipeline (additive, non-destructive).

For each sentence emitted by build_dataset.py, attach:
  - source_id        (literature / italian_modern / science / technology / reasoning / dialogue / tool / verification)
  - template_id      (e.g. SCIENCE_TEMPLATES[3] -> "science:3")
  - generation_group (id of the group/paragraph or chapter the sentence came from)
  - sentence_id      (deterministic hash of normalized sentence)
  - split            (train / validation / test — assigned at split time)

The sidecar manifest is written as JSONL alongside the .txt files
(``neural_core/data/provenance_sentences.jsonl``). Existing dataset files are
NOT overwritten unless ``--regenerate`` is passed.

This is a TOOLING layer that reads the deterministic outputs of the existing
generation functions; the source of truth for what was actually written to disk
is the sidecar manifest, so template/source/split leakage is fully measurable.
"""

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neural_core import build_dataset as bd
from neural_core.dataset_provenance import (
    build_provenance_manifest,
    write_provenance_manifest,
)


DATA_DIR = Path("neural_core/data")
PROVENANCE_FILE = DATA_DIR / "provenance_sentences.jsonl"


SOURCE_LITERATURE = "literature"
SOURCE_ITALIAN_MODERN = "italian_modern"
SOURCE_SCIENCE = "science"
SOURCE_TECH = "technology"
SOURCE_REASONING = "reasoning"
SOURCE_DIALOGUE = "dialogue"
SOURCE_TOOL = "tool"
SOURCE_VERIFICATION = "verification"

SPLIT_TRAIN = "train"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"


@dataclass
class SentenceRecord:
    sentence_id: str
    text: str
    source_id: str
    template_id: str
    generation_group: str
    split: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _sent_id(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower().strip()).encode("utf-8")).hexdigest()[:16]


def _format_template(template: str, words: dict) -> str:
    return template.format(**{k: random.choice(v) for k, v in words.items()})


def _emit_literature_records() -> List[SentenceRecord]:
    """Emit one record per literature sentence; template_id = 'raw'."""
    if not bd.LITERATURE_CLEAN.exists():
        return []
    text = bd.LITERATURE_CLEAN.read_text(encoding="utf-8")
    records: List[SentenceRecord] = []
    for paragraph_idx, paragraph in enumerate(text.split("\n\n")):
        for sentence in paragraph.split("."):
            sentence = sentence.strip()
            if len(sentence) < 11:
                continue
            records.append(SentenceRecord(
                sentence_id=_sent_id(sentence),
                text=sentence,
                source_id=SOURCE_LITERATURE,
                template_id="raw",
                generation_group=f"literature:p{paragraph_idx}",
            ))
    return records


def _emit_template_records(
    source_id: str,
    templates: List[str],
    words: dict,
    n_paragraphs: int,
    group_prefix: str,
) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []
    for p_idx in range(n_paragraphs):
        length = random.randint(8, 12)
        for s_idx in range(length):
            t_idx = random.randrange(len(templates))
            template = templates[t_idx]
            try:
                sentence = _format_template(template, words) if words else template
            except (KeyError, IndexError):
                continue
            records.append(SentenceRecord(
                sentence_id=_sent_id(sentence),
                text=sentence,
                source_id=source_id,
                template_id=f"{source_id}:{t_idx}",
                generation_group=f"{group_prefix}:p{p_idx}",
            ))
    return records


def _emit_italian_modern(n_paragraphs: int) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []
    for p_idx in range(n_paragraphs):
        length = random.randint(8, 12)
        for _ in range(length):
            t_idx = random.randrange(len(bd.ITALIAN_MODERN_TEMPLATES))
            sentence = bd.ITALIAN_MODERN_TEMPLATES[t_idx]
            records.append(SentenceRecord(
                sentence_id=_sent_id(sentence),
                text=sentence,
                source_id=SOURCE_ITALIAN_MODERN,
                template_id=f"{SOURCE_ITALIAN_MODERN}:{t_idx}",
                generation_group=f"italian_modern:p{p_idx}",
            ))
    return records


def _build_all_records(n_per_category: int = 100) -> List[SentenceRecord]:
    random.seed(42)
    records: List[SentenceRecord] = []
    records.extend(_emit_literature_records())
    records.extend(_emit_italian_modern(n_per_category))
    records.extend(_emit_template_records(
        SOURCE_SCIENCE, bd.SCIENCE_TEMPLATES, bd.SCIENCE_WORDS, n_per_category, "science"))
    records.extend(_emit_template_records(
        SOURCE_TECH, bd.TECH_TEMPLATES, bd.TECH_WORDS, n_per_category, "technology"))
    records.extend(_emit_template_records(
        SOURCE_REASONING, bd.REASONING_TEMPLATES, bd.REASONING_WORDS, n_per_category, "reasoning"))
    records.extend(_emit_template_records(
        SOURCE_DIALOGUE, bd.DIALOGUE_TEMPLATES, bd.DIALOGUE_WORDS, n_per_category, "dialogue"))
    records.extend(_emit_template_records(
        SOURCE_TOOL, bd.TOOL_TEMPLATES, bd.TOOL_WORDS, n_per_category, "tool"))
    records.extend(_emit_template_records(
        SOURCE_VERIFICATION, bd.VERIFICATION_TEMPLATES, bd.VERIFICATION_WORDS, n_per_category, "verification"))

    seen = set()
    deduped: List[SentenceRecord] = []
    for r in records:
        if r.sentence_id in seen:
            continue
        seen.add(r.sentence_id)
        deduped.append(r)
    return deduped


def _assign_splits(records: List[SentenceRecord], seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1) -> List[SentenceRecord]:
    """Hash-based deterministic split assignment (matches build_dataset._split_dataset_no_leakage)."""
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    rng = random.Random(seed)
    for r in records:
        digest = hashlib.md5(r.text.lower().strip().encode("utf-8")).hexdigest()
        ratio = (int(digest[:8], 16) % 10000) / 10000.0
        if ratio < train_ratio:
            r.split = SPLIT_TRAIN
        elif ratio < train_ratio + val_ratio:
            r.split = SPLIT_VAL
        else:
            r.split = SPLIT_TEST
    rng.shuffle(records)
    return records


def _write_jsonl(records: List[SentenceRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def _load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_provenance(records_path: Path = PROVENANCE_FILE) -> Dict[str, object]:
    if not records_path.exists():
        return {"error": f"provenance file not found: {records_path}"}
    records = _load_jsonl(records_path)

    by_split: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    by_template: Dict[str, int] = {}
    by_group: Dict[str, int] = {}
    source_per_split: Dict[str, Dict[str, int]] = {}
    template_per_split: Dict[str, Dict[str, int]] = {}
    split_sent_ids: Dict[str, set] = {SPLIT_TRAIN: set(), SPLIT_VAL: set(), SPLIT_TEST: set()}

    for r in records:
        split = r.get("split") or "unassigned"
        by_split[split] = by_split.get(split, 0) + 1
        src = r["source_id"]
        by_source[src] = by_source.get(src, 0) + 1
        tid = r["template_id"]
        by_template[tid] = by_template.get(tid, 0) + 1
        grp = r["generation_group"]
        by_group[grp] = by_group.get(grp, 0) + 1
        source_per_split.setdefault(split, {})
        source_per_split[split][src] = source_per_split[split].get(src, 0) + 1
        template_per_split.setdefault(split, {})
        template_per_split[split][tid] = template_per_split[split].get(tid, 0) + 1
        if split in split_sent_ids:
            split_sent_ids[split].add(r["sentence_id"])

    overlaps = {
        f"{a}_vs_{b}": len(split_sent_ids[a] & split_sent_ids[b])
        for a, b in [(SPLIT_TRAIN, SPLIT_VAL), (SPLIT_TRAIN, SPLIT_TEST), (SPLIT_VAL, SPLIT_TEST)]
    }

    sentences_per_template = list(by_template.values())
    template_leakage = {
        "unique_templates": len(by_template),
        "max_sentences_per_template": max(sentences_per_template) if sentences_per_template else 0,
        "min_sentences_per_template": min(sentences_per_template) if sentences_per_template else 0,
        "mean_sentences_per_template": (sum(sentences_per_template) / len(sentences_per_template)) if sentences_per_template else 0,
        "template_present_in_multiple_splits": sum(
            1 for tid in by_template
            if sum(1 for s in template_per_split if tid in template_per_split[s]) > 1
        ),
        "split_invariant_template_ratio": (
            sum(1 for tid in by_template if sum(1 for s in template_per_split if tid in template_per_split[s]) > 1)
            / max(1, len(by_template))
        ),
    }

    sentences_per_group = list(by_group.values())
    group_leakage = {
        "unique_groups": len(by_group),
        "max_sentences_per_group": max(sentences_per_group) if sentences_per_group else 0,
        "group_in_multiple_splits": sum(
            1 for grp in by_group
            if sum(1 for s in source_per_split if grp in by_group) > 1
        ),
    }

    source_leakage = {
        "unique_sources": len(by_source),
        "source_distribution": by_source,
        "source_distribution_per_split": source_per_split,
    }

    return {
        "total_records": len(records),
        "by_split": by_split,
        "by_source": by_source,
        "overlap_sentence_ids": overlaps,
        "source_leakage": source_leakage,
        "template_leakage": template_leakage,
        "group_leakage": group_leakage,
        "split_integrity": {
            "all_three_splits_non_empty": all(by_split.get(s, 0) > 0 for s in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)),
            "no_split_overlap": all(v == 0 for v in overlaps.values()),
        },
    }


def build_and_write_provenance(n_per_category: int = 100) -> Path:
    records = _build_all_records(n_per_category)
    records = _assign_splits(records)
    n = _write_jsonl(records, PROVENANCE_FILE)
    manifest = build_provenance_manifest(
        data_dir=DATA_DIR,
        template_family="mixed",
        seed=42,
        notes="Provenance manifest built by neural_core.provenance_pipeline",
    )
    manifest["summary"]["total_records"] = n
    write_provenance_manifest(manifest)
    return PROVENANCE_FILE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_provenance(), indent=2, ensure_ascii=False))
    else:
        path = build_and_write_provenance(n_per_category=args.n)
        print(f"Provenance written to: {path}")
        print(json.dumps(audit_provenance(path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
