"""
Sentence-level MinHash-based deduplication for the JARVIS dataset.

Reads the existing ``train.txt`` / ``validation.txt`` / ``test.txt`` files,
identifies near-duplicate sentences across splits (and within splits) using
MinHash + LSH, and writes deduplicated versions into a sibling directory
(``neural_core/data/dedup/``). The original files are NOT modified.

This is a remediation tool that addresses the gap surfaced by
``minhash_lsh.audit_dataset_near_duplicates``: short fragments like
"— Un momento !" can leak across splits because the original
``_split_dataset_no_leakage`` only keeps sentences longer than 10 chars.
"""

import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from neural_core.minhash_lsh import (
    find_near_duplicates,
    _shingles,
    _tokenize,
    _jaccard,
)


DATA_DIR = Path("neural_core/data")
DEDUP_DIR = DATA_DIR / "dedup"
THRESHOLD = 0.85
NUM_PERM = 64
NGRAM = 3


def _load_split(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def _split_paragraphs_to_sentences(paragraphs: List[str], min_len: int = 5) -> List[str]:
    out: List[str] = []
    for p in paragraphs:
        for s in re.split(r"(?<=[.!?])\s+", p):
            s = s.strip()
            if len(s) >= min_len:
                out.append(s)
    return out


def _normalize(s: str) -> str:
    return re.sub(r"[^a-zàèéìòù\s]+", " ", s.lower()).strip()


def _paragraphs_to_text(paragraphs: List[str]) -> str:
    return "\n\n".join(paragraphs)


def _dedup_paragraphs_within(paragraphs: List[str], sentences_to_drop: set) -> List[str]:
    out: List[str] = []
    for p in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s.strip()]
        kept = [s for s in sents if _normalize(s) not in sentences_to_drop]
        if kept:
            out.append(" ".join(kept))
    return out


def run_dedup() -> Dict[str, object]:
    if not (DATA_DIR / "train.txt").exists():
        return {"error": "train.txt not found"}

    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, List[str]] = {}
    for label in ("train", "validation", "test"):
        path = DATA_DIR / f"{label}.txt"
        if not path.exists():
            continue
        splits[label] = _load_split(path)

    all_sentences: List[str] = []
    boundaries: Dict[str, Tuple[int, int]] = {}
    for label, paragraphs in splits.items():
        start = len(all_sentences)
        all_sentences.extend(_split_paragraphs_to_sentences(paragraphs))
        boundaries[label] = (start, len(all_sentences))

    report = find_near_duplicates(
        all_sentences, threshold=THRESHOLD, num_perm=NUM_PERM, ngram=NGRAM
    )

    drop_set: set = set()
    cross_split_drop: List[Tuple[str, str, float, str, str]] = []
    for i, j, sim in report["pairs"]:
        split_i = _split_of(i, boundaries)
        split_j = _split_of(j, boundaries)
        if split_i != split_j:
            keep_split = "train" if "train" in (split_i, split_j) else (
                "validation" if "validation" in (split_i, split_j) else "test"
            )
            if split_i != keep_split:
                drop_set.add(_normalize(all_sentences[i]))
            if split_j != keep_split:
                drop_set.add(_normalize(all_sentences[j]))
            cross_split_drop.append((split_i, split_j, sim, all_sentences[i][:60], all_sentences[j][:60]))
        else:
            if i < j:
                drop_set.add(_normalize(all_sentences[j]))
            else:
                drop_set.add(_normalize(all_sentences[i]))

    out_paths: Dict[str, Path] = {}
    for label, paragraphs in splits.items():
        deduped_paragraphs = _dedup_paragraphs_within(paragraphs, drop_set)
        out_path = DEDUP_DIR / f"{label}.txt"
        out_path.write_text(_paragraphs_to_text(deduped_paragraphs), encoding="utf-8")
        out_paths[label] = out_path

    return {
        "params": {
            "threshold": THRESHOLD,
            "num_perm": NUM_PERM,
            "ngram": NGRAM,
            "method": report["method"],
        },
        "totals": {
            "num_sentences_before": len(all_sentences),
            "num_sentences_dropped": len(drop_set),
            "num_verified_pairs": report["num_verified_pairs"],
            "num_cross_split_pairs": len(cross_split_drop),
        },
        "output_dir": str(DEDUP_DIR),
        "output_files": {k: str(v) for k, v in out_paths.items()},
        "cross_split_pairs_sample": cross_split_drop[:10],
    }


def _split_of(idx: int, boundaries: Dict[str, Tuple[int, int]]) -> str:
    for label, (s, e) in boundaries.items():
        if s <= idx < e:
            return label
    return "unknown"


if __name__ == "__main__":
    import json
    print(json.dumps(run_dedup(), indent=2, ensure_ascii=False))
