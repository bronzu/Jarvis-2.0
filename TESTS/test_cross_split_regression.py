"""
Regression tests for cross-split near-duplicate leakage.

These tests fail if ANY cross-split near-duplicate pair (Jaccard >= 0.85)
is detected by the MinHash/LSH pipeline. They serve as a hard regression
boundary for the strict normalization rule.
"""

import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("split", ["train.txt", "validation.txt", "test.txt"])
def test_dataset_split_files_exist(split):
    path = Path(f"neural_core/data/{split}")
    assert path.exists(), f"missing {path}"
    assert path.stat().st_size > 0, f"empty {path}"


def test_no_exact_line_overlap_across_splits():
    train = [l for l in Path("neural_core/data/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [l for l in Path("neural_core/data/validation.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    test = [l for l in Path("neural_core/data/test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_no_cross_split_near_duplicates_minhash():
    from neural_core.minhash_lsh import audit_dataset_near_duplicates
    r = audit_dataset_near_duplicates(
        train_path=Path("neural_core/data/train.txt"),
        val_path=Path("neural_core/data/validation.txt"),
        test_path=Path("neural_core/data/test.txt"),
    )
    assert r["cross_split_count"] == 0, (
        f"REGRESSION: {r['cross_split_count']} cross-split near-duplicate pairs detected. "
        f"Sample: {r.get('cross_split_pairs', [])[:3]}"
    )


def test_no_within_split_sentence_duplicates():
    from collections import Counter
    for label in ("train", "validation", "test"):
        text = Path(f"neural_core/data/{label}.txt").read_text(encoding="utf-8")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 5]
        c = Counter(sents)
        dups = {k: v for k, v in c.items() if v > 1}
        assert not dups, f"within-{label} duplicates: {len(dups)}"


def test_strict_normalize_collapses_punctuation_variants():
    from neural_core.build_dataset import _strict_normalize
    pairs = [
        ("— Un momento !", "Un momento !."),
        ("lo sapete ?", "Lo sapete ?"),
        ("CAPITOLO VI .", "CAPITOLO Vi ."),
    ]
    for a, b in pairs:
        assert _strict_normalize(a) == _strict_normalize(b), f"normalize failed for {a!r} vs {b!r}"


def test_pre_split_normalize_handles_apostrophes():
    from neural_core.build_dataset import _pre_split_normalize
    s = _pre_split_normalize("d' amore e d' accordo .")
    assert "'" in s
    assert s == "d'amore e d'accordo."


def test_strict_normalize_preserves_different_sentences():
    from neural_core.build_dataset import _strict_normalize
    pairs = [
        ("Il cane corre", "Il gatto corre"),
        ("Ciao Mario", "Ciao Maria"),
        ("Buongiorno", "Buonasera"),
    ]
    for a, b in pairs:
        assert _strict_normalize(a) != _strict_normalize(b), f"false positive: {a!r} vs {b!r}"


def test_pre_split_normalize_collapses_apostrophe_spacing():
    from neural_core.build_dataset import _pre_split_normalize
    s = _pre_split_normalize("d' amore e d' accordo .")
    assert s == "d'amore e d'accordo."
