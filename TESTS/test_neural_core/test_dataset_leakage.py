"""
Dataset leakage audit.

Verifies that train/validation/test splits have no:
- exact line overlap
- normalized line overlap
- sentence-level overlap
- duplicate lines within a split
"""

import pytest
from pathlib import Path
from collections import Counter

from neural_core.dataset import Dataset
from neural_core.tokenizer import Tokenizer


DATA_DIR = Path(__file__).resolve().parents[2] / "neural_core" / "data"


def _read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dataset_files_exist():
    for name in ["train.txt", "validation.txt", "test.txt"]:
        assert (DATA_DIR / name).exists(), f"{name} missing"


def test_dataset_no_exact_line_overlap():
    train = set(_read_lines(DATA_DIR / "train.txt"))
    val = set(_read_lines(DATA_DIR / "validation.txt"))
    test = set(_read_lines(DATA_DIR / "test.txt"))

    assert len(train & val) == 0, "Train and val have exact line overlap"
    assert len(train & test) == 0, "Train and test have exact line overlap"
    assert len(val & test) == 0, "Val and test have exact line overlap"


def test_dataset_no_normalized_line_overlap():
    def normalize(text):
        return " ".join(text.lower().split())

    train = set(normalize(line) for line in _read_lines(DATA_DIR / "train.txt"))
    val = set(normalize(line) for line in _read_lines(DATA_DIR / "validation.txt"))
    test = set(normalize(line) for line in _read_lines(DATA_DIR / "test.txt"))

    assert len(train & val) == 0, "Train and val have normalized line overlap"
    assert len(train & test) == 0, "Train and test have normalized line overlap"
    assert len(val & test) == 0, "Val and test have normalized line overlap"


def test_dataset_sentence_level_overlap_is_zero():
    """Sentence-level leakage must be zero across all splits.

    The previous audit documented template-based sentence leakage.
    This test now enforces it: any sentence-level overlap between
    train/val/test is a dataset integrity failure.
    """
    def get_sentences(text):
        return [s.strip() for s in text.split(".") if s.strip() and len(s.strip()) > 10]

    train = _read_lines(DATA_DIR / "train.txt")
    val = _read_lines(DATA_DIR / "validation.txt")
    test = _read_lines(DATA_DIR / "test.txt")

    train_sents = set()
    for line in train:
        train_sents.update(get_sentences(line))

    val_sents = set()
    for line in val:
        val_sents.update(get_sentences(line))

    test_sents = set()
    for line in test:
        test_sents.update(get_sentences(line))

    train_val_overlap = len(train_sents & val_sents)
    train_test_overlap = len(train_sents & test_sents)
    val_test_overlap = len(val_sents & test_sents)

    assert train_val_overlap == 0, f"Train-Val sentence overlap: {train_val_overlap}"
    assert train_test_overlap == 0, f"Train-Test sentence overlap: {train_test_overlap}"
    assert val_test_overlap == 0, f"Val-Test sentence overlap: {val_test_overlap}"


def test_dataset_no_duplicates_within_splits():
    for name in ["train.txt", "validation.txt", "test.txt"]:
        lines = _read_lines(DATA_DIR / name)
        counts = Counter(lines)
        duplicates = {line: count for line, count in counts.items() if count > 1}
        assert len(duplicates) == 0, f"{name} has {len(duplicates)} duplicate lines"


def test_dataset_from_text_warns_on_overlap():
    text = "alpha beta gamma delta\ngamma delta epsilon zeta\n" * 20
    tokenizer = Tokenizer()

    with pytest.warns(UserWarning, match="sovrapposizione"):
        Dataset.from_text(text, tokenizer, validation_split=0.5)


def test_dataset_deduplicate_lines_removes_exact_duplicates():
    text = "alpha\nbeta\nalpha\ngamma\nbeta"
    deduped = Dataset.deduplicate_lines(text)
    assert deduped.count("alpha") == 1
    assert deduped.count("beta") == 1
    assert deduped.count("gamma") == 1


def test_dataset_audit_split_produces_report():
    tokens = list(range(20))
    ds_a = Dataset(tokens[:10])
    ds_b = Dataset(tokens[10:])
    report = Dataset.audit_split(ds_a, ds_b, names=["train", "test"])
    assert "sizes" in report
    assert "overlap" in report
