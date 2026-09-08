"""
Test per neural_core.dataset_audit.
"""

import json

from neural_core.dataset_audit import audit_dataset


def test_dataset_audit_produces_valid_report():
    report = audit_dataset()
    assert "splits" in report
    assert "duplicates" in report
    assert "near_duplicates" in report
    assert "vocabulary" in report
    assert "valid" in report
    assert "provenance" in report
    assert report["valid"] is True
    assert report["provenance"]["template_leakage_measured"] is False
    assert report["provenance"]["source_leakage_measured"] is False
    assert report["duplicates"]["exact"]["train_val"] == 0
    assert report["duplicates"]["normalized"]["train_val"] == 0
    assert report["duplicates"]["sentence"]["train_val"] == 0
    assert isinstance(report["near_duplicates"]["pairs"], list)


def test_dataset_audit_sizes_are_positive():
    report = audit_dataset()
    for split in ("train", "validation", "test"):
        stats = report["splits"][split]
        assert stats["examples"] > 0
        assert stats["token_count"] > 0
        assert stats["mean_length"] > 0
        assert stats["median_length"] > 0


def test_dataset_audit_vocabulary_is_consistent():
    report = audit_dataset()
    vocab = report["vocabulary"]
    assert vocab["size"] == vocab["max_vocab_size"]
    assert vocab["size"] > 0
