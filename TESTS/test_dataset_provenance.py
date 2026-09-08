"""
Tests for neural_core.dataset_provenance.
"""

from pathlib import Path

from neural_core.dataset_provenance import (
    DatasetProvenance,
    build_provenance_manifest,
    detect_template_leakage,
    load_provenance_manifest,
    write_provenance_manifest,
)


def test_dataset_provenance_dataclass():
    prov = DatasetProvenance(
        name="train.txt",
        path="neural_core/data/train.txt",
        size_bytes=1234,
        md5="abc123",
        examples=10,
        template_families=["italian_modern"],
        generation_seed=42,
        notes="test",
    )
    assert prov.name == "train.txt"
    assert prov.template_families == ["italian_modern"]
    assert prov.generation_seed == 42


def test_detect_template_leakage_empty():
    result = detect_template_leakage([], ["hello world"])
    assert result["match_count"] == 0
    assert result["matches"] == []


def test_detect_template_leakage_exact_match():
    dataset = ["hello world foo bar", "another sentence here"]
    templates = ["hello world foo bar"]
    result = detect_template_leakage(dataset, templates, threshold=1.0)
    assert result["match_count"] == 1
    assert result["matches"][0]["template"] == "hello world foo bar"


def test_detect_template_leakage_below_threshold():
    dataset = ["completely different text with no overlap"]
    templates = ["hello world foo bar"]
    result = detect_template_leakage(dataset, templates, threshold=0.9)
    assert result["match_count"] == 0


def test_build_provenance_manifest_counts_examples(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (data_dir / "test.txt").write_text("lineA\nlineB\n", encoding="utf-8")

    manifest = build_provenance_manifest(data_dir=data_dir, template_family="test", seed=7)
    assert manifest["template_family"] == "test"
    assert manifest["seed"] == 7
    assert manifest["summary"]["total_examples"] == 5
    assert len(manifest["files"]) == 2


def test_write_and_load_provenance_manifest(tmp_path: Path):
    out = tmp_path / "out.json"
    manifest = {"name": "test", "values": [1, 2, 3]}
    write_provenance_manifest(manifest, output_path=out)
    loaded = load_provenance_manifest(out)
    assert loaded == manifest
