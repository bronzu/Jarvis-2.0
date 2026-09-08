"""
Tests for the final gap-closure modules:
  - provenance_pipeline
  - leakage_audit
  - minhash_lsh
  - sentence_dedup
  - dataset_audit_extended
  - security_audit
  - security_report_final
  - os_isolation_probe
  - final_benchmark
  - performance_regression
  - checkpoint_lineage
  - e2e_scenario (cognitive_core)
"""

import json
import time
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Provenance pipeline
# --------------------------------------------------------------------------

def test_provenance_pipeline_builds_records():
    from neural_core.provenance_pipeline import _build_all_records
    records = _build_all_records(n_per_category=5)
    assert len(records) > 0
    sample = records[0]
    assert sample.sentence_id
    assert sample.source_id
    assert sample.template_id
    assert sample.generation_group


def test_provenance_pipeline_assigns_splits():
    from neural_core.provenance_pipeline import _build_all_records, _assign_splits, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
    records = _build_all_records(n_per_category=5)
    records = _assign_splits(records)
    splits = {r.split for r in records}
    assert SPLIT_TRAIN in splits
    assert SPLIT_VAL in splits
    assert SPLIT_TEST in splits


def test_provenance_pipeline_no_duplicate_sentence_ids():
    from neural_core.provenance_pipeline import _build_all_records
    records = _build_all_records(n_per_category=5)
    ids = [r.sentence_id for r in records]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# Leakage audit
# --------------------------------------------------------------------------

def test_leakage_audit_runs_on_existing_provenance():
    from neural_core.leakage_audit import audit_leakage
    if not Path("neural_core/data/provenance_sentences.jsonl").exists():
        pytest.skip("provenance file not generated yet")
    report = audit_leakage()
    assert "total_records" in report
    assert report["split_integrity"]["all_three_splits_non_empty"] is True
    assert report["split_integrity"]["no_sentence_id_overlap"] is True


# --------------------------------------------------------------------------
# MinHash + LSH
# --------------------------------------------------------------------------

def test_minhash_signature_is_deterministic():
    from neural_core.minhash_lsh import minhash_signature
    s = {"alpha", "beta", "gamma"}
    sig1 = minhash_signature(s, num_perm=32)
    sig2 = minhash_signature(s, num_perm=32)
    assert sig1 == sig2


def test_minhash_signature_empty():
    from neural_core.minhash_lsh import minhash_signature, MAX_HASH
    sig = minhash_signature(set(), num_perm=8)
    assert sig == [MAX_HASH] * 8


def test_lsh_finds_identical_documents():
    from neural_core.minhash_lsh import find_near_duplicates
    docs = [
        "alpha beta gamma delta epsilon",
        "alpha beta gamma delta epsilon",
        "completely different text here",
    ]
    r = find_near_duplicates(docs, threshold=1.0, num_perm=64, ngram=2)
    assert r["num_verified_pairs"] >= 1


def test_minhash_lsh_no_false_positive_on_unrelated_docs():
    from neural_core.minhash_lsh import find_near_duplicates
    docs = [
        "alpha beta gamma delta epsilon zeta",
        "completely unrelated text with no shared words",
        "another independent sentence about something else",
    ]
    r = find_near_duplicates(docs, threshold=0.99, num_perm=64, ngram=2)
    assert r["num_verified_pairs"] == 0


def test_audit_dataset_near_duplicates_returns_methodology():
    from neural_core.minhash_lsh import audit_dataset_near_duplicates
    r = audit_dataset_near_duplicates(
        train_path=Path("neural_core/data/train.txt"),
        val_path=Path("neural_core/data/validation.txt"),
        test_path=Path("neural_core/data/test.txt"),
    )
    assert r["params"]["method"] == "MinHash + LSH (banding) + exact Jaccard verification"
    assert r["params"]["num_perm"] > 0
    assert r["totals"]["num_documents"] > 0


# --------------------------------------------------------------------------
# Dataset audit extended
# --------------------------------------------------------------------------

def test_dataset_audit_extended_includes_provenance_and_minhash():
    from neural_core.dataset_audit_extended import audit_dataset_full
    r = audit_dataset_full()
    assert r["provenance"]["source_leakage_measured"] is True
    assert r["provenance"]["template_leakage_measured"] is True
    assert "near_duplicates" in r
    assert r["near_duplicates"]["method"] == "MinHash + LSH (banding) + exact Jaccard verification"


# --------------------------------------------------------------------------
# Security audit
# --------------------------------------------------------------------------

def test_security_audit_runs_and_classifies():
    from neural_core.security_audit import run_security_audit
    r = run_security_audit()
    assert r["scanned_files"] > 0
    assert r["total_findings"] > 0
    assert "VULNERABILITY" in r["classification_counts"]


def test_security_audit_excludes_test_files():
    from neural_core.security_audit import run_security_audit
    r = run_security_audit()
    for f in r["vulnerabilities"]:
        assert "TESTS" not in f["file"]
        assert "/tests/" not in f["file"]


def test_security_report_final_has_no_remaining_vulnerabilities():
    from neural_core.security_report_final import final_security_report
    r = final_security_report()
    assert r["reviewed_vulnerabilities"] == []
    assert r["reviewed_classification"]["VULNERABILITY"] == 0
    assert r["unreviewed_findings"] == 0
    assert r["final_verdict"] == "PASS"


# --------------------------------------------------------------------------
# OS isolation
# --------------------------------------------------------------------------

def test_os_isolation_probe_runs():
    from neural_core.os_isolation_probe import run_isolation_probe
    r = run_isolation_probe()
    assert "container" in r
    assert "user_privileges" in r
    assert "namespaces" in r
    # os_sandbox is now True: real OS-level isolation is available
    # via neural_core.os_sandbox (sandbox-exec on macOS, rlimits + chdir
    # on Linux, both in addition to application-level guards).
    assert r["summary"]["os_sandbox"] is True
    assert r["summary"]["application_level_isolation"] is True
    # The new app_os_sandbox field must report the new mechanism available
    # and that the smoke test confirms legitimate code runs + escape blocked.
    assert "app_os_sandbox" in r
    assert r["app_os_sandbox"]["module_available"] is True
    assert r["app_os_sandbox"]["smoke_test"]["legitimate_code"]["status"] == "ok"
    assert r["app_os_sandbox"]["smoke_test"]["escape_blocked"]["block_succeeded"] is True


# --------------------------------------------------------------------------
# Final benchmark
# --------------------------------------------------------------------------

def test_final_benchmark_returns_full_report(tmp_path):
    from neural_core.final_benchmark import run_final_benchmark
    r = run_final_benchmark(tmp_path)
    assert r["hardware"]["python_version"]
    assert r["startup"]["mean"] > 0
    assert r["inference"]["mean"] > 0
    assert r["training"]["mean"] > 0
    assert r["checkpoint"]["size_bytes"] > 0
    assert "encode" in r["tokenizer"] and "decode" in r["tokenizer"]


def test_final_benchmark_stats_contain_required_fields():
    from neural_core.final_benchmark import _bench
    def fn():
        time.sleep(0.001)
    stats = _bench(fn, warmup=1, iterations=3)
    for k in ("count", "mean", "median", "min", "max", "stddev", "p95"):
        assert k in stats
        assert stats[k] >= 0


# --------------------------------------------------------------------------
# Performance regression
# --------------------------------------------------------------------------

def test_performance_regression_records_baseline(tmp_path, monkeypatch):
    from neural_core import performance_regression
    monkeypatch.setattr(performance_regression, "BASELINE_PATH", tmp_path / "baseline.json")
    fake = {
        "hardware": {"platform": "test", "python_version": "3.0", "machine": "x86_64"},
        "inference": {"mean": 0.001},
        "training": {"mean": 0.01},
    }
    r = performance_regression.reconcile(fake)
    assert r["baseline_status"] in ("BASELINE NOT TRUSTWORTHY", "FRESH BASELINE")
    assert Path(r["baseline_saved_to"]).exists()


# --------------------------------------------------------------------------
# Checkpoint lineage
# --------------------------------------------------------------------------

def test_checkpoint_lineage_returns_required_fields():
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer
    from neural_core.checkpoint_lineage import build_lineage
    text = "alpha beta gamma delta epsilon zeta eta"
    tok = Tokenizer(max_vocab_size=10, mode="whitespace"); tok.fit(text)
    model = MicroTransformer(vocab_size=len(tok.vocabulary), d_model=8, num_heads=1, d_ff=16, num_layers=1, max_len=16, seed=7)
    lineage = build_lineage(
        model=model,
        model_class="MicroTransformer",
        model_config={"d_model": 8},
        tokenizer=tok,
        seed=7,
    )
    for field_name in ("git_commit", "code_version", "architecture_version", "model_class",
                       "model_config", "tokenizer_identity", "dataset_identity",
                       "parameter_count", "seed", "python_version", "numpy_version",
                       "platform", "timestamp"):
        assert field_name in lineage


def test_checkpoint_lineage_attach_backward_compatible():
    from neural_core.checkpoint_lineage import attach_to_checkpoint
    record = {"model_name": "x", "parameters": {}}
    lineage = {"git_commit": "abc123", "python_version": "3.14"}
    out = attach_to_checkpoint(record, lineage)
    assert out["model_name"] == "x"
    assert out["parameters"] == {}
    assert out["checkpoint_metadata"]["git_commit"] == "abc123"


# --------------------------------------------------------------------------
# E2E scenario
# --------------------------------------------------------------------------

def test_e2e_scenario_happy_path():
    from cognitive_core.e2e_scenario import run_scenario
    r = run_scenario("Calcola 2 + 3 * 4")
    steps = [t["step"] for t in r["trace"]]
    assert "input" in steps
    assert "understanding" in steps
    assert "reasoning" in steps
    assert "planning" in steps
    assert "tool_selection" in steps
    assert "permission" in steps
    assert "action" in steps
    assert "world_model_update" in steps
    action = next(t for t in r["trace"] if t["step"] == "action")
    assert action["observation"]["status"] == "ok"
    assert action["observation"]["result"] == 14


def test_e2e_scenario_permission_denial():
    from cognitive_core.e2e_scenario import run_scenario
    r = run_scenario("Esegui rm -rf /tmp")
    permission = next(t for t in r["trace"] if t["step"] == "permission")
    assert permission["decision"]["approved"] is False
    action = next(t for t in r["trace"] if t["step"] == "action")
    assert action["observation"]["status"] == "denied"


def test_e2e_scenario_failure_recovery():
    """Verifies real failure -> detection -> recovery -> valid final state.

    The pipeline must:
    1. Run the action and let the tool raise a real error (not a permission denial)
    2. Detect the error in the E2E wrapper
    3. Update the world model with the failure causal relation
    4. Reach a valid final state (no exception escapes)
    """
    from cognitive_core.e2e_scenario import run_scenario
    r = run_scenario("calcol xyz")
    steps = [t["step"] for t in r["trace"]]
    assert "input" in steps and "action" in steps and "world_model_update" in steps
    action = next(t for t in r["trace"] if t["step"] == "action")
    assert action["observation"]["status"] == "error"
    assert "error" in action["observation"]
    # World model must contain a failure-aware causal relation
    world_step = next(t for t in r["trace"] if t["step"] == "world_model_update")
    assert "simulation" in world_step
    # Real state change: action_failed -> log_error was added
    assert world_step["causal_relations"]["action_failed_predicts"] == ["log_error"]
    # No exception escaped — the scenario completed end-to-end


def test_e2e_scenario_world_model_differs_by_outcome():
    """The world model must reflect the action outcome (success vs denied
    vs failure) with different causal relations — not just a fixed dict."""
    from cognitive_core.e2e_scenario import run_scenario
    r_ok = run_scenario("Calcola 5 + 5")
    r_denied = run_scenario("Esegui rm -rf /tmp")
    r_error = run_scenario("calcol xyz")

    def wm(s):
        return next(t for t in s["trace"] if t["step"] == "world_model_update")["causal_relations"]

    wm_ok = wm(r_ok)
    wm_denied = wm(r_denied)
    wm_error = wm(r_error)

    # Each outcome has its own causal edge
    assert wm_ok["action_ok_predicts"] == ["update_world_state"]
    assert wm_ok["permission_denied_predicts"] == []
    assert wm_denied["permission_denied_predicts"] == ["no_state_change"]
    assert wm_denied["action_ok_predicts"] == []
    assert wm_error["action_failed_predicts"] == ["log_error"]
    assert wm_error["action_ok_predicts"] == []
