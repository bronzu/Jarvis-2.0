"""
Self-Improvement must remain DISABLED BY DESIGN.

These tests verify that:
1. The config flag is False
2. Components cannot be activated without authorization
3. No autonomous modification of code/model/version is possible
"""

import pytest
from pathlib import Path


def test_self_improvement_config_is_disabled():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "self_improvement:" in content
    assert "enabled: false" in content, "Self-improvement must be disabled in config"


def test_autonomy_config_is_disabled():
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "autonomy:" in content
    assert "enabled: false" in content, "Autonomy must be disabled in config"


def test_self_improvement_components_exist_but_require_explicit_activation():
    from expansion_system.self_improvement.candidate_builder import CandidateBuilder
    from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator

    builder = CandidateBuilder()
    comparator = BenchmarkComparator()

    candidate = builder.build("test_change")
    assert candidate["state"] == "sandbox", "Candidate must start in sandbox state"

    result = comparator.compare(10, 11)
    assert result == "approval_required", "Better candidate requires approval, not auto-activation"


def test_autonomy_manager_requires_approval():
    from executive_system.autonomy_manager import AutonomyManager
    from INFRASTRUCTURE.security.permission_guard import PermissionGuard

    guard = PermissionGuard()
    manager = AutonomyManager(guard)

    with pytest.raises(PermissionError):
        manager.act(lambda: "unauthorized_action", approved=False)

    # Even with approval, action must be explicitly granted
    assert manager.act(lambda: "explicit_approved_action", approved=True) == "explicit_approved_action"


def test_self_improvement_cannot_modify_active_version():
    from expansion_system.self_improvement.candidate_builder import CandidateBuilder
    from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator

    builder = CandidateBuilder()
    comparator = BenchmarkComparator()

    candidate = builder.build("version_change")
    assert candidate.get("active_version") is None, "Candidate must not have access to active version"

    report = comparator.compare_versions(
        current_version="v1.0",
        candidate_version="v2.0",
        current_metrics={"logical_accuracy": 0.8, "execution_time": 1.0, "memory_usage": 100, "json_adherence": 0.9, "tool_success": 0.85},
        candidate_metrics={"logical_accuracy": 0.9, "execution_time": 0.8, "memory_usage": 80, "json_adherence": 0.95, "tool_success": 0.9},
    )
    assert report["winner"] == "v2.0"
    assert report["comparison"]["current_version"] == "v1.0"
    assert report["comparison"]["candidate_version"] == "v2.0"
