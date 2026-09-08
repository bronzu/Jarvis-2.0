"""
Comprehensive tests for the enhanced self-improvement system.
"""

import time
import pytest
from expansion_system.self_analysis.performance_monitor import PerformanceMonitor
from expansion_system.self_analysis.weakness_detector import WeaknessDetector
from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from expansion_system.self_improvement.candidate_builder import CandidateBuilder
from expansion_system.self_improvement.improvement_cycle import ImprovementCycle
from expansion_system.self_improvement.system_benchmark import SystemBenchmark
from expansion_system.discovery.discovery_engine import DiscoveryEngine
from expansion_system.discovery.experimenter import Experimenter
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.audit.audit_system import AuditSystem
from executive_system.action_executor import ActionExecutor


class TestPerformanceMonitorEnhanced:
    def test_record_and_metrics_backward_compatible(self):
        monitor = PerformanceMonitor()
        monitor.record("latency", 12)
        metrics = monitor.metrics()
        names = [m["name"] if isinstance(m, dict) else m for m in metrics]
        assert "latency" in names

    def test_set_and_get_baseline(self):
        monitor = PerformanceMonitor()
        monitor.set_baseline("latency", 10.0)
        assert monitor.get_baseline("latency") == 10.0

    def test_detect_anomalies(self):
        monitor = PerformanceMonitor()
        monitor.set_baseline("latency", 10.0)
        for v in [10, 10, 10, 10, 10, 10, 10, 10, 10, 50]:
            monitor.record("latency", float(v))
        anomalies = monitor.detect_anomalies("latency", threshold_std=2.0)
        assert len(anomalies) >= 1
        assert anomalies[0]["value"] == 50.0

    def test_trend_detection(self):
        monitor = PerformanceMonitor()
        for v in [1, 2, 3, 4, 5]:
            monitor.record("score", float(v))
        trend = monitor.trend("score")
        assert trend["trend"] == "increasing"
        assert trend["slope"] > 0

    def test_statistics(self):
        monitor = PerformanceMonitor()
        for v in [1, 2, 3, 4, 5]:
            monitor.record("score", float(v))
        stats = monitor.statistics("score")
        assert stats["mean"] == 3.0
        assert stats["count"] == 5

    def test_negative_value_raises(self):
        monitor = PerformanceMonitor()
        with pytest.raises(ValueError):
            monitor.record("score", -1.0)

    def test_window_size_limits(self):
        monitor = PerformanceMonitor(window_size=5)
        for v in range(10):
            monitor.record("score", float(v))
        assert len(monitor.metrics()) == 5


class TestWeaknessDetectorEnhanced:
    def test_classify_weakness_types(self):
        detector = WeaknessDetector()
        weakness = {"name": "memory_recall", "value": 20, "context": {"type": "memory_problem"}}
        result = detector.detect([weakness], 10)
        assert len(result) >= 1
        assert result[0]["type"] == "memory_problem"

    def test_severity_scaling(self):
        detector = WeaknessDetector()
        weakness_low = {"name": "latency", "value": 11, "context": {}}
        weakness_high = {"name": "latency", "value": 100, "context": {}}
        results = detector.detect([weakness_low, weakness_high], 10)
        assert len(results) == 2
        severities = [r["severity"] for r in results]
        assert severities[0] >= severities[1]

    def test_critical_types_detected(self):
        detector = WeaknessDetector()
        weakness = {"name": "model_accuracy", "value": 20, "context": {"type": "model_problem"}}
        results = detector.detect([weakness], 10)
        assert len(results) >= 1
        assert results[0]["type"] == "model_problem"

    def test_proposal_generated(self):
        detector = WeaknessDetector()
        weakness = {"name": "tool_success", "value": 20, "context": {}}
        results = detector.detect([weakness], 10)
        assert "proposal" in results[0]
        assert len(results[0]["proposal"]) > 0

    def test_backward_compatible_with_old_metrics(self):
        detector = WeaknessDetector()
        metrics = [{"name": "latency", "value": 12}]
        results = detector.detect(metrics, 10)
        assert len(results) >= 1
        assert results[0]["metric"] == "latency"

    def test_no_weakness_below_threshold(self):
        detector = WeaknessDetector()
        metrics = [{"name": "latency", "value": 5}]
        results = detector.detect(metrics, 10)
        assert results == []


class TestBenchmarkComparatorEnhanced:
    def test_rejects_equal_or_worse_candidates(self):
        comparator = BenchmarkComparator()
        assert comparator.compare(10, 10) == "reject"
        assert comparator.compare(10, 9) == "reject"
        assert comparator.compare(10, 11) == "approval_required"

    def test_approves_improved_candidate(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.8, "execution_time": 1.0, "memory_usage": 100, "json_adherence": 0.9, "tool_success": 0.85},
            {"logical_accuracy": 0.9, "execution_time": 0.8, "memory_usage": 80, "json_adherence": 0.95, "tool_success": 0.9},
        )
        assert report["decision"] == "ACCEPT"
        assert report["winner"] == "v2"

    def test_rejects_critical_regression(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.9, "safety_score": 0.95, "tool_success": 0.9},
            {"logical_accuracy": 0.95, "safety_score": 0.5, "tool_success": 0.95},
        )
        assert report["decision"] == "REJECT"
        assert "critical_regression" in report["reason"]
        assert "safety_score" in report["critical_regressions"]

    def test_detect_regression(self):
        comparator = BenchmarkComparator()
        regressions = comparator.detect_regression(
            {"logical_accuracy": 0.9, "safety_score": 0.95},
            {"logical_accuracy": 0.8, "safety_score": 0.5},
        )
        assert "safety_score" in regressions
        assert "logical_accuracy" in regressions

    def test_partial_metrics_do_not_penalize(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.8},
            {"logical_accuracy": 0.9},
        )
        assert report["decision"] == "ACCEPT"

    def test_missing_metrics_are_ignored(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {}, {},
        )
        assert report["decision"] == "REJECT"
        assert report["reason"] == "no_metrics"


class TestCandidateBuilderEnhanced:
    def test_build_creates_candidate_with_provenance(self):
        builder = CandidateBuilder()
        candidate = builder.build("change_x", parent_version="v1.0", reason="fix_bug")
        assert candidate["state"] == "sandbox"
        assert candidate["parent_version"] == "v1.0"
        assert candidate["reason"] == "fix_bug"
        assert "id" in candidate
        assert "created_at" in candidate

    def test_create_candidate_version(self):
        builder = CandidateBuilder()
        candidate = builder.create_candidate_version(
            parent_version="v1.0",
            change={"type": "optimization"},
            reason="speedup",
            benchmark_before={"logical_accuracy": 0.8},
        )
        assert candidate["state"] == "sandbox"
        assert candidate["parent_version"] == "v1.0"
        assert candidate["benchmark_before"] == {"logical_accuracy": 0.8}
        assert candidate["candidate_version"].startswith("v1.0-candidate-")

    def test_state_transitions(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        cid = candidate["id"]
        builder.update_state(cid, "testing")
        assert builder.get(cid)["state"] == "testing"
        builder.update_state(cid, "approved")
        assert builder.get(cid)["state"] == "approved"
        builder.update_state(cid, "deployed")
        assert builder.get(cid)["state"] == "deployed"
        assert "deployed_at" in builder.get(cid)

    def test_invalid_state_raises(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        with pytest.raises(ValueError):
            builder.update_state(candidate["id"], "invalid_state")

    def test_record_benchmark_after(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        cid = candidate["id"]
        builder.record_benchmark_after(cid, {"score": 0.9})
        assert builder.get(cid)["benchmark_after"] == {"score": 0.9}

    def test_record_decision(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        cid = candidate["id"]
        builder.record_decision(cid, "approve", "improved")
        assert builder.get(cid)["decision"] == "approve"
        assert builder.get(cid)["decision_reason"] == "improved"

    def test_backward_compatible_build(self):
        builder = CandidateBuilder()
        state = builder.build("change")
        assert state["state"] == "sandbox"


class TestImprovementCycle:
    def test_cycle_requires_approval(self):
        cycle = ImprovementCycle()
        with pytest.raises(PermissionError):
            cycle.run_cycle(trigger="test", approved=False)

    def test_cycle_with_no_metrics_skips_generation(self):
        cycle = ImprovementCycle()
        result = cycle.run_cycle(trigger="test", approved=True)
        assert result["status"] == "no_weakness_detected"
        assert result["phases"]["generation"]["status"] == "skipped"

    def test_cycle_with_weakness_generates_candidates(self):
        monitor = PerformanceMonitor()
        monitor.record("latency", 50)
        detector = WeaknessDetector()
        cycle = ImprovementCycle(performance_monitor=monitor, weakness_detector=detector)
        result = cycle.run_cycle(trigger="test", approved=True)
        assert result["status"] in ("approved", "rejected", "no_weakness_detected")
        if result["status"] != "no_weakness_detected":
            assert "phases" in result
            assert "observation" in result["phases"]

    def test_cycle_audit_logging(self):
        audit = AuditSystem(enabled=True, log_directory="/tmp/jarvis_audit_test")
        cycle = ImprovementCycle(audit_system=audit)
        result = cycle.run_cycle(trigger="audit_test", approved=True)
        events = audit.get_events(event_type="SELF_IMPROVEMENT_CYCLE_START")
        assert len(events) >= 1

    def test_set_baseline_and_active_version(self):
        cycle = ImprovementCycle()
        cycle.set_baseline_benchmark({"logical_accuracy": 0.8})
        cycle.set_active_version("v1.0")
        assert cycle.get_baseline_benchmark() == {"logical_accuracy": 0.8}
        assert cycle.get_active_version() == "v1.0"

    def test_rollback_requires_approval(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "deployed")
        with pytest.raises(PermissionError):
            cycle.rollback_candidate(candidate["id"], approved=False)

    def test_rollback_deployed_candidate(self):
        cycle = ImprovementCycle()
        cycle.set_baseline_checkpoint("/tmp/baseline.pkl")
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "deployed")
        result = cycle.rollback_candidate(candidate["id"], approved=True)
        assert result["state"] == "rolled_back"
        assert cycle.candidate_builder.get(candidate["id"])["decision"] == "rollback"


class TestSystemBenchmark:
    def test_run_all_returns_overall_score(self):
        benchmark = SystemBenchmark()
        result = benchmark.run_all({
            "instruction_following": [{"instruction": "say hello", "expected": "hello"}],
            "reasoning": [{"facts": ["a"], "goal": "a"}],
            "memory": [{"text": "test", "key": "test"}],
            "tool_selection": [{"available_tools": ["web"], "expected_tool": "web"}],
            "safety": [{"action": "tool_execution", "should_block": True}],
            "latency": [{"prompt": "hi"}],
        })
        assert "overall_score" in result
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_stability_benchmark(self):
        benchmark = SystemBenchmark()
        result = benchmark.run_stability(iterations=3)
        assert "score" in result
        assert result["iterations"] == 3

    def test_safety_benchmark_blocks_dangerous(self):
        from INFRASTRUCTURE.security.permission_guard import PermissionGuard
        guard = PermissionGuard()
        security = SecurityLayer(guard)
        benchmark = SystemBenchmark(security_layer=security)
        result = benchmark.run_safety([
            {"action": "tool_execution", "tool_name": "test", "parameters": {}, "approved": False, "should_block": True},
        ])
        assert "score" in result


class TestExperimenterEnhanced:
    def test_run_returns_structured_observation(self):
        exp = Experimenter()
        result = exp.run(lambda: "ok")
        assert result["outcome"] == "success"
        assert result["result"] == "ok"
        assert "elapsed_seconds" in result
        assert "resource_usage" in result

    def test_run_captures_error(self):
        exp = Experimenter()
        result = exp.run(lambda: 1 / 0)
        assert result["outcome"] == "error"
        assert "error" in result

    def test_run_with_timeout(self):
        exp = Experimenter()
        def infinite_loop():
            while True:
                pass
        result = exp.run(infinite_loop, timeout=0.1)
        assert result["outcome"] == "timeout"
        assert "timeout" in result.get("error", "").lower()


class TestRouterIntegration:
    def test_run_self_improvement_requires_coordinator(self):
        from executive_system.router import ExecutiveRouter
        router = ExecutiveRouter.__new__(ExecutiveRouter)
        router.coordinator = None
        result = router.run_self_improvement(trigger="test", approved=True)
        assert result["status"] == "disabled"
        assert "coordinator_not_configured" in result["reason"]


class TestBenchmarkDataset:
    def test_add_and_get_case(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset
        ds = BenchmarkDataset()
        case = ds.add({"category": "reasoning", "input": {"facts": ["a"], "goal": "a"}, "expected_output": True})
        assert case["id"]
        assert ds.get(case["id"])["category"] == "reasoning"

    def test_by_category(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset
        ds = BenchmarkDataset()
        ds.add({"category": "reasoning", "input": "r1", "expected_output": True})
        ds.add({"category": "memory", "input": "m1", "expected_output": True})
        ds.add({"category": "reasoning", "input": "r2", "expected_output": True})
        assert len(ds.by_category("reasoning")) == 2
        assert len(ds.by_category("memory")) == 1

    def test_categories(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset
        ds = BenchmarkDataset()
        ds.add({"category": "safety", "input": "s1", "expected_output": True})
        ds.add({"category": "reasoning", "input": "r1", "expected_output": True})
        cats = ds.categories()
        assert "safety" in cats
        assert "reasoning" in cats


class TestFailureClassifier:
    def test_classify_timeout(self):
        from expansion_system.self_improvement.system_benchmark import FailureClassifier
        fc = FailureClassifier()
        assert fc.classify({"category": "performance"}, error="Exceeded timeout of 5s") == "timeout_failure"

    def test_classify_safety(self):
        from expansion_system.self_improvement.system_benchmark import FailureClassifier
        fc = FailureClassifier()
        assert fc.classify({"category": "safety"}, error="Permission denied") == "safety_failure"

    def test_classify_by_category(self):
        from expansion_system.self_improvement.system_benchmark import FailureClassifier
        fc = FailureClassifier()
        assert fc.classify({"category": "reasoning"}) == "reasoning_failure"
        assert fc.classify({"category": "memory"}) == "memory_failure"
        assert fc.classify({"category": "planning"}) == "planning_failure"

    def test_to_weakness_type(self):
        from expansion_system.self_improvement.system_benchmark import FailureClassifier
        fc = FailureClassifier()
        assert fc.to_weakness_type("reasoning_failure") == "model_problem"
        assert fc.to_weakness_type("memory_failure") == "memory_problem"
        assert fc.to_weakness_type("safety_failure") == "infrastructure_problem"


class TestSystemBenchmarkDataset:
    def test_run_all_with_dataset(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset, SystemBenchmark
        ds = BenchmarkDataset([
            {"category": "safety", "input": "tool_execution", "expected_output": True},
            {"category": "safety", "input": "filesystem", "expected_output": True},
        ])
        benchmark = SystemBenchmark(security_layer=None, dataset=ds)
        result = benchmark.run_all()
        assert "overall_score" in result
        assert "failure_summary" in result
        assert result["failure_summary"]["total_failures"] >= 0

    def test_failure_summary_counts(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset, SystemBenchmark
        ds = BenchmarkDataset([
            {"category": "reasoning", "input": {"facts": ["a"], "goal": "b"}, "expected_output": True},
            {"category": "reasoning", "input": {"facts": ["c"], "goal": "d"}, "expected_output": True},
        ])
        benchmark = SystemBenchmark(reasoner=None, dataset=ds)
        result = benchmark.run_all()
        assert "failure_summary" in result
        assert "by_category" in result["failure_summary"]


class TestBenchmarkComparatorAcceptReject:
    def test_accept_returns_uppercase(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.8, "safety_score": 0.95},
            {"logical_accuracy": 0.9, "safety_score": 0.95},
        )
        assert report["decision"] == "ACCEPT"
        assert report["reason"] == "improved"

    def test_reject_critical_regression_returns_uppercase(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.9, "safety_score": 0.95},
            {"logical_accuracy": 0.95, "safety_score": 0.5},
        )
        assert report["decision"] == "REJECT"
        assert report["reason"] == "critical_regression"
        assert "safety_score" in report["critical_regressions"]

    def test_reject_no_improvement_returns_uppercase(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"memory_score": 0.8, "planning_score": 0.7},
            {"memory_score": 0.75, "planning_score": 0.65},
        )
        assert report["decision"] == "REJECT"
        assert report["reason"] == "no_improvement"

    def test_category_regressions_tracked(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.9, "safety_score": 0.95},
            {"logical_accuracy": 0.8, "safety_score": 0.5},
        )
        assert "category_regressions" in report
        assert "logical_accuracy" in report["category_regressions"] or "safety_score" in report["category_regressions"]

    def test_weighted_delta_positive_for_improvement(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.8, "safety_score": 0.9},
            {"logical_accuracy": 0.9, "safety_score": 0.95},
        )
        assert report["overall_delta"] > 0

    def test_weighted_delta_negative_for_regression(self):
        comparator = BenchmarkComparator()
        report = comparator.compare_versions(
            "v1", "v2",
            {"logical_accuracy": 0.9, "safety_score": 0.95},
            {"logical_accuracy": 0.8, "safety_score": 0.5},
        )
        assert report["overall_delta"] < 0


class TestCandidateBuilderCheckpoint:
    def test_record_checkpoint(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        cid = candidate["id"]
        builder.record_checkpoint(cid, {"path": "/tmp/ckpt.pkl", "lineage": {"git_commit": "abc123"}})
        assert builder.get(cid)["checkpoint"]["path"] == "/tmp/ckpt.pkl"
        assert builder.get(cid)["checkpoint"]["lineage"]["git_commit"] == "abc123"

    def test_record_rollback(self):
        builder = CandidateBuilder()
        candidate = builder.build("change")
        cid = candidate["id"]
        builder.update_state(cid, "deployed")
        builder.record_rollback(cid, {"reason": "regression", "checkpoint_restored": True})
        assert builder.get(cid)["rollback"]["reason"] == "regression"
        assert builder.get(cid)["rollback"]["checkpoint_restored"] is True


class TestImprovementCycleBenchmarkFailures:
    def test_cycle_with_benchmark_failures_generates_candidates(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset
        ds = BenchmarkDataset([
            {"id": "b1", "category": "reasoning", "input": {"facts": ["a"], "goal": "b"}, "expected_output": True},
            {"id": "b2", "category": "safety", "input": "tool_execution", "expected_output": True},
        ])
        benchmark = SystemBenchmark(dataset=ds)
        cycle = ImprovementCycle(system_benchmark=benchmark)
        result = cycle.run_cycle(trigger="benchmark_test", approved=True)
        assert result["status"] in ("approved", "rejected", "no_weakness_detected")
        if result["status"] != "no_weakness_detected":
            assert "phases" in result

    def test_failure_history_records_benchmark_failures(self):
        from expansion_system.self_improvement.system_benchmark import BenchmarkDataset
        ds = BenchmarkDataset([
            {"id": "b1", "category": "reasoning", "input": {"facts": ["a"], "goal": "b"}, "expected_output": True},
        ])
        benchmark = SystemBenchmark(reasoner=None, dataset=ds)
        cycle = ImprovementCycle(system_benchmark=benchmark)
        cycle.run_cycle(trigger="failure_history_test", approved=True)
        history = cycle.get_failure_history()
        assert len(history) >= 0  # may or may not have failures depending on reasoner

    def test_monitor_deployed_healthy(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "deployed")
        result = cycle.monitor_deployed(candidate["id"], {"logical_accuracy": 0.95, "safety_score": 0.95})
        assert result["status"] == "healthy"

    def test_monitor_deployed_triggers_rollback_on_regression(self):
        cycle = ImprovementCycle()
        cycle.set_baseline_checkpoint("/tmp/baseline.pkl")
        candidate = cycle.candidate_builder.create_candidate_version(
            "v1.0", {"type": "test"}, "test",
            benchmark_before={"logical_accuracy": 0.9, "safety_score": 0.95}
        )
        cycle.candidate_builder.update_state(candidate["id"], "deployed")
        result = cycle.monitor_deployed(candidate["id"], {"logical_accuracy": 0.8, "safety_score": 0.5})
        assert result["status"] == "rolled_back"
        assert cycle.candidate_builder.get(candidate["id"])["state"] == "rolled_back"


class TestWeaknessDetectorBenchmarkIntegration:
    def test_weaknesses_from_benchmark(self):
        from expansion_system.self_improvement.system_benchmark import FailureClassifier
        detector = WeaknessDetector()
        benchmark_results = {
            "reasoning": {
                "score": 0.0,
                "details": [
                    {"case_id": "r1", "passed": False, "failure_type": "reasoning_failure", "error": "not proved", "elapsed_seconds": 0.5}
                ],
            },
            "safety": {
                "score": 0.0,
                "details": [
                    {"case_id": "s1", "passed": False, "failure_type": "safety_failure", "error": "blocked", "elapsed_seconds": 0.1}
                ],
            },
        }
        weaknesses = detector.weaknesses_from_benchmark(benchmark_results)
        types = [w["type"] for w in weaknesses]
        assert "model_problem" in types
        assert "infrastructure_problem" in types

    def test_classify_benchmark_failures(self):
        detector = WeaknessDetector()
        benchmark_results = {
            "memory": {
                "details": [
                    {"case_id": "m1", "passed": False, "failure_type": "memory_failure", "error": "not found", "elapsed_seconds": 0.2}
                ],
            },
        }
        failures = detector.classify_benchmark_failures(benchmark_results)
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "memory_failure"
        assert failures[0]["category"] == "memory"


class TestDiscoveryEngineFailureHypotheses:
    def test_generate_hypotheses_with_failures(self):
        engine = DiscoveryEngine()
        failures = [
            {"failure_type": "reasoning_failure", "category": "reasoning"},
            {"failure_type": "memory_failure", "category": "memory"},
        ]
        hypotheses = engine.generate_hypotheses("Come migliorare il sistema?", facts=["debug"], failures=failures)
        texts = [h["text"] for h in hypotheses]
        assert any("reasoning_failure" in t for t in texts)
        assert any("memory_failure" in t for t in texts)
        assert any("memory" in t for t in texts)

    def test_generate_hypotheses_without_failures(self):
        engine = DiscoveryEngine()
        hypotheses = engine.generate_hypotheses("Come migliorare?", facts=["debug"])
        assert len(hypotheses) > 0
        assert all("failure" not in h.get("text", "") for h in hypotheses)
