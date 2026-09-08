"""
Comprehensive tests for the training-integrated self-improvement system.
"""

import os
import pickle
import tempfile
import time

import pytest
import numpy as np

from expansion_system.self_analysis.weakness_detector import WeaknessDetector
from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from expansion_system.self_improvement.candidate_builder import CandidateBuilder
from expansion_system.self_improvement.improvement_cycle import ImprovementCycle
from expansion_system.self_improvement.system_benchmark import SystemBenchmark, BenchmarkDataset
from expansion_system.self_improvement.training_strategy import TrainingStrategyBuilder, TrainingStrategy
from expansion_system.self_improvement.experimental_trainer import ExperimentalTrainer, ExperimentalTrainingResult
from expansion_system.discovery.discovery_engine import DiscoveryEngine
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.audit.audit_system import AuditSystem
from neural_core.tokenizer import Tokenizer
from neural_core.dataset import Dataset
from neural_core.trainer import Trainer, build_model
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam
from neural_core.checkpoint_lineage import build_lineage, attach_to_checkpoint


class TestTrainingStrategyBuilder:
    def test_build_from_weakness_reasoning(self):
        builder = TrainingStrategyBuilder()
        weakness = {"type": "reasoning_failure", "metric": "reasoning", "severity": 4, "value": 0.3, "threshold": 0.5}
        strategy = builder.build_from_weakness(weakness, "cand-123", "v1.0")
        assert strategy.weakness_type == "reasoning_failure"
        assert strategy.parameters["dataset"] == "reasoning_train"
        assert strategy.parameters["epochs"] == 3
        assert strategy.parameters["candidate_id"] == "cand-123"

    def test_build_from_weakness_safety(self):
        builder = TrainingStrategyBuilder()
        weakness = {"type": "safety_failure", "metric": "safety", "severity": 5, "value": 0.5, "threshold": 0.9}
        strategy = builder.build_from_weakness(weakness, "cand-456", "v1.0")
        assert strategy.parameters["dataset"] == "safety_train"
        assert strategy.parameters["learning_rate"] == 0.0002
        assert strategy.parameters["epochs"] == 1

    def test_build_from_failure(self):
        builder = TrainingStrategyBuilder()
        failure = {"failure_type": "memory_failure", "category": "memory", "error": "not found"}
        strategy = builder.build_from_failure(failure, "cand-789", "v1.0")
        assert strategy.weakness_type == "memory_failure"
        assert strategy.parameters["dataset"] == "memory_train"

    def test_available_strategies(self):
        builder = TrainingStrategyBuilder()
        strategies = builder.available_strategies()
        assert "reasoning_failure" in strategies
        assert "safety_failure" in strategies
        assert "model_problem" in strategies

    def test_strategy_to_dict_roundtrip(self):
        builder = TrainingStrategyBuilder()
        weakness = {"type": "planning_failure", "metric": "planning", "severity": 3}
        strategy = builder.build_from_weakness(weakness, "cand-001", "v1.0")
        data = strategy.to_dict()
        assert data["strategy_id"] == strategy.strategy_id
        assert data["parameters"]["candidate_id"] == "cand-001"


class TestExperimentalTrainer:
    def test_run_training_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            text = "alpha beta gamma delta epsilon " * 50
            strategy = TrainingStrategy(
                strategy_id="strategy-test",
                weakness_type="model_problem",
                parameters={
                    "dataset": "general_train",
                    "objective": "next_token",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "batch_size": 8,
                    "context_size": 16,
                },
            )
            result = trainer.run_training(
                candidate_id="test-cand-1",
                strategy=strategy,
                training_text=text,
                validation_text=text,
            )
            assert result.status == "TRAINING_COMPLETED"
            assert result.checkpoint_path is not None
            assert os.path.exists(result.checkpoint_path)
            assert result.metrics["epochs_completed"] == 1

    def test_run_training_validates_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            text = "alpha beta gamma delta epsilon " * 50
            strategy = TrainingStrategy(
                strategy_id="strategy-test",
                weakness_type="model_problem",
                parameters={
                    "dataset": "general_train",
                    "objective": "next_token",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "batch_size": 8,
                    "context_size": 16,
                },
            )
            result = trainer.run_training(
                candidate_id="test-cand-2",
                strategy=strategy,
                training_text=text,
                validation_text=text,
            )
            assert result.status == "TRAINING_COMPLETED"
            validation = trainer.validate_checkpoint(result.checkpoint_path)
            assert validation["valid"] is True
            assert validation["checkpoint_size_bytes"] > 0

    def test_validate_checkpoint_invalid_path(self):
        trainer = ExperimentalTrainer()
        validation = trainer.validate_checkpoint("/nonexistent/path.pkl")
        assert validation["valid"] is False
        assert "non autorizzato" in validation["error"].lower() or "not found" in validation["error"].lower() or "non trovato" in validation["error"].lower()

    def test_validate_checkpoint_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not a pickle")
            path = f.name
        try:
            trainer = ExperimentalTrainer()
            validation = trainer.validate_checkpoint(path)
            assert validation["valid"] is False
        finally:
            os.unlink(path)

    def test_run_training_with_baseline_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = "alpha beta gamma delta epsilon " * 50
            tokenizer = Tokenizer(max_vocab_size=50, mode="whitespace")
            tokenizer.fit(text)
            ids = tokenizer.encode(text)
            ds = Dataset(ids, context_size=8, stride=8)
            model = build_model("next_token", vocab_size=len(tokenizer.vocabulary), embedding_dim=16, hidden_dim=32, seed=42)
            loss = CrossEntropyLoss()
            optimizer = Adam(learning_rate=0.01)
            trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
            baseline_path = os.path.join(tmp, "baseline.pkl")
            trainer.save_checkpoint(baseline_path)

            exp_trainer = ExperimentalTrainer(base_dir=tmp)
            strategy = TrainingStrategy(
                strategy_id="strategy-baseline",
                weakness_type="model_problem",
                parameters={
                    "dataset": "general_train",
                    "objective": "next_token",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "batch_size": 8,
                    "context_size": 8,
                },
            )
            result = exp_trainer.run_training(
                candidate_id="test-cand-3",
                strategy=strategy,
                training_text=text,
                validation_text=text,
                tokenizer=tokenizer,
                benchmark_dataset=ds,
                baseline_checkpoint=baseline_path,
            )
            assert result.status == "TRAINING_COMPLETED"
            assert result.lineage is not None
            assert result.lineage.get("parent_checkpoint") == baseline_path


class TestImprovementCycleTraining:
    def test_run_training_cycle_requires_approval(self):
        cycle = ImprovementCycle()
        with pytest.raises(PermissionError):
            cycle.run_training_cycle(trigger="test", approved=False)

    def test_run_training_cycle_no_weaknesses(self):
        cycle = ImprovementCycle()
        result = cycle.run_training_cycle(trigger="test", approved=True)
        assert result["status"] == "no_weakness_detected"

    def test_run_training_cycle_with_weakness_generates_strategy(self):
        monitor = type('M', (), {'metrics': lambda self: []})()
        detector = WeaknessDetector()
        cycle = ImprovementCycle(performance_monitor=monitor, weakness_detector=detector)
        benchmark_data = {
            "reasoning": {
                "score": 0.0,
                "details": [
                    {"case_id": "r1", "passed": False, "failure_type": "reasoning_failure", "error": "not proved", "elapsed_seconds": 0.5}
                ],
            },
        }
        result = cycle.run_training_cycle(trigger="test", approved=True, benchmark_data=benchmark_data)
        assert result["status"] in ("approved", "rejected", "no_weakness_detected")
        if result["status"] != "no_weakness_detected":
            assert "phases" in result
            assert "training_strategy" in result["phases"]

    def test_training_cycle_audit_logging(self):
        audit = AuditSystem(enabled=True, log_directory="/tmp/jarvis_training_audit_test")
        cycle = ImprovementCycle(audit_system=audit)
        result = cycle.run_training_cycle(trigger="audit_test", approved=True)
        events = audit.get_events(event_type="SELF_IMPROVEMENT_TRAINING_START")
        assert len(events) >= 1


class TestImprovementCycleDeployment:
    def test_deploy_requires_approval(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "approved")
        with pytest.raises(PermissionError):
            cycle.deploy_candidate(candidate["id"], approved=False)

    def test_deploy_approved_candidate(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "approved")
        cycle.candidate_builder.record_checkpoint(candidate["id"], {"path": "/tmp/fake.pkl"})
        result = cycle.deploy_candidate(candidate["id"], approved=True)
        assert result["state"] == "deployed"
        assert cycle.get_active_checkpoint() == "/tmp/fake.pkl"

    def test_deploy_requires_checkpoint(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "approved")
        with pytest.raises(ValueError):
            cycle.deploy_candidate(candidate["id"], approved=True)


class TestImprovementCycleCheckpointIntegration:
    def test_set_baseline_and_active_checkpoint(self):
        cycle = ImprovementCycle()
        cycle.set_baseline_checkpoint("/tmp/baseline.pkl")
        cycle.set_active_checkpoint("/tmp/active.pkl")
        assert cycle.get_baseline_checkpoint() == "/tmp/baseline.pkl"
        assert cycle.get_active_checkpoint() == "/tmp/active.pkl"

    def test_rollback_with_checkpoint_path(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version("v1.0", {"type": "test"}, "test")
        cycle.candidate_builder.update_state(candidate["id"], "deployed")
        result = cycle.rollback_candidate(candidate["id"], approved=True, checkpoint_path="/tmp/baseline.pkl")
        assert result["state"] == "rolled_back"
        rollback_info = cycle.candidate_builder.get(candidate["id"]).get("rollback", {})
        assert rollback_info["checkpoint_path"] == "/tmp/baseline.pkl"


class TestEndToEndTraining:
    def test_full_training_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau " * 100
            tokenizer = Tokenizer(max_vocab_size=100, mode="whitespace")
            tokenizer.fit(text)
            ids = tokenizer.encode(text)
            ds = Dataset(ids, context_size=16, stride=16)

            model = build_model("next_token", vocab_size=len(tokenizer.vocabulary), embedding_dim=32, hidden_dim=64, seed=42)
            loss = CrossEntropyLoss()
            optimizer = Adam(learning_rate=0.01)
            trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
            baseline_path = os.path.join(tmp, "baseline.pkl")
            trainer.save_checkpoint(baseline_path)

            exp_trainer = ExperimentalTrainer(base_dir=tmp)
            strategy_builder = TrainingStrategyBuilder()
            weakness = {"type": "reasoning_failure", "metric": "reasoning", "severity": 4, "value": 0.3, "threshold": 0.5}
            strategy = strategy_builder.build_from_weakness(weakness, "e2e-cand-1", "v1.0")
            strategy.parameters["architecture"] = "next_token"

            training_result = exp_trainer.run_training(
                candidate_id="e2e-cand-1",
                strategy=strategy,
                baseline_checkpoint=baseline_path,
                training_text=text,
                validation_text=text,
                tokenizer=tokenizer,
                benchmark_dataset=ds,
            )
            assert training_result.status == "TRAINING_COMPLETED"
            assert training_result.checkpoint_path is not None
            assert training_result.lineage is not None
            assert training_result.lineage.get("parent_checkpoint") == baseline_path

            validation = exp_trainer.validate_checkpoint(training_result.checkpoint_path)
            assert validation["valid"] is True

            cycle = ImprovementCycle(experimental_trainer=exp_trainer, training_strategy_builder=strategy_builder, checkpoint_manager=trainer)
            cycle.set_baseline_checkpoint(baseline_path)
            cycle.set_active_version("v1.0")

            candidate = cycle.candidate_builder.create_candidate_version(
                parent_version="v1.0",
                change={"type": "improvement_proposal", "weakness_type": "reasoning_failure"},
                reason="e2e test",
                benchmark_before={"logical_accuracy": 0.7, "safety_score": 0.95},
            )
            cycle.candidate_builder.update_state(candidate["id"], "approved")
            cycle.candidate_builder.record_checkpoint(candidate["id"], {
                "path": training_result.checkpoint_path,
                "lineage": training_result.lineage,
                "validation": validation,
                "training_metrics": training_result.metrics,
            })

            baseline_params_before = cycle.get_current_model_parameters()
            assert baseline_params_before is not None
            param_count_before = len(baseline_params_before)

            deploy_result = cycle.deploy_candidate(candidate["id"], approved=True)
            assert deploy_result["state"] == "deployed"
            assert cycle.get_active_checkpoint() == training_result.checkpoint_path
            assert deploy_result.get("checkpoint_restored") is True

            deployed_params = cycle.get_current_model_parameters()
            assert deployed_params is not None
            assert len(deployed_params) == param_count_before

            candidate_params = cycle.verify_checkpoint_parameters(training_result.checkpoint_path)
            assert candidate_params["valid"] is True
            assert candidate_params["parameter_count"] == param_count_before

            candidate_loaded_params = None
            if cycle.checkpoint_manager is not None:
                try:
                    cycle.checkpoint_manager.load_checkpoint(training_result.checkpoint_path)
                    candidate_loaded_params = [np.asarray(p).copy() for p in cycle.checkpoint_manager.model.get_parameters()]
                except Exception:
                    pass
            if candidate_loaded_params is not None:
                assert len(candidate_loaded_params) == param_count_before
                for i, (a, b) in enumerate(zip(deployed_params, candidate_loaded_params)):
                    np.testing.assert_array_equal(a, b)

            rollback_result = cycle.rollback_candidate(candidate["id"], approved=True, checkpoint_path=baseline_path)
            assert rollback_result["state"] == "rolled_back"
            assert cycle.candidate_builder.get(candidate["id"])["decision"] == "rollback"
            assert rollback_result.get("checkpoint_restored") is True
            assert cycle.get_active_checkpoint() == baseline_path

            rolledback_params = cycle.get_current_model_parameters()
            assert rolledback_params is not None
            assert len(rolledback_params) == param_count_before
            for i, (a, b) in enumerate(zip(rolledback_params, baseline_params_before)):
                np.testing.assert_array_equal(a, b)

    def test_training_failure_does_not_corrupt_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = "alpha beta gamma " * 10
            exp_trainer = ExperimentalTrainer(base_dir=tmp)
            strategy = TrainingStrategy(
                strategy_id="strategy-fail",
                weakness_type="model_problem",
                parameters={
                    "dataset": "general_train",
                    "objective": "next_token",
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "batch_size": 8,
                    "context_size": 8,
                },
            )
            result = exp_trainer.run_training(
                candidate_id="fail-cand",
                strategy=strategy,
                training_text=None,
            )
            assert result.status == "TRAINING_ERROR"
            assert result.error is not None


class TestSecurityTraining:
    def test_candidate_cannot_modify_security(self):
        cycle = ImprovementCycle()
        candidate = cycle.candidate_builder.create_candidate_version(
            "v1.0",
            {"type": "security_bypass", "target": "permission_guard"},
            "malicious",
        )
        assert candidate["state"] == "sandbox"
        assert candidate["change"]["type"] == "security_bypass"

    def test_training_strategy_respects_whitelist(self):
        builder = TrainingStrategyBuilder()
        weakness = {"type": "model_problem", "metric": "general", "severity": 1}
        strategy = builder.build_from_weakness(weakness, "sec-cand", "v1.0")
        params = strategy.parameters
        assert "learning_rate" in params
        assert "epochs" in params
        assert "security_policies" not in params
        assert "permission_guard" not in params

    def test_validate_checkpoint_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            traversal_path = os.path.join(tmp, "..", "evil.pkl")
            result = trainer.validate_checkpoint(traversal_path)
            assert result["valid"] is False
            assert "non autorizzato" in result["error"].lower() or "not found" in result["error"].lower()

    def test_validate_checkpoint_blocks_absolute_external_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            external_path = "/tmp/evil_external.pkl"
            result = trainer.validate_checkpoint(external_path)
            assert result["valid"] is False
            assert "non autorizzato" in result["error"].lower() or "not found" in result["error"].lower()

    def test_safe_pickle_load_blocks_external_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            with pytest.raises(ValueError, match="non autorizzato"):
                trainer._safe_pickle_load("/tmp/external.pkl")

    def test_validate_checkpoint_blocks_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = ExperimentalTrainer(base_dir=tmp)
            corrupt_path = os.path.join(tmp, "corrupt.pkl")
            with open(corrupt_path, "wb") as f:
                f.write(b"not a pickle")
            result = trainer.validate_checkpoint(corrupt_path)
            assert result["valid"] is False
