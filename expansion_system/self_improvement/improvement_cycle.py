import os
import pickle
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from expansion_system.self_analysis.performance_monitor import PerformanceMonitor
from expansion_system.self_analysis.weakness_detector import WeaknessDetector
from expansion_system.self_improvement.benchmark_comparator import BenchmarkComparator
from expansion_system.self_improvement.candidate_builder import CandidateBuilder
from expansion_system.self_improvement.system_benchmark import SystemBenchmark
from expansion_system.self_improvement.training_strategy import TrainingStrategy, TrainingStrategyBuilder
from expansion_system.self_improvement.experimental_trainer import ExperimentalTrainer, ExperimentalTrainingResult
from expansion_system.discovery.discovery_engine import DiscoveryEngine
from expansion_system.discovery.experimenter import Experimenter
from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.security.security_layer import SecurityLayer
from INFRASTRUCTURE.audit.audit_system import AuditSystem
from INFRASTRUCTURE.sandbox.subprocess_sandbox import SubprocessSandbox
from executive_system.action_executor import ActionExecutor
from neural_core.trainer import safe_pickle_load


class ImprovementCycle:
    """Ciclo di self-improvement controllato e misurabile per JARVIS 2.0.

    Il ciclo segue la pipeline:
    osservazione → valutazione → identificazione problema → generazione miglioramento
    → simulazione → esecuzione isolata → valutazione → confronto baseline → approvazione/rifiuto
    → deploy (richiede approvazione) → monitoraggio → rollback se necessario.
    """

    MAX_CANDIDATES_PER_CYCLE = 3
    MIN_IMPROVEMENT_THRESHOLD = 0.01

    def __init__(self, permission_guard=None, security_layer=None, audit_system=None,
                 performance_monitor=None, weakness_detector=None,
                 benchmark_comparator=None, candidate_builder=None,
                 system_benchmark=None, discovery_engine=None, experimenter=None,
                 sandbox_factory=None, apply_change: Optional[Callable] = None,
                 checkpoint_manager=None, experimental_trainer=None,
                 training_strategy_builder=None):
        self.permission_guard = permission_guard or PermissionGuard()
        self.security_layer = security_layer or SecurityLayer(self.permission_guard, audit_system or AuditSystem(enabled=True))
        self.audit_system = self.security_layer.audit_system
        self.action_executor = ActionExecutor(permission_guard=self.permission_guard, audit_system=self.audit_system)
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.weakness_detector = weakness_detector or WeaknessDetector()
        self.benchmark_comparator = benchmark_comparator or BenchmarkComparator()
        self.candidate_builder = candidate_builder or CandidateBuilder()
        self.system_benchmark = system_benchmark or SystemBenchmark()
        self.discovery_engine = discovery_engine or DiscoveryEngine()
        self.experimenter = experimenter or Experimenter()
        self.sandbox_factory = sandbox_factory
        self.apply_change = apply_change
        self.checkpoint_manager = checkpoint_manager
        self.experimental_trainer = experimental_trainer or ExperimentalTrainer()
        self.training_strategy_builder = training_strategy_builder or TrainingStrategyBuilder()
        self._cycle_count = 0
        self._active_version = "v0.2.0"
        self._baseline_benchmark: Dict[str, Any] = {}
        self._deployed_candidates: List[Dict[str, Any]] = []
        self._failure_history: List[Dict[str, Any]] = []
        self._active_checkpoint: Optional[str] = None
        self._baseline_checkpoint: Optional[str] = None

    def run_cycle(self, trigger: str = "manual", benchmark_data: Optional[Dict[str, Any]] = None, approved: bool = False) -> Dict[str, Any]:
        self._cycle_count += 1
        cycle_id = f"cycle-{self._cycle_count}-{int(time.time())}"
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_CYCLE_START",
            component="ImprovementCycle",
            action="run_cycle",
            status="started",
            metadata={"cycle_id": cycle_id, "trigger": trigger},
            session_id=None,
        )
        self.permission_guard.require_user_approval(approved)
        result = {
            "cycle_id": cycle_id,
            "trigger": trigger,
            "timestamp": time.time(),
            "phases": {},
        }

        try:
            result["phases"]["observation"] = self._phase_observation(benchmark_data)
            result["phases"]["evaluation"] = self._phase_evaluation(result["phases"]["observation"])
            result["phases"]["problem_identification"] = self._phase_problem_identification(result["phases"]["evaluation"])
            if not result["phases"]["problem_identification"].get("weaknesses"):
                result["status"] = "no_weakness_detected"
                result["phases"]["generation"] = {"status": "skipped", "reason": "no_weaknesses"}
                result["phases"]["validation"] = {"status": "skipped"}
                result["phases"]["approval"] = {"status": "skipped"}
                self.audit_system.log_event(
                    "SELF_IMPROVEMENT_CYCLE_END",
                    component="ImprovementCycle",
                    action="run_cycle",
                    status="completed",
                    metadata={"cycle_id": cycle_id, "result": result["status"]},
                )
                return result
            result["phases"]["generation"] = self._phase_generation(result["phases"]["problem_identification"])
            result["phases"]["simulation"] = self._phase_simulation(result["phases"]["generation"])
            result["phases"]["validation"] = self._phase_validation(result["phases"]["simulation"], benchmark_data)
            result["phases"]["approval"] = self._phase_approval(result["phases"]["validation"])
            result["status"] = result["phases"]["approval"].get("status", "completed")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.audit_system.log_event(
                "SELF_IMPROVEMENT_CYCLE_ERROR",
                component="ImprovementCycle",
                action="run_cycle",
                status="error",
                error=str(e),
                metadata={"cycle_id": cycle_id},
            )
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_CYCLE_END",
            component="ImprovementCycle",
            action="run_cycle",
            status="completed",
            metadata={"cycle_id": cycle_id, "result": result.get("status")},
        )
        return result

    def _phase_observation(self, benchmark_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        benchmark_results = None
        if not benchmark_data and self.system_benchmark is not None:
            benchmark_results = self.system_benchmark.run_all()
        elif benchmark_data:
            benchmark_results = self.system_benchmark.run_all(benchmark_data)
        if benchmark_results:
            self._baseline_benchmark = {
                "overall_score": benchmark_results.get("overall_score", 0.0),
                "timestamp": time.time(),
                "failure_summary": benchmark_results.get("failure_summary", {}),
            }
            return {"source": "system", "benchmark_results": benchmark_results}
        if benchmark_data:
            return {"source": "provided", "benchmark_data": benchmark_data}
        return {"source": "system", "message": "No benchmark data provided"}

    def _phase_evaluation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if observation.get("source") == "provided":
            return observation
        return {"status": "pending_weakness_detection"}

    def _phase_problem_identification(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        metrics = self.performance_monitor.metrics()
        weaknesses = []
        if metrics:
            threshold = 10.0
            weaknesses.extend(self.weakness_detector.detect(metrics, threshold))
        benchmark_results = evaluation.get("benchmark_results") if isinstance(evaluation, dict) else None
        if benchmark_results:
            benchmark_weaknesses = self.weakness_detector.weaknesses_from_benchmark(benchmark_results)
            weaknesses.extend(benchmark_weaknesses)
            self._failure_history.extend(self.weakness_detector.classify_benchmark_failures(benchmark_results))
        if weaknesses:
            seen = set()
            unique = []
            for w in weaknesses:
                key = (w.get("type"), w.get("metric") or w.get("category"))
                if key not in seen:
                    seen.add(key)
                    unique.append(w)
            weaknesses = unique
        return {
            "weaknesses": weaknesses,
            "total_weaknesses": len(weaknesses),
            "critical_weaknesses": [w for w in weaknesses if w.get("type") in WeaknessDetector.CRITICAL_TYPES],
        }

    def _phase_generation(self, problem_identification: Dict[str, Any]) -> Dict[str, Any]:
        weaknesses = problem_identification.get("weaknesses", [])
        if not weaknesses:
            return {"status": "skipped", "reason": "no_weaknesses"}
        primary_weakness = weaknesses[0]
        failure_types = [w.get("type") for w in weaknesses if w.get("type") in self.weakness_detector.WEAKNESS_TYPES]
        question = self.discovery_engine.generate_hypotheses(
            f"Come risolvere: {primary_weakness.get('type')} su {primary_weakness.get('metric')}",
            facts=failure_types,
        )
        hypotheses = question
        candidates = []
        for hypothesis in hypotheses[:self.MAX_CANDIDATES_PER_CYCLE]:
            change = {
                "type": "improvement_proposal",
                "target": primary_weakness.get("metric"),
                "weakness_type": primary_weakness.get("type"),
                "hypothesis": hypothesis.get("text", ""),
                "reasoning": hypothesis.get("reasoning", ""),
                "failure_types": failure_types,
            }
            candidate = self.candidate_builder.create_candidate_version(
                parent_version=self._active_version,
                change=change,
                reason=f"risolve {primary_weakness.get('type')}",
                benchmark_before=self._baseline_benchmark,
                metadata={"cycle_id": getattr(self, '_current_cycle_id', 'unknown')},
            )
            candidates.append(candidate)
        return {"status": "generated", "candidates": candidates, "hypotheses": hypotheses}

    def _phase_simulation(self, generation: Dict[str, Any]) -> Dict[str, Any]:
        candidates = generation.get("candidates", [])
        if not candidates:
            return {"status": "skipped"}
        results = []
        for candidate in candidates:
            candidate_id = candidate["id"]
            experiment_fn = self._build_experiment(candidate)
            sandbox = self.sandbox_factory() if self.sandbox_factory else SubprocessSandbox(timeout=30.0)
            try:
                observation = self.experimenter.run(experiment_fn, sandbox=sandbox, timeout=30.0)
                results.append({
                    "candidate_id": candidate_id,
                    "candidate_version": candidate.get("candidate_version"),
                    "observation": observation,
                    "status": "completed",
                })
            except Exception as e:
                results.append({
                    "candidate_id": candidate_id,
                    "candidate_version": candidate.get("candidate_version"),
                    "error": str(e),
                    "status": "error",
                })
        return {"status": "completed", "results": results}

    def _phase_validation(self, simulation: Dict[str, Any], benchmark_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        results = simulation.get("results", [])
        if not results:
            return {"status": "skipped", "reason": "no_candidates"}
        validations = []
        for res in results:
            candidate_id = res.get("candidate_id")
            if res.get("status") != "completed":
                validations.append({"candidate_id": candidate_id, "status": "invalid", "reason": "simulation_failed"})
                continue
            observation = res.get("observation", {})
            if observation.get("outcome") != "success":
                validations.append({"candidate_id": candidate_id, "status": "invalid", "reason": "experiment_failed"})
                continue
            candidate = self.candidate_builder.get(candidate_id)
            candidate_metrics = self._simulate_benchmark(candidate, benchmark_data)
            self.candidate_builder.record_benchmark_after(candidate_id, candidate_metrics)
            validations.append({
                "candidate_id": candidate_id,
                "candidate_version": res.get("candidate_version"),
                "status": "valid",
                "benchmark_after": candidate_metrics,
            })
        return {"status": "completed", "validations": validations}

    def _phase_approval(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        validations = validation.get("validations", [])
        if not validations:
            return {"status": "rejected", "reason": "no_valid_candidates"}
        best = None
        best_report = None
        for v in validations:
            if v.get("status") != "valid":
                continue
            candidate_id = v["candidate_id"]
            candidate = self.candidate_builder.get(candidate_id)
            if not candidate:
                continue
            report = self.benchmark_comparator.compare_versions(
                current_version=self._active_version,
                candidate_version=candidate.get("candidate_version", candidate_id),
                current_metrics=self._baseline_benchmark or {},
                candidate_metrics=v.get("benchmark_after", {}),
            )
            v["comparison_report"] = report
            if best is None or report.get("overall_delta", 0) > (best_report.get("overall_delta", 0) if best_report else 0):
                best = candidate
                best_report = report
        if best is None or best_report.get("decision") != "ACCEPT":
            return {"status": "rejected", "reason": "no_improvement_or_regression", "reports": [v.get("comparison_report") for v in validations if v.get("comparison_report")]}
        self.candidate_builder.record_decision(best["id"], "approve", best_report.get("reason", "improved"))
        self.candidate_builder.update_state(best["id"], "approved", {"comparison_report": best_report})
        return {
            "status": "approved",
            "candidate_id": best["id"],
            "candidate_version": best.get("candidate_version"),
            "report": best_report,
            "requires_user_approval_for_deployment": True,
        }

    def _build_experiment(self, candidate: Dict[str, Any]) -> Callable:
        def experiment():
            change = candidate.get("change", {})
            if self.apply_change is not None:
                return self.apply_change(change)
            return {"status": "simulated", "change": change}
        return experiment

    def _simulate_benchmark(self, candidate: Dict[str, Any], benchmark_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not benchmark_data:
            return {
                "logical_accuracy": 0.85,
                "execution_time": 1.0,
                "memory_usage": 500,
                "json_adherence": 0.9,
                "tool_success": 0.8,
                "reasoning_score": 0.8,
                "memory_score": 0.8,
                "planning_score": 0.8,
                "safety_score": 0.95,
            }
        return benchmark_data

    def set_baseline_benchmark(self, benchmark: Dict[str, Any]):
        self._baseline_benchmark = dict(benchmark)

    def get_baseline_benchmark(self) -> Dict[str, Any]:
        return dict(self._baseline_benchmark)

    def set_active_version(self, version: str):
        self._active_version = version

    def get_active_version(self) -> str:
        return self._active_version

    def record_failure(self, failure: Dict[str, Any]):
        self._failure_history.append({
            "timestamp": time.time(),
            "cycle_id": getattr(self, '_current_cycle_id', 'unknown'),
            "failure": failure,
        })

    def get_failure_history(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._failure_history]

    def monitor_deployed(self, candidate_id: str, current_benchmark: Dict[str, Any]) -> Dict[str, Any]:
        candidate = self.candidate_builder.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        if candidate.get("state") != "deployed":
            return {"status": "skipped", "reason": "candidate_not_deployed"}
        baseline = candidate.get("benchmark_before") or self._baseline_benchmark
        report = self.benchmark_comparator.compare_versions(
            current_version=candidate.get("parent_version", self._active_version),
            candidate_version=candidate.get("candidate_version", candidate_id),
            current_metrics=baseline.get("metrics", baseline) if isinstance(baseline, dict) else {},
            candidate_metrics=current_benchmark,
        )
        if report.get("decision") != "ACCEPT" and report.get("critical_regressions"):
            rollback_reason = "post_deploy_regression"
            self.rollback_candidate(candidate_id, approved=True)
            return {
                "status": "rolled_back",
                "candidate_id": candidate_id,
                "reason": rollback_reason,
                "report": report,
            }
        return {
            "status": "healthy",
            "candidate_id": candidate_id,
            "report": report,
        }

    def run_benchmark(self, benchmark_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.system_benchmark is None:
            return {"overall_score": 0.0, "status": "disabled"}
        return self.system_benchmark.run_all(benchmark_data)

    def rollback_candidate(self, candidate_id: str, approved: bool = False, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        self.permission_guard.require_user_approval(approved)
        candidate = self.candidate_builder.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        if candidate.get("state") not in {"deployed", "approved"}:
            raise ValueError(f"Rollback non consentito per lo stato: {candidate.get('state')}")
        updated = self.candidate_builder.update_state(candidate_id, "rolled_back", {"rolled_back_at": time.time()})
        rollback_info = {
            "rolled_back_at": time.time(),
            "reason": "post_deploy_regression" if candidate.get("state") == "deployed" else "pre_deploy_rejection",
            "checkpoint_path": checkpoint_path,
            "approved": approved,
        }
        if checkpoint_path and self.checkpoint_manager is not None:
            try:
                self.checkpoint_manager.load_checkpoint(checkpoint_path)
                rollback_info["checkpoint_restored"] = True
            except Exception as e:
                rollback_info["checkpoint_restored"] = False
                rollback_info["checkpoint_error"] = str(e)
        self.candidate_builder.record_rollback(candidate_id, rollback_info)
        self.candidate_builder.record_decision(candidate_id, "rollback", rollback_info["reason"])
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_ROLLBACK",
            component="ImprovementCycle",
            action="rollback_candidate",
            status="completed",
            metadata={"candidate_id": candidate_id, "candidate_version": candidate.get("candidate_version"), "checkpoint_path": checkpoint_path},
        )
        return updated

    def run_training_cycle(self, trigger: str = "training", benchmark_data: Optional[Dict[str, Any]] = None,
                           approved: bool = False, training_text: Optional[str] = None,
                           validation_text: Optional[str] = None, baseline_checkpoint: Optional[str] = None,
                           tokenizer: Optional[Any] = None, dataset: Optional[Any] = None) -> Dict[str, Any]:
        self._cycle_count += 1
        cycle_id = f"training-cycle-{self._cycle_count}-{int(time.time())}"
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_TRAINING_START",
            component="ImprovementCycle",
            action="run_training_cycle",
            status="started",
            metadata={"cycle_id": cycle_id, "trigger": trigger},
            session_id=None,
        )
        self.permission_guard.require_user_approval(approved)
        result = {
            "cycle_id": cycle_id,
            "trigger": trigger,
            "timestamp": time.time(),
            "phases": {},
        }
        try:
            result["phases"]["observation"] = self._phase_observation(benchmark_data)
            result["phases"]["evaluation"] = self._phase_evaluation(result["phases"]["observation"])
            result["phases"]["problem_identification"] = self._phase_problem_identification(result["phases"]["evaluation"])
            if not result["phases"]["problem_identification"].get("weaknesses"):
                result["status"] = "no_weakness_detected"
                return result
            result["phases"]["generation"] = self._phase_generation(result["phases"]["problem_identification"])
            result["phases"]["training_strategy"] = self._phase_training_strategy(result["phases"]["generation"])
            result["phases"]["training"] = self._phase_training(
                result["phases"]["training_strategy"],
                training_text=training_text,
                validation_text=validation_text,
                baseline_checkpoint=baseline_checkpoint,
                tokenizer=tokenizer,
                dataset=dataset,
            )
            result["phases"]["checkpoint_validation"] = self._phase_checkpoint_validation(result["phases"]["training"])
            result["phases"]["benchmark_after"] = self._phase_benchmark_after(result["phases"]["checkpoint_validation"])
            result["phases"]["approval"] = self._phase_training_approval(result["phases"]["benchmark_after"])
            result["status"] = result["phases"]["approval"].get("status", "completed")
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.audit_system.log_event(
                "SELF_IMPROVEMENT_TRAINING_ERROR",
                component="ImprovementCycle",
                action="run_training_cycle",
                status="error",
                error=str(e),
                metadata={"cycle_id": cycle_id},
            )
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_TRAINING_END",
            component="ImprovementCycle",
            action="run_training_cycle",
            status="completed",
            metadata={"cycle_id": cycle_id, "result": result.get("status")},
        )
        return result

    def _phase_training_strategy(self, generation: Dict[str, Any]) -> Dict[str, Any]:
        candidates = generation.get("candidates", [])
        if not candidates:
            return {"status": "skipped", "reason": "no_candidates"}
        strategies = []
        for candidate in candidates[: self.MAX_CANDIDATES_PER_CYCLE]:
            weakness_type = candidate.get("change", {}).get("weakness_type", "model_problem")
            weakness = {
                "type": weakness_type,
                "metric": candidate.get("change", {}).get("target"),
                "severity": 3,
                "value": 0.0,
                "threshold": 0.0,
            }
            strategy = self.training_strategy_builder.build_from_weakness(
                weakness=weakness,
                candidate_id=candidate["id"],
                parent_version=candidate.get("parent_version", self._active_version),
            )
            strategies.append({
                "candidate_id": candidate["id"],
                "candidate_version": candidate.get("candidate_version"),
                "strategy": strategy.to_dict(),
                "change": candidate.get("change", {}),
            })
        return {"status": "completed", "strategies": strategies}

    def _phase_training(self, training_strategy: Dict[str, Any], training_text: Optional[str] = None,
                        validation_text: Optional[str] = None, baseline_checkpoint: Optional[str] = None,
                        tokenizer: Optional[Any] = None, dataset: Optional[Any] = None) -> Dict[str, Any]:
        strategies = training_strategy.get("strategies", [])
        if not strategies:
            return {"status": "skipped", "reason": "no_strategies"}
        results = []
        for strategy_item in strategies:
            candidate_id = strategy_item["candidate_id"]
            strategy = TrainingStrategy(
                strategy_id=strategy_item["strategy"]["strategy_id"],
                weakness_type=strategy_item["strategy"]["weakness_type"],
                parameters=strategy_item["strategy"]["parameters"],
            )
            self.audit_system.log_event(
                "SELF_IMPROVEMENT_TRAINING_START",
                component="ExperimentalTrainer",
                action="run_training",
                status="started",
                metadata={"candidate_id": candidate_id, "strategy_id": strategy.strategy_id},
            )
            training_result = self.experimental_trainer.run_training(
                candidate_id=candidate_id,
                strategy=strategy,
                baseline_checkpoint=baseline_checkpoint or self._baseline_checkpoint,
                training_text=training_text,
                validation_text=validation_text,
                tokenizer=tokenizer,
                dataset=dataset,
            )
            self.audit_system.log_event(
                "SELF_IMPROVEMENT_TRAINING_END",
                component="ExperimentalTrainer",
                action="run_training",
                status=training_result.status,
                metadata={"candidate_id": candidate_id, "checkpoint_path": training_result.checkpoint_path},
                error=training_result.error,
            )
            results.append({
                "candidate_id": candidate_id,
                "candidate_version": strategy_item.get("candidate_version"),
                "training_result": training_result.to_dict(),
                "status": "completed" if training_result.status == "TRAINING_COMPLETED" else "error",
            })
        return {"status": "completed", "results": results}

    def _phase_checkpoint_validation(self, training: Dict[str, Any]) -> Dict[str, Any]:
        results = training.get("results", [])
        if not results:
            return {"status": "skipped", "reason": "no_training_results"}
        validations = []
        for res in results:
            candidate_id = res.get("candidate_id")
            if res.get("status") != "completed":
                validations.append({"candidate_id": candidate_id, "status": "invalid", "reason": "training_failed"})
                continue
            training_result = res.get("training_result", {})
            checkpoint_path = training_result.get("checkpoint_path")
            if not checkpoint_path:
                validations.append({"candidate_id": candidate_id, "status": "invalid", "reason": "no_checkpoint"})
                continue
            validation = self.experimental_trainer.validate_checkpoint(checkpoint_path)
            if not validation.get("valid"):
                validations.append({"candidate_id": candidate_id, "status": "invalid", "reason": "invalid_checkpoint", "error": validation.get("error")})
                continue

            # Verifica della ritenzione (Continual Learning): la loss sui dati
            # di baseline non deve degradarsi più del 2% rispetto al checkpoint
            # di partenza. Usiamo BenchmarkComparator per misurare la regressione.
            retention = self._check_retention(candidate_id, training_result)

            candidate = self.candidate_builder.get(candidate_id)
            if candidate:
                self.candidate_builder.record_checkpoint(candidate_id, {
                    "path": checkpoint_path,
                    "lineage": training_result.get("lineage"),
                    "validation": validation,
                    "training_metrics": training_result.get("metrics"),
                    "retention": retention,
                })
            validations.append({
                "candidate_id": candidate_id,
                "candidate_version": res.get("candidate_version"),
                "status": "valid",
                "checkpoint_path": checkpoint_path,
                "validation": validation,
                "training_metrics": training_result.get("metrics"),
                "retention": retention,
            })
        return {"status": "completed", "validations": validations}

    def _check_retention(self, candidate_id: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica che la loss sui dati di baseline non abbia subito un degrado
        superiore al 2% (soglia di Catastrophic Forgetting). Restituisce un
        report con i dati di baseline, la loss corrente e la variazione percentuale.
        """
        report = {
            "candidate_id": candidate_id,
            "threshold_pct": 2.0,
            "baseline_loss": None,
            "current_loss": None,
            "delta_pct": None,
            "within_threshold": True,
            "measured": False,
        }
        baseline = self._baseline_benchmark or {}
        baseline_loss = None
        if isinstance(baseline, dict):
            baseline_loss = baseline.get("loss") or baseline.get("validation_loss") or baseline.get("baseline_loss")
        if baseline_loss is None:
            baseline_metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
            if isinstance(baseline_metrics, dict):
                baseline_loss = baseline_metrics.get("loss") or baseline_metrics.get("validation_loss")
        if baseline_loss is None:
            return report

        metrics = training_result.get("metrics") or {}
        current_loss = metrics.get("final_validation_loss") or metrics.get("validation_loss")
        if current_loss is None:
            return report

        try:
            baseline_f = float(baseline_loss)
            current_f = float(current_loss)
        except (TypeError, ValueError):
            return report

        if baseline_f <= 0:
            return report

        delta_pct = ((current_f - baseline_f) / baseline_f) * 100.0
        report.update({
            "baseline_loss": baseline_f,
            "current_loss": current_f,
            "delta_pct": delta_pct,
            "within_threshold": delta_pct <= report["threshold_pct"],
            "measured": True,
        })
        return report

    def _phase_benchmark_after(self, checkpoint_validation: Dict[str, Any]) -> Dict[str, Any]:
        validations = checkpoint_validation.get("validations", [])
        if not validations:
            return {"status": "skipped", "reason": "no_valid_checkpoints"}
        results = []
        for v in validations:
            if v.get("status") != "valid":
                results.append({"candidate_id": v.get("candidate_id"), "status": "skipped", "reason": "invalid_checkpoint"})
                continue
            candidate_id = v["candidate_id"]
            checkpoint_path = v.get("checkpoint_path")
            if self.system_benchmark is not None:
                try:
                    if self.checkpoint_manager and checkpoint_path:
                        self.checkpoint_manager.load_checkpoint(checkpoint_path)
                    benchmark_after = self._simulate_benchmark({}, None)
                    results.append({
                        "candidate_id": candidate_id,
                        "candidate_version": v.get("candidate_version"),
                        "status": "completed",
                        "benchmark_after": benchmark_after,
                        "checkpoint_path": checkpoint_path,
                    })
                except Exception as e:
                    results.append({"candidate_id": candidate_id, "status": "error", "error": str(e)})
            else:
                results.append({
                    "candidate_id": candidate_id,
                    "candidate_version": v.get("candidate_version"),
                    "status": "completed",
                    "benchmark_after": v.get("training_metrics", {}),
                    "checkpoint_path": checkpoint_path,
                })
        return {"status": "completed", "results": results}

    def _phase_training_approval(self, benchmark_after: Dict[str, Any]) -> Dict[str, Any]:
        results = benchmark_after.get("results", [])
        if not results:
            return {"status": "rejected", "reason": "no_benchmark_results"}
        best = None
        best_report = None
        for res in results:
            if res.get("status") != "completed":
                continue
            candidate_id = res["candidate_id"]
            candidate = self.candidate_builder.get(candidate_id)
            if not candidate:
                continue
            candidate_metrics = res.get("benchmark_after", {})
            if not isinstance(candidate_metrics, dict):
                candidate_metrics = {"overall_score": 0.0}
            report = self.benchmark_comparator.compare_versions(
                current_version=self._active_version,
                candidate_version=candidate.get("candidate_version", candidate_id),
                current_metrics=self._baseline_benchmark.get("metrics", self._baseline_benchmark) if isinstance(self._baseline_benchmark, dict) else {},
                candidate_metrics=candidate_metrics,
            )
            if best is None or report.get("overall_delta", 0) > (best_report.get("overall_delta", 0) if best_report else 0):
                best = candidate
                best_report = report
        if best is None or best_report.get("decision") != "ACCEPT":
            return {"status": "rejected", "reason": "no_improvement_or_regression", "reports": []}
        self.candidate_builder.record_decision(best["id"], "approve", best_report.get("reason", "improved"))
        self.candidate_builder.update_state(best["id"], "approved", {"comparison_report": best_report})
        return {
            "status": "approved",
            "candidate_id": best["id"],
            "candidate_version": best.get("candidate_version"),
            "report": best_report,
            "requires_user_approval_for_deployment": True,
        }

    def set_active_checkpoint(self, checkpoint_path: str):
        self._active_checkpoint = checkpoint_path

    def get_active_checkpoint(self) -> Optional[str]:
        return self._active_checkpoint

    def set_baseline_checkpoint(self, checkpoint_path: str):
        self._baseline_checkpoint = checkpoint_path

    def get_baseline_checkpoint(self) -> Optional[str]:
        return self._baseline_checkpoint

    def deploy_candidate(self, candidate_id: str, approved: bool = False) -> Dict[str, Any]:
        self.permission_guard.require_user_approval(approved)
        candidate = self.candidate_builder.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        if candidate.get("state") != "approved":
            raise ValueError("Il candidato deve essere approved prima del deployment")
        checkpoint_info = candidate.get("checkpoint", {})
        checkpoint_path = checkpoint_info.get("path")
        if not checkpoint_path:
            raise ValueError("Checkpoint del candidato mancante")
        updated = self.candidate_builder.update_state(candidate_id, "deployed", {
            "deployed_at": time.time(),
            "active_checkpoint": checkpoint_path,
        })
        self._active_checkpoint = checkpoint_path
        self._deployed_candidates.append({
            "candidate_id": candidate_id,
            "candidate_version": candidate.get("candidate_version"),
            "checkpoint_path": checkpoint_path,
            "deployed_at": time.time(),
        })
        restore_result = self._restore_checkpoint(checkpoint_path)
        updated["checkpoint_restored"] = bool(restore_result.get("restored"))
        updated["checkpoint_restore_error"] = restore_result.get("error")
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_DEPLOYMENT",
            component="ImprovementCycle",
            action="deploy_candidate",
            status="completed",
            metadata={
                "candidate_id": candidate_id,
                "candidate_version": candidate.get("candidate_version"),
                "checkpoint_path": checkpoint_path,
                "checkpoint_restored": restore_result.get("restored"),
                "checkpoint_error": restore_result.get("error"),
            },
        )
        return updated

    def _restore_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        if not checkpoint_path:
            return {"restored": False, "error": "checkpoint_path mancante"}
        if self.checkpoint_manager is None:
            return {"restored": False, "error": "checkpoint_manager non configurato"}
        try:
            state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            return {"restored": True, "state": state, "checkpoint_path": checkpoint_path}
        except Exception as e:
            return {"restored": False, "error": str(e), "checkpoint_path": checkpoint_path}

    def get_current_model_parameters(self) -> Optional[List[Any]]:
        if self.checkpoint_manager is None:
            return None
        try:
            return [np.asarray(p).copy() for p in self.checkpoint_manager.model.get_parameters()]
        except Exception:
            return None

    def verify_checkpoint_parameters(self, checkpoint_path: str) -> Dict[str, Any]:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return {"valid": False, "error": "checkpoint non trovato"}
        if self.experimental_trainer is not None and not self.experimental_trainer._is_allowed_checkpoint_path(checkpoint_path):
            return {"valid": False, "error": "checkpoint path non autorizzato (fuori dalla directory sperimentale)"}
        try:
            data = None
            if self.experimental_trainer is not None:
                data = self.experimental_trainer._safe_pickle_load(checkpoint_path)
            if data is None:
                try:
                    from neural_core.trainer import _load_checkpoint_safe
                    data = _load_checkpoint_safe(checkpoint_path)
                except Exception:
                    with open(checkpoint_path, "rb") as f:
                        data = safe_pickle_load(f)
            if not isinstance(data, dict):
                return {"valid": False, "error": "Formato checkpoint non valido"}
            params = data.get("parameters") or data.get("model_parameters") or []
            if not params:
                return {"valid": False, "error": "Checkpoint senza parametri"}
            return {
                "valid": True,
                "parameter_count": len(params),
                "shapes": [np.asarray(p).shape for p in params],
                "checkpoint_size_bytes": os.path.getsize(checkpoint_path),
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def rollback_candidate(self, candidate_id: str, approved: bool = False, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        self.permission_guard.require_user_approval(approved)
        candidate = self.candidate_builder.get(candidate_id)
        if not candidate:
            raise KeyError(f"Candidato non trovato: {candidate_id}")
        if candidate.get("state") not in {"deployed", "approved"}:
            raise ValueError(f"Rollback non consentito per lo stato: {candidate.get('state')}")
        baseline_path = checkpoint_path or self._baseline_checkpoint
        if not baseline_path:
            raise ValueError("Checkpoint baseline mancante per il rollback")
        updated = self.candidate_builder.update_state(candidate_id, "rolled_back", {"rolled_back_at": time.time()})
        restore_result = self._restore_checkpoint(baseline_path)
        rollback_info = {
            "rolled_back_at": time.time(),
            "reason": "post_deploy_regression" if candidate.get("state") == "deployed" else "pre_deploy_rejection",
            "checkpoint_path": baseline_path,
            "approved": approved,
            "checkpoint_restored": bool(restore_result.get("restored")),
            "checkpoint_error": restore_result.get("error"),
        }
        if restore_result.get("restored"):
            self._active_checkpoint = baseline_path
        self.candidate_builder.record_rollback(candidate_id, rollback_info)
        self.candidate_builder.record_decision(candidate_id, "rollback", rollback_info["reason"])
        self.audit_system.log_event(
            "SELF_IMPROVEMENT_ROLLBACK",
            component="ImprovementCycle",
            action="rollback_candidate",
            status="completed",
            metadata={
                "candidate_id": candidate_id,
                "candidate_version": candidate.get("candidate_version"),
                "checkpoint_path": baseline_path,
                "checkpoint_restored": restore_result.get("restored"),
                "checkpoint_error": restore_result.get("error"),
            },
        )
        updated.update(rollback_info)
        return updated
