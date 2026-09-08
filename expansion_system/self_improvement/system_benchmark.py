"""
System behavioral benchmark for JARVIS 2.0.

Provides a structured, reproducible benchmark dataset and scoring engine
for the self-improvement cycle. The dataset is separate from training data
and supports multiple categories with explicit failure modes.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class BenchmarkDataset:
    """In-memory structured benchmark dataset.

    Each case has:
      - id
      - category
      - input
      - expected_behavior
      - expected_output or evaluation criteria
      - difficulty
      - criticality
      - timeout
      - metadata
    """

    def __init__(self, cases: Optional[List[Dict[str, Any]]] = None):
        self._cases: List[Dict[str, Any]] = []
        self._index: Dict[str, Dict[str, Any]] = {}
        if cases:
            for case in cases:
                self.add(case)

    def add(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if not case.get("id"):
            case["id"] = hashlib.sha256(str(case).encode()).hexdigest()[:16]
        case.setdefault("category", "general")
        case.setdefault("difficulty", "medium")
        case.setdefault("criticality", "medium")
        case.setdefault("timeout", 5.0)
        case.setdefault("metadata", {})
        self._cases.append(case)
        self._index[case["id"]] = case
        return case

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        return dict(self._index.get(case_id, {}))

    def all(self) -> List[Dict[str, Any]]:
        return [dict(case) for case in self._cases]

    def by_category(self, category: str) -> List[Dict[str, Any]]:
        return [dict(case) for case in self._cases if case.get("category") == category]

    def categories(self) -> List[str]:
        return sorted({case.get("category") for case in self._cases})

    def __len__(self) -> int:
        return len(self._cases)


class FailureClassifier:
    """Classifica i fallimenti del benchmark in tipi strutturati."""

    TYPES = [
        "instruction_following_failure",
        "reasoning_failure",
        "memory_failure",
        "planning_failure",
        "tool_selection_failure",
        "tool_execution_failure",
        "error_recovery_failure",
        "safety_failure",
        "performance_regression",
        "timeout_failure",
        "unexpected_error",
    ]

    def classify(self, case: Dict[str, Any], error: Optional[str] = None, result: Any = None) -> str:
        category = str(case.get("category", "")).lower()
        if error:
            text = str(error).lower()
            if "timeout" in text or "timed out" in text:
                return "timeout_failure"
            if "permission" in text or "security" in text or "bloccato" in text:
                return "safety_failure"
            if "memory" in text or "recall" in text:
                return "memory_failure"
        if category in {"instruction_following", "language"}:
            return "instruction_following_failure"
        if category == "reasoning":
            return "reasoning_failure"
        if category == "memory":
            return "memory_failure"
        if category == "planning":
            return "planning_failure"
        if category == "tool_selection":
            return "tool_selection_failure"
        if category == "tool_execution":
            return "tool_execution_failure"
        if category == "error_recovery":
            return "error_recovery_failure"
        if category == "safety":
            return "safety_failure"
        if category == "performance":
            return "performance_regression"
        if error:
            return "unexpected_error"
        return "unexpected_error"

    def to_weakness_type(self, failure_type: str) -> str:
        mapping = {
            "instruction_following_failure": "prompt_problem",
            "reasoning_failure": "model_problem",
            "memory_failure": "memory_problem",
            "planning_failure": "planner_problem",
            "tool_selection_failure": "tool_use_problem",
            "tool_execution_failure": "tool_use_problem",
            "error_recovery_failure": "infrastructure_problem",
            "safety_failure": "infrastructure_problem",
            "performance_regression": "infrastructure_problem",
            "timeout_failure": "infrastructure_problem",
            "unexpected_error": "model_problem",
        }
        return mapping.get(failure_type, "occasional_error")


class SystemBenchmark:
    """Benchmark di sistema per JARVIS 2.0.

    Misura capacità funzionali e non funzionali senza modificare lo stato attivo.
    Tutti i benchmark sono riproducibili e restituiscono metriche strutturate.
    """

    def __init__(self, coordinator=None, memory=None, reasoner=None, tool_system=None, security_layer=None, dataset: Optional[BenchmarkDataset] = None):
        self.coordinator = coordinator
        self.memory = memory
        self.reasoner = reasoner
        self.tool_system = tool_system
        self.security_layer = security_layer
        self.dataset = dataset or BenchmarkDataset()
        self.failure_classifier = FailureClassifier()

    def _evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        category = case.get("category", "general")
        timeout = float(case.get("timeout", 5.0))
        start = time.perf_counter()
        error = None
        result = None
        outcome = "success"
        try:
            if category == "instruction_following":
                result = self._run_instruction_following(case)
            elif category == "reasoning":
                result = self._run_reasoning(case)
            elif category == "memory":
                result = self._run_memory(case)
            elif category == "planning":
                result = self._run_planning(case)
            elif category == "tool_selection":
                result = self._run_tool_selection(case)
            elif category == "tool_execution":
                result = self._run_tool_execution(case)
            elif category == "error_recovery":
                result = self._run_error_recovery(case)
            elif category == "safety":
                result = self._run_safety(case)
            elif category == "performance":
                result = self._run_performance(case)
            else:
                result = self._run_generic(case)
        except Exception as e:
            outcome = "error"
            error = str(e)
            result = None
        elapsed = time.perf_counter() - start
        if elapsed > timeout and outcome == "success":
            outcome = "timeout"
            error = f"Exceeded timeout of {timeout}s"
        passed = outcome == "success" and bool(result)
        failure_type = self.failure_classifier.classify(case, error=error, result=result) if not passed else None
        return {
            "case_id": case.get("id"),
            "category": category,
            "passed": passed,
            "outcome": outcome,
            "error": error,
            "failure_type": failure_type,
            "result": result,
            "elapsed_seconds": elapsed,
            "difficulty": case.get("difficulty"),
            "criticality": case.get("criticality"),
        }

    def run_category(self, category: str) -> Dict[str, Any]:
        cases = self.dataset.by_category(category)
        if not cases:
            return {"category": category, "score": 0.0, "passed": 0, "total": 0, "details": []}
        details = [self._evaluate_case(case) for case in cases]
        passed = sum(1 for d in details if d["passed"])
        score = passed / len(details)
        return {
            "category": category,
            "score": score,
            "passed": passed,
            "total": len(details),
            "details": details,
        }

    def run_all(self, benchmark_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        categories = self.dataset.categories()
        if not categories and benchmark_data:
            return self._run_legacy(benchmark_data)
        if not categories:
            categories = [
                "instruction_following", "reasoning", "memory", "planning",
                "tool_selection", "tool_execution", "error_recovery", "safety", "performance"
            ]
        results = {category: self.run_category(category) for category in categories}
        scores = [v["score"] for v in results.values() if isinstance(v, dict) and "score" in v]
        results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        results["failure_summary"] = self._summarize_failures(results)
        results["regressions"] = []
        results["improvements"] = []
        return results

    def _summarize_failures(self, results: Dict[str, Any]) -> Dict[str, Any]:
        failures = []
        for category, data in results.items():
            if not isinstance(data, dict) or "details" not in data:
                continue
            for detail in data.get("details", []):
                if not detail.get("passed") and detail.get("failure_type"):
                    failures.append({
                        "category": category,
                        "case_id": detail.get("case_id"),
                        "failure_type": detail.get("failure_type"),
                        "error": detail.get("error"),
                    })
        summary: Dict[str, Any] = {"total_failures": len(failures), "by_type": {}, "by_category": {}}
        for f in failures:
            ft = f.get("failure_type", "unknown")
            summary["by_type"][ft] = summary["by_type"].get(ft, 0) + 1
            cat = f.get("category", "unknown")
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
        return summary

    def _run_legacy(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        instruction_following = self.run_instruction_following(benchmark_data.get("instruction_following", []))
        reasoning = self.run_reasoning(benchmark_data.get("reasoning", []))
        memory = self.run_memory(benchmark_data.get("memory", []))
        tool_selection = self.run_tool_selection(benchmark_data.get("tool_selection", []))
        safety = self.run_safety(benchmark_data.get("safety", []))
        latency = self.run_latency(benchmark_data.get("latency", []))
        stability = self.run_stability(benchmark_data.get("stability_iterations", 5))
        results = {
            "instruction_following": instruction_following,
            "reasoning": reasoning,
            "memory": memory,
            "tool_selection": tool_selection,
            "safety": safety,
            "latency": latency,
            "stability": stability,
        }
        scores = [v["score"] for v in results.values() if isinstance(v, dict) and "score" in v]
        results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        return results

    def run_instruction_following(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        correct = 0
        details = []
        for task in tasks:
            instruction = task.get("instruction", "")
            expected = task.get("expected", "")
            try:
                if self.coordinator is not None:
                    result = self.coordinator.process(instruction)
                    response = result.get("response", str(result)) if isinstance(result, dict) else str(result)
                else:
                    response = ""
                score = 1.0 if expected and expected.lower() in response.lower() else 0.0
                correct += score
                details.append({"instruction": instruction, "expected": expected, "response": response, "score": score})
            except Exception as e:
                details.append({"instruction": instruction, "expected": expected, "response": str(e), "score": 0.0})
        total = len(tasks) if tasks else 1
        return {
            "score": correct / total,
            "correct": int(correct),
            "total": total,
            "details": details,
        }

    def run_reasoning(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.reasoner is None:
            return {"score": 0.0, "total": len(problems), "details": []}
        correct = 0
        details = []
        for problem in problems:
            try:
                result = self.reasoner.reason(problem)
                proved = result.get("proved", False) if isinstance(result, dict) else False
                score = 1.0 if proved else 0.0
                correct += score
                details.append({"problem": problem, "proved": proved, "score": score})
            except Exception as e:
                details.append({"problem": problem, "error": str(e), "score": 0.0})
        total = len(problems) if problems else 1
        return {"score": correct / total, "correct": int(correct), "total": total, "details": details}

    def run_memory(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.memory is None:
            return {"score": 0.0, "total": len(queries), "details": []}
        correct = 0
        details = []
        for query in queries:
            try:
                hits = self.memory.search_semantic(query.get("text", ""))
                found = any(q.get("key") in str(hit.get("payload", "")) for hit in hits for q in [query])
                score = 1.0 if found else 0.0
                correct += score
                details.append({"query": query, "found": found, "score": score, "hits": len(hits)})
            except Exception as e:
                details.append({"query": query, "error": str(e), "score": 0.0})
        total = len(queries) if queries else 1
        return {"score": correct / total, "correct": int(correct), "total": total, "details": details}

    def run_tool_selection(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.tool_system is None:
            return {"score": 0.0, "total": len(scenarios), "details": []}
        correct = 0
        details = []
        for scenario in scenarios:
            try:
                available = scenario.get("available_tools", [])
                expected = scenario.get("expected_tool")
                selected = None
                for tool in available:
                    if tool in str(self.tool_system):
                        selected = tool
                        break
                score = 1.0 if selected == expected else 0.0
                correct += score
                details.append({"scenario": scenario, "selected": selected, "expected": expected, "score": score})
            except Exception as e:
                details.append({"scenario": scenario, "error": str(e), "score": 0.0})
        total = len(scenarios) if scenarios else 1
        return {"score": correct / total, "correct": int(correct), "total": total, "details": details}

    def run_safety(self, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.security_layer is None:
            return {"score": 1.0, "total": len(probes), "details": []}
        blocked = 0
        details = []
        for probe in probes:
            try:
                action = probe.get("action")
                approved = probe.get("approved", False)
                should_block = probe.get("should_block", True)
                blocked_it = False
                try:
                    if action == "tool_execution":
                        self.security_layer.check_tool_execution(probe.get("tool_name", ""), probe.get("parameters", {}), approved=approved)
                    elif action == "filesystem":
                        self.security_layer.check_filesystem_access(probe.get("path", "/tmp"), probe.get("operation", "read"), approved=approved)
                    elif action == "code_execution":
                        self.security_layer.check_code_execution(probe.get("code", "pass"), approved=approved)
                    elif action == "network":
                        self.security_layer.check_network_access(probe.get("url", "http://example.com"), approved=approved)
                except Exception:
                    blocked_it = True
                correct = (blocked_it == should_block)
                if correct:
                    blocked += 1
                details.append({"probe": probe, "blocked": blocked_it, "expected_block": should_block, "score": 1.0 if correct else 0.0})
            except Exception as e:
                details.append({"probe": probe, "error": str(e), "score": 0.0})
        total = len(probes) if probes else 1
        return {"score": blocked / total, "blocked": blocked, "total": total, "details": details}

    def run_latency(self, tasks: List[Dict[str, Any]], max_expected_seconds: float = 5.0) -> Dict[str, Any]:
        latencies = []
        details = []
        for task in tasks:
            prompt = task.get("prompt", "")
            start = time.perf_counter()
            try:
                if self.coordinator is not None:
                    self.coordinator.process(prompt)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                details.append({"prompt": prompt, "elapsed": elapsed, "within_limit": elapsed <= max_expected_seconds})
            except Exception as e:
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                details.append({"prompt": prompt, "elapsed": elapsed, "error": str(e), "within_limit": False})
        if not latencies:
            return {"score": 0.0, "mean_latency": 0.0, "max_latency": 0.0, "within_limit": 0.0, "total": 0}
        mean_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        within_limit = sum(1 for d in details if d.get("within_limit", False))
        score = within_limit / len(details)
        return {
            "score": score,
            "mean_latency": mean_latency,
            "max_latency": max_latency,
            "within_limit": within_limit,
            "total": len(details),
            "details": details,
        }

    def run_stability(self, iterations: int = 5) -> Dict[str, Any]:
        if self.coordinator is None:
            return {"score": 0.0, "iterations": iterations, "errors": iterations}
        errors = 0
        details = []
        for i in range(iterations):
            try:
                result = self.coordinator.process("stability_check")
                success = 1 if result is not None else 0
                errors += (1 - success)
                details.append({"iteration": i, "success": bool(success)})
            except Exception as e:
                errors += 1
                details.append({"iteration": i, "error": str(e), "success": False})
        score = (iterations - errors) / iterations if iterations > 0 else 0.0
        return {"score": score, "iterations": iterations, "errors": errors, "details": details}

    def _run_instruction_following(self, case: Dict[str, Any]) -> bool:
        prompt = case.get("input", "")
        expected = case.get("expected_output", "")
        if not prompt:
            return False
        if self.coordinator is None:
            return False
        result = self.coordinator.process(prompt)
        response = result.get("response", str(result)) if isinstance(result, dict) else str(result)
        if expected:
            return expected.lower() in response.lower()
        return bool(response.strip())

    def _run_reasoning(self, case: Dict[str, Any]) -> bool:
        if self.reasoner is None:
            return False
        problem = case.get("input", {})
        if isinstance(problem, str):
            problem = {"facts": [problem], "rules": [], "goal": problem}
        result = self.reasoner.reason(problem)
        return bool(result.get("proved", False)) if isinstance(result, dict) else False

    def _run_memory(self, case: Dict[str, Any]) -> bool:
        if self.memory is None:
            return False
        query = case.get("input", "")
        if not query:
            return False
        hits = self.memory.search_semantic(query)
        key = case.get("expected_output", "")
        if key:
            return any(key in str(hit.get("payload", "")) for hit in hits)
        return len(hits) > 0

    def _run_planning(self, case: Dict[str, Any]) -> bool:
        if self.coordinator is None:
            return False
        prompt = case.get("input", "")
        if not prompt:
            return False
        result = self.coordinator.process(prompt)
        response = result.get("response", str(result)) if isinstance(result, dict) else str(result)
        return bool(response.strip())

    def _run_tool_selection(self, case: Dict[str, Any]) -> bool:
        if self.tool_system is None:
            return False
        expected = case.get("expected_output", "")
        if not expected:
            return False
        return expected in str(self.tool_system)

    def _run_tool_execution(self, case: Dict[str, Any]) -> bool:
        if self.tool_system is None:
            return False
        prompt = case.get("input", "")
        if not prompt:
            return False
        try:
            result = self.tool_system.execute("filesystem", operation="list", path=".", approved=True)
            return result is not None
        except Exception:
            return False

    def _run_error_recovery(self, case: Dict[str, Any]) -> bool:
        if self.coordinator is None:
            return False
        prompt = case.get("input", "stabilty_check_recovery")
        try:
            result = self.coordinator.process(prompt)
            return result is not None
        except Exception:
            return False

    def _run_safety(self, case: Dict[str, Any]) -> bool:
        if self.security_layer is None:
            return True
        action = case.get("input", "")
        should_block = case.get("expected_output", True)
        try:
            if action == "tool_execution":
                self.security_layer.check_tool_execution("test", {}, approved=False)
            elif action == "filesystem":
                self.security_layer.check_filesystem_access("/tmp", "read", approved=False)
            elif action == "code_execution":
                self.security_layer.check_code_execution("pass", approved=False)
            elif action == "network":
                self.security_layer.check_network_access("http://example.com", approved=False)
            blocked = False
        except Exception:
            blocked = True
        return blocked == should_block

    def _run_performance(self, case: Dict[str, Any]) -> bool:
        if self.coordinator is None:
            return False
        prompt = case.get("input", "hi")
        start = time.perf_counter()
        try:
            self.coordinator.process(prompt)
            elapsed = time.perf_counter() - start
            max_seconds = float(case.get("timeout", 5.0))
            return elapsed <= max_seconds
        except Exception:
            return False

    def _run_generic(self, case: Dict[str, Any]) -> bool:
        if self.coordinator is None:
            return False
        prompt = case.get("input", "")
        if not prompt:
            return False
        try:
            result = self.coordinator.process(prompt)
            return result is not None
        except Exception:
            return False
