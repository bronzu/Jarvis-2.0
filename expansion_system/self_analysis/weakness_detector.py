import time
from typing import Any, Dict, List, Optional


class WeaknessDetector:
    WEAKNESS_TYPES = {
        "occasional_error": "Errore occasionale",
        "systematic_error": "Errore sistematico",
        "regression": "Regressione",
        "capability_gap": "Mancanza di capacità",
        "model_problem": "Problema del modello",
        "prompt_problem": "Problema del prompt/context",
        "planner_problem": "Problema del planner",
        "tool_use_problem": "Problema del tool-use",
        "memory_problem": "Problema della memoria",
        "infrastructure_problem": "Problema dell'infrastruttura",
    }

    CRITICAL_TYPES = {
        "regression",
        "model_problem",
        "infrastructure_problem",
    }

    BENCHMARK_FAILURE_TYPES = {
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

    def detect(self, metrics, threshold, contexts: Optional[List[Dict[str, Any]]] = None, failures: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        results = []
        contexts = contexts or []
        failures = failures or []
        for item in metrics:
            if isinstance(item, dict):
                name = item.get("name") or item.get("metric")
                value = item.get("value")
                if name is None or value is None:
                    continue
            else:
                continue
            if value > threshold:
                weakness_type = self._classify(item, contexts, failures)
                severity = self._severity(value, threshold, weakness_type)
                results.append({
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "type": weakness_type,
                    "severity": severity,
                    "proposal": self._propose_action(weakness_type),
                    "timestamp": item.get("timestamp", time.time()),
                    "context": item.get("context", {}),
                })
        for failure in failures:
            failure_type = failure.get("failure_type") or failure.get("type")
            if failure_type and failure_type in self.BENCHMARK_FAILURE_TYPES:
                weakness_type = self.BENCHMARK_FAILURE_TYPES[failure_type]
                results.append({
                    "metric": failure.get("category", failure_type),
                    "value": failure.get("elapsed_seconds", 1.0),
                    "threshold": 0.0,
                    "type": weakness_type,
                    "severity": 4 if weakness_type in self.CRITICAL_TYPES else 3,
                    "proposal": self._propose_action(weakness_type),
                    "timestamp": failure.get("timestamp", time.time()),
                    "context": {"source": "benchmark_failure", "failure_type": failure_type},
                })
        results.sort(key=lambda w: w["severity"], reverse=True)
        return results

    def _classify(self, metric_item: Dict[str, Any], contexts: List[Dict[str, Any]], failures: Optional[List[Dict[str, Any]]] = None) -> str:
        failures = failures or []
        context = metric_item.get("context", {})
        text = " ".join(str(v) for v in context.values()).lower()
        name = str(metric_item.get("name", "")).lower()
        if "regression" in text or "previous" in text:
            return "regression"
        if any(k in name for k in ["memory", "memoria"]):
            return "memory_problem"
        if any(k in name for k in ["tool", "tool_success"]):
            return "tool_use_problem"
        if any(k in name for k in ["plan", "piano", "planner"]):
            return "planner_problem"
        if any(k in name for k in ["prompt", "context", "input"]):
            return "prompt_problem"
        if any(k in name for k in ["model", "neural", "inference"]):
            return "model_problem"
        if any(k in name for k in ["latency", "timeout", "resource", "cpu", "ram"]):
            return "infrastructure_problem"
        if any(k in text for k in ["missing", "gap", "not found", "mancante"]):
            return "capability_gap"
        if any(k in text for k in ["sometimes", "occasional", "occasionalmente"]):
            return "occasional_error"
        if any(k in text for k in ["always", "sempre", "systematic"]):
            return "systematic_error"
        for failure in failures:
            ft = failure.get("failure_type") or failure.get("type")
            if ft and ft in self.BENCHMARK_FAILURE_TYPES:
                return self.BENCHMARK_FAILURE_TYPES[ft]
        return "occasional_error"

    def _severity(self, value: float, threshold: float, weakness_type: str) -> int:
        ratio = value / threshold if threshold > 0 else float("inf")
        base = 1 if ratio <= 1.5 else (2 if ratio <= 3.0 else 3)
        if weakness_type in self.CRITICAL_TYPES:
            base += 1
        return min(base, 5)

    def _propose_action(self, weakness_type: str) -> str:
        proposals = {
            "occasional_error": "rieseguire_e_valutare",
            "systematic_error": "investigare_causa_root",
            "regression": "confrontare_baseline_e_rollback_se_necessario",
            "capability_gap": "creare_capacità_o_tool_mancante",
            "model_problem": "valutare_riaddestramento_o_prompt",
            "prompt_problem": "ottimizzare_prompt_o_context",
            "planner_problem": "migliorare_pianificazione_o_decomposizione",
            "tool_use_problem": "verificare_registry_e_selezione_tool",
            "memory_problem": "analizzare_retrieval_e_consolidamento",
            "infrastructure_problem": "controllare_risorse_e_latenza",
        }
        return proposals.get(weakness_type, "investigare")

    def classify_benchmark_failures(self, benchmark_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        failures = []
        for category, data in benchmark_results.items():
            if not isinstance(data, dict) or "details" not in data:
                continue
            for detail in data.get("details", []):
                if detail.get("passed"):
                    continue
                failure_type = detail.get("failure_type")
                if not failure_type:
                    continue
                failures.append({
                    "category": category,
                    "case_id": detail.get("case_id"),
                    "failure_type": failure_type,
                    "error": detail.get("error"),
                    "elapsed_seconds": detail.get("elapsed_seconds"),
                    "timestamp": time.time(),
                })
        return failures

    def weaknesses_from_benchmark(self, benchmark_results: Dict[str, Any], threshold: float = 0.0) -> List[Dict[str, Any]]:
        failures = self.classify_benchmark_failures(benchmark_results)
        return self.detect([], threshold=threshold, failures=failures)
