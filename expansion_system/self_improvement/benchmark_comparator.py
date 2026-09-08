import statistics
from typing import Any, Dict, List, Optional


class BenchmarkComparator:
    DEFAULT_METRICS = {
        "logical_accuracy": {"direction": "higher", "critical": True, "weight": 0.15, "threshold": 0.05},
        "execution_time": {"direction": "lower", "critical": False, "weight": 0.10, "threshold": 0.10},
        "memory_usage": {"direction": "lower", "critical": False, "weight": 0.10, "threshold": 0.15},
        "json_adherence": {"direction": "higher", "critical": True, "weight": 0.10, "threshold": 0.03},
        "tool_success": {"direction": "higher", "critical": True, "weight": 0.15, "threshold": 0.05},
        "reasoning_score": {"direction": "higher", "critical": True, "weight": 0.15, "threshold": 0.05},
        "memory_score": {"direction": "higher", "critical": False, "weight": 0.10, "threshold": 0.05},
        "planning_score": {"direction": "higher", "critical": False, "weight": 0.10, "threshold": 0.05},
        "safety_score": {"direction": "higher", "critical": True, "weight": 0.10, "threshold": 0.02},
    }

    def __init__(self, metrics_config: Optional[Dict[str, Any]] = None):
        self.metrics_config = dict(self.DEFAULT_METRICS)
        if metrics_config:
            self.metrics_config.update(metrics_config)

    def compare(self, active_score, candidate_score):
        if candidate_score > active_score:
            return "approval_required"
        return "reject"

    def compare_versions(self, current_version, candidate_version, current_metrics, candidate_metrics, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if not isinstance(current_metrics, dict) or not isinstance(candidate_metrics, dict):
            raise TypeError("Le metriche devono essere dizionari.")

        thresholds = thresholds or {}
        deltas = {}
        wins_current = 0
        wins_candidate = 0
        ties = 0
        critical_regressions = []
        category_regressions: Dict[str, List[str]] = {}
        metric_results = {}

        for metric, cfg in self.metrics_config.items():
            current_value = current_metrics.get(metric)
            candidate_value = candidate_metrics.get(metric)
            if current_value is None and candidate_value is None:
                continue
            if current_value is None:
                current_value = 0.0
            if candidate_value is None:
                candidate_value = 0.0

            delta = candidate_value - current_value
            deltas[metric] = {
                "current": current_value,
                "candidate": candidate_value,
                "delta": delta,
            }
            direction = cfg["direction"]
            if direction == "higher":
                is_better = candidate_value > current_value
                is_equal = candidate_value == current_value
            else:
                is_better = candidate_value < current_value
                is_equal = candidate_value == current_value

            threshold = thresholds.get(metric, cfg.get("threshold", 0.0))
            if current_value != 0:
                relative_delta = abs(delta / current_value)
            else:
                relative_delta = abs(delta)

            regression = not is_better and not is_equal and relative_delta > threshold
            if regression and cfg.get("critical", False):
                critical_regressions.append(metric)
            if regression:
                category_regressions.setdefault(metric, []).append("regression")

            if is_better:
                wins_candidate += 1
            elif is_equal:
                ties += 1
            else:
                wins_current += 1

            metric_results[metric] = {
                "current": current_value,
                "candidate": candidate_value,
                "delta": delta,
                "direction": direction,
                "is_better": is_better,
                "regression": regression,
                "critical": cfg.get("critical", False),
            }

        total_compared = wins_current + wins_candidate + ties
        if total_compared == 0:
            decision = "REJECT"
            reason = "no_metrics"
        elif critical_regressions:
            decision = "REJECT"
            reason = "critical_regression"
        elif wins_candidate > wins_current:
            decision = "ACCEPT"
            reason = "improved"
        elif wins_candidate == wins_current and wins_candidate > 0:
            decision = "ACCEPT"
            reason = "tie_broken_by_non_critical"
        else:
            decision = "REJECT"
            reason = "no_improvement"

        summary = {
            "winner": candidate_version if decision == "ACCEPT" else current_version,
            "deltas": deltas,
            "metric_results": metric_results,
            "wins_current": wins_current,
            "wins_candidate": wins_candidate,
            "ties": ties,
            "total_compared": total_compared,
            "critical_regressions": critical_regressions,
            "category_regressions": category_regressions,
            "decision": decision,
            "reason": reason,
        }
        report = {
            "decision": decision,
            "reason": reason,
            "winner": summary["winner"],
            "summary": deltas,
            "comparison": {
                "current_version": current_version,
                "candidate_version": candidate_version,
                "current_metrics": current_metrics,
                "candidate_metrics": candidate_metrics,
            },
            "critical_regressions": critical_regressions,
            "category_regressions": category_regressions,
            "overall_delta": self._weighted_delta(current_metrics, candidate_metrics),
        }
        return report

    def detect_regression(self, current_metrics, candidate_metrics) -> List[str]:
        regressions = []
        for metric, cfg in self.metrics_config.items():
            current_value = current_metrics.get(metric)
            candidate_value = candidate_metrics.get(metric)
            if current_value is None or candidate_value is None:
                continue
            direction = cfg["direction"]
            is_better = (direction == "higher" and candidate_value > current_value) or (direction == "lower" and candidate_value < current_value)
            if not is_better and cfg.get("critical", False):
                regressions.append(metric)
        return regressions

    def _weighted_delta(self, current_metrics: Dict[str, Any], candidate_metrics: Dict[str, Any]) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for metric, cfg in self.metrics_config.items():
            current_value = current_metrics.get(metric)
            candidate_value = candidate_metrics.get(metric)
            if current_value is None or candidate_value is None:
                continue
            weight = cfg.get("weight", 0.1)
            total_weight += weight
            if cfg["direction"] == "higher":
                weighted_sum += (candidate_value - current_value) * weight
            else:
                weighted_sum += (current_value - candidate_value) * weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0
