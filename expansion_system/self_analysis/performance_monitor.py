import time
import statistics
from typing import Any, Dict, List, Optional


class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self._metrics: List[Dict[str, Any]] = []
        self._window_size = window_size
        self._baselines: Dict[str, float] = {}

    def record(self, name: str, value: float, context: Optional[Dict[str, Any]] = None, timestamp: Optional[float] = None):
        if value < 0:
            raise ValueError("Una metrica non può essere negativa")
        entry = {
            "name": name,
            "value": value,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "context": dict(context) if context else {},
        }
        self._metrics.append(entry)
        while len(self._metrics) > self._window_size:
            self._metrics.pop(0)
        return entry

    def metrics(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._metrics]

    def set_baseline(self, name: str, value: float):
        if value < 0:
            raise ValueError("Una baseline non può essere negativa")
        self._baselines[name] = float(value)

    def get_baseline(self, name: str) -> Optional[float]:
        return self._baselines.get(name)

    def detect_anomalies(self, name: str, threshold_std: float = 2.0) -> List[Dict[str, Any]]:
        values = [m["value"] for m in self._metrics if m.get("name") == name]
        if len(values) < 2:
            return []
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        if std == 0:
            return []
        anomalies = []
        for m in self._metrics:
            if m.get("name") == name:
                z_score = abs(m["value"] - mean) / std
                if z_score > threshold_std:
                    anomalies.append({
                        "name": name,
                        "value": m["value"],
                        "mean": mean,
                        "std": std,
                        "z_score": z_score,
                        "timestamp": m.get("timestamp"),
                    })
        return anomalies

    def trend(self, name: str, last_n: int = 10) -> Dict[str, Any]:
        values = [m["value"] for m in self._metrics if m.get("name") == name][-last_n:]
        if len(values) < 2:
            return {"trend": "stable", "slope": 0.0, "samples": len(values)}
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0.0
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        return {"trend": trend, "slope": slope, "samples": n}

    def statistics(self, name: str, last_n: Optional[int] = None) -> Dict[str, Any]:
        values = [m["value"] for m in self._metrics if m.get("name") == name]
        if last_n is not None:
            values = values[-last_n:]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    def clear(self):
        self._metrics.clear()
        self._baselines.clear()
