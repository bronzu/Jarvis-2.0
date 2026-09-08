from expansion_system.self_analysis.performance_monitor import PerformanceMonitor
from expansion_system.self_analysis.weakness_detector import WeaknessDetector


def test_self_analysis_identifies_anomaly():
    monitor = PerformanceMonitor()
    monitor.record("latency", 12)
    metrics = monitor.metrics()
    names = [m["name"] if isinstance(m, dict) else m for m in metrics]
    assert "latency" in names
    weaknesses = WeaknessDetector().detect(metrics, 10)
    assert len(weaknesses) >= 1
    assert weaknesses[0]["metric"] == "latency"


def test_self_analysis_ignores_metrics_within_threshold():
    monitor = PerformanceMonitor()
    monitor.record("latency", 5)
    weaknesses = WeaknessDetector().detect(monitor.metrics(), 10)
    assert weaknesses == []
