"""
Performance profiler for JARVIS 2.0.

Measures:
  - RAM usage (Python and system)
  - CPU usage (user/system time)
  - Training and inference throughput
  - Hardware info

Uses only the standard library: no external dependencies.
"""

import os
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def _hardware_info() -> Dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
    }


def _get_cpu_times() -> Dict[str, float]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": r.ru_utime,
        "system_seconds": r.ru_stime,
        "max_rss_bytes": getattr(r, "ru_maxrss", 0) * 1024,
    }


def _get_python_memory() -> Dict[str, Optional[int]]:
    try:
        current, peak = tracemalloc.get_traced_memory()
        return {
            "tracemalloc_current_bytes": current,
            "tracemalloc_peak_bytes": peak,
        }
    except Exception:
        return {
            "tracemalloc_current_bytes": None,
            "tracemalloc_peak_bytes": None,
        }


@dataclass
class PerformanceSnapshot:
    label: str
    cpu: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def profile_section(label: str, fn, *args, **kwargs) -> Any:
    """Profile a callable and return its result alongside performance metrics."""
    tracemalloc.start()
    start = time.perf_counter()
    cpu_before = _get_cpu_times()
    result = fn(*args, **kwargs)
    cpu_after = _get_cpu_times()
    elapsed = time.perf_counter() - start
    mem = _get_python_memory()
    tracemalloc.stop()

    snapshot = PerformanceSnapshot(
        label=label,
        cpu={
            "user_seconds": cpu_after["user_seconds"] - cpu_before["user_seconds"],
            "system_seconds": cpu_after["system_seconds"] - cpu_before["system_seconds"],
            "max_rss_bytes": cpu_after["max_rss_bytes"],
        },
        memory=mem,
        hardware=_hardware_info(),
        elapsed_seconds=elapsed,
    )
    return result, snapshot


def profile_training_step(trainer, batch_inputs, batch_targets) -> Dict[str, Any]:
    """Profile a single training step."""
    def _step():
        return trainer.train_step(batch_inputs, batch_targets)

    _, snap = profile_section("training_step", _step)
    return snap.to_dict()


def profile_inference(model, batch_inputs) -> Dict[str, Any]:
    """Profile a single inference forward pass."""
    def _forward():
        return model.forward(batch_inputs, training=False)

    _, snap = profile_section("inference", _forward)
    return snap.to_dict()
