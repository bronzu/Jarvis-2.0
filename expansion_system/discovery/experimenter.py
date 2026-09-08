import time
import traceback
from typing import Any, Callable, Dict, List, Optional


class Experimenter:
    """Esegue esperimenti isolati nella sandbox e restituisce risultati strutturati."""

    def run(self, experiment, sandbox=None, timeout: Optional[float] = None, resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not callable(experiment):
            raise TypeError("Esperimento non eseguibile")

        started = time.time()
        resource_limits = resource_limits or {}
        max_memory_mb = resource_limits.get("max_memory_mb")
        max_cpu_percent = resource_limits.get("max_cpu_percent")
        result = None
        outcome = "success"
        error = None
        elapsed = 0.0
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()

        try:
            if timeout is not None:
                result = self._run_with_timeout(experiment, timeout)
            else:
                result = experiment()
        except TimeoutError:
            outcome = "timeout"
            error = "Experiment exceeded timeout"
            result = None
        except Exception as e:
            outcome = "error"
            error = str(e)
            result = None
        elapsed = time.time() - started
        memory_after = self._get_memory_usage()
        cpu_after = self._get_cpu_usage()

        observation = {
            "outcome": outcome,
            "result": result,
            "error": error,
            "traceback": traceback.format_exc() if error else None,
            "elapsed_seconds": elapsed,
            "sandbox": bool(sandbox),
            "timeout": timeout,
            "resource_usage": {
                "memory_delta_mb": max(0.0, memory_after - memory_before),
                "cpu_time_estimate": max(0.0, cpu_after - cpu_before),
            },
            "timestamp": time.time(),
        }
        return observation

    def _run_with_timeout(self, experiment, timeout):
        import signal

        def handler(signum, frame):
            raise TimeoutError("Experiment timed out")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(timeout) + 1)
        try:
            result = experiment()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result

    def _get_memory_usage(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_utime
        except Exception:
            return 0.0
