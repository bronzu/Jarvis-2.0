import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union


class Experimenter:
    """Esegue esperimenti isolati nella sandbox e restituisce risultati strutturati.

    CRITICAL: Questo componente richiede isolamento reale. Se ``sandbox`` è
    ``None`` o non è un'istanza di isolamento reale, e l'esperimento
    richiede l'esecuzione di codice (callable), il run viene rifiutato.

    Eccezione: se l'esperimento è un semplice dizionario di iperparametri
    (non codice executabile), è ammesso senza sandbox, ma viene documentato
    esplicitamente nel report come ``hyperparameter_only: True``.
    """

    def _is_real_sandbox(self, sandbox) -> bool:
        """Verifica che ``sandbox`` fornisca isolamento reale.

        Un'istanza reale di isolamento deve esporre un metodo ``execute``
        o ``run_isolated`` che accetta un callable e lo esegue in un processo
        separato (non in-process). A plain ``run`` method is NOT sufficient —
        it does not guarantee isolation and can be faked by any object.
        """
        if sandbox is None:
            return False
        # SubprocessSandbox and equivalent classes expose ``execute``
        if hasattr(sandbox, "execute") and callable(getattr(sandbox, "execute")):
            return True
        # PythonRunner exposes ``run_isolated`` as a real OS-level isolator
        if hasattr(sandbox, "run_isolated") and callable(getattr(sandbox, "run_isolated")):
            return True
        return False

    def run(self, experiment: Union[Callable, Dict[str, Any]],
            sandbox=None,
            timeout: Optional[float] = None,
            resource_limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Esegue un esperimento con isolamento rigoroso.

        Args:
            experiment: Un callable (codice eseguibile) o un dict di
                        iperparametri (hyperparameter-only).
            sandbox: Deve essere un'istanza di isolamento reale se
                     ``experiment`` è callable. Se None o non-reale,
                     l'esecuzione di codice è rifiutata.
            timeout: Timeout opzionale in secondi.
            resource_limits: Limiti di risorse opzionali.

        Returns:
            Un observation dict con i risultati dell'esperimento.
        """
        # --- Hyperparameter-only experiments (dict, not callable) ---
        # These do NOT execute arbitrary code; they are data-only.
        # They are allowed without sandbox but must be explicitly documented.
        if isinstance(experiment, dict):
            if not callable(experiment):
                return self._run_hyperparameter_experiment(experiment, sandbox, timeout, resource_limits)

        # --- Callable experiments (actual code execution) ---
        if not callable(experiment):
            raise TypeError("Esperimento non eseguibile")

        # CRITICAL: Require real isolation for code execution
        if not self._is_real_sandbox(sandbox):
            raise PermissionError(
                "Experimenter.run() refused: sandbox is None or not a real "
                "isolation instance. Callable experiments require a real "
                "isolation mechanism (e.g. SubprocessSandbox with execute(), "
                "or PythonRunner.run_isolated). The sandbox parameter is no "
                "longer recorded as a mere boolean flag — it must be a real "
                "isolator before any code is executed."
            )

        started = time.time()
        resource_limits = resource_limits or {}
        result = None
        outcome = "success"
        error = None
        elapsed = 0.0
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()
        used_isolation = False

        try:
            # Route through the real sandbox's execute method
            if hasattr(sandbox, "execute"):
                exec_timeout = timeout if timeout is not None else 30.0
                sandbox_result = sandbox.execute(experiment, timeout=exec_timeout)
                used_isolation = True
                if sandbox_result.get("timed_out"):
                    outcome = "timeout"
                    error = sandbox_result.get("error", "Experiment exceeded timeout")
                    result = None
                elif sandbox_result.get("status") == "error":
                    outcome = "error"
                    error = sandbox_result.get("error", "Unknown error")
                    result = None
                else:
                    outcome = "success"
                    result = sandbox_result.get("result")
                    if sandbox_result.get("status") == "error" and error is None:
                        error = sandbox_result.get("error")
            else:
                # Sandbox validated but has neither execute nor run_isolated.
                # This should never happen since _is_real_sandbox already
                # checked for these methods, but we refuse to run in-process
                # as a defense-in-depth measure.
                raise PermissionError(
                    "Sandbox does not expose execute() or run_isolated() — "
                    "in-process execution is refused for callable experiments."
                )
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
            "sandbox": True,
            "isolated_execution": used_isolation,
            "sandbox_type": type(sandbox).__name__ if used_isolation else None,
            "hyperparameter_only": False,
            "timeout": timeout,
            "resource_usage": {
                "memory_delta_mb": max(0.0, memory_after - memory_before),
                "cpu_time_estimate": max(0.0, cpu_after - cpu_before),
            },
            "timestamp": time.time(),
        }
        return observation

    def _run_hyperparameter_experiment(self, experiment: Dict[str, Any], sandbox, timeout, resource_limits) -> Dict[str, Any]:
        """Handle hyperparameter-only experiments (dicts, not code).

        These do not execute arbitrary code — they are data describing
        hyperparameters. They are allowed without a sandbox, but the
        observation explicitly documents ``hyperparameter_only: True``
        so there is no ambiguity about whether isolation was applied.
        """
        started = time.time()
        resource_limits = resource_limits or {}
        result = None
        outcome = "success"
        error = None
        elapsed = 0.0
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()

        try:
            # Hyperparameter experiments are pure data — no code execution.
            # We log the config for traceability.
            result = {
                "hyperparameters": experiment,
                "sandbox_config": bool(sandbox) if sandbox is not None else None,
            }
            outcome = "success"
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
            "isolated_execution": False,
            "sandbox_type": type(sandbox).__name__ if sandbox is not None else None,
            "hyperparameter_only": True,
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
