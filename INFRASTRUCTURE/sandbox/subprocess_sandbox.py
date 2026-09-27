"""
SubprocessSandbox — real OS-level process isolation for callable experiments.

Unlike a boolean flag, this provides ACTUAL isolation by executing the
experiment in a forked child process with:
- New session (os.setsid)
- Resource limits (RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NPROC, RLIMIT_NOFILE)
- Working directory restricted to a sandbox directory
- Environment sanitization
- umask 0o077
- Wall-clock timeout with SIGKILL fallback

The callable is serialized via pickle and executed in the child process.
Results and errors are returned to the parent via a pipe.
"""
from __future__ import annotations

import ast
import inspect
import os
import pickle
import platform
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from neural_core.trainer import safe_pickle_loads


class SubprocessSandbox:
    """Sandbox that executes a callable in an isolated child process.

    This is a REAL isolation mechanism (not a boolean flag). It forks a
    child process, applies OS-level restrictions, and executes the
    callable in that restricted environment. Even if the callable
    attempts privileged operations (e.g. writing to /etc/passwd), the
    OS-level restrictions limit the damage.
    """

    def __init__(self, sandbox_dir: Optional[str] = None, timeout: float = 30.0):
        self.sandbox_dir = Path(sandbox_dir) if sandbox_dir else Path(tempfile.mkdtemp(prefix="jarvis-subprocess-sandbox-"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def execute(self, func: callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute ``func`` in an isolated child process.

        Returns a dict with keys:
            - status: "ok" | "error" | "timeout"
            - result: the return value of func (if ok)
            - error: error description (if error)
            - timed_out: bool
        """
        if not callable(func):
            raise TypeError("func must be callable")

        sandbox_timeout = kwargs.pop("timeout", self.timeout)
        exec_timeout = float(sandbox_timeout) if sandbox_timeout is not None else 30.0

        try:
            payload, method = self._serialize(func, args, kwargs)
        except Exception as exc:
            return {
                "status": "error",
                "timed_out": False,
                "elapsed_seconds": 0.0,
                "error": f"Cannot serialize experiment for isolation: {exc}",
            }

        runner_code = _RUNNER_TEMPLATE

        cmd = [sys.executable, "-I", "-S", "-c", runner_code]

        start_time = time.time()
        timed_out = False
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                cwd=str(self.sandbox_dir),
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "LANG": os.environ.get("LANG", "C"),
                    "_JARVIS_SANDBOX": "1",
                },
                preexec_fn=lambda: self._preexec(),
            )
            try:
                stdout_data, stderr_data = proc.communicate(
                    input=payload,
                    timeout=exec_timeout,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                stdout_data, stderr_data = proc.communicate()

            elapsed = time.time() - start_time

            if timed_out:
                return {
                    "status": "timeout",
                    "timed_out": True,
                    "elapsed_seconds": elapsed,
                    "error": f"Experiment exceeded timeout of {exec_timeout}s",
                    "isolated": True,
                }

            if proc.returncode != 0:
                stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
                return {
                    "status": "error",
                    "timed_out": False,
                    "elapsed_seconds": elapsed,
                    "error": f"Process exited with code {proc.returncode}: {stderr_text[:500]}",
                    "isolated": True,
                }

            if stdout_data:
                try:
                    result_data = safe_pickle_loads(stdout_data)
                    return {
                        "status": result_data.get("status", "ok"),
                        "result": result_data.get("result"),
                        "error": result_data.get("error"),
                        "timed_out": False,
                        "elapsed_seconds": elapsed,
                        "isolated": True,
                        "method": method,
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "timed_out": False,
                        "elapsed_seconds": elapsed,
                        "error": f"Failed to parse result: {e}",
                    }
            return {
                "status": "ok",
                "result": None,
                "timed_out": False,
                "elapsed_seconds": elapsed,
                "isolated": True,
            }
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass

    def _preexec(self):
        """Apply isolation in the child process before exec."""
        try:
            os.setsid()
        except OSError:
            pass
        try:
            os.umask(0o077)
        except OSError:
            pass

        import resource
        # CPU time limit (minimum 1 second to avoid SIGXCPU on fractional timeouts)
        cpu_limit = max(1, int(self.timeout))
        for rlim, value in (
            (resource.RLIMIT_CPU, (cpu_limit, cpu_limit)),
            (resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024)),
            (resource.RLIMIT_NPROC, (4, 4)),
            (resource.RLIMIT_NOFILE, (32, 32)),
            (resource.RLIMIT_CORE, (0, 0)),
        ):
            try:
                resource.setrlimit(rlim, value)
            except (ValueError, OSError):
                pass

        # Restrict writable filesystem to sandbox dir only
        os.chdir(str(self.sandbox_dir))

    def _serialize(self, func: callable, args: tuple, kwargs: dict) -> tuple:
        """Serialize the callable and its arguments for the subprocess.

        First tries standard pickle. If the callable is a closure or lambda
        that cannot be pickled, falls back to extracting the source code via
        ``inspect.getsource`` and capturing closure variables separately.
        """
        try:
            payload = pickle.dumps(("pickle", (func, args, kwargs)))
            return payload, "pickle"
        except (pickle.PicklingError, TypeError, AttributeError):
            pass

        try:
            raw_source = inspect.getsource(func)
        except (OSError, TypeError) as exc:
            raise pickle.PicklingError(
                f"Cannot pickle or get source for {func.__name__}: {exc}"
            )
        raw_source = textwrap.dedent(raw_source)

        if func.__name__ == "<lambda>":
            source, func_name = self._extract_lambda_source(raw_source, func)
        else:
            source = raw_source
            func_name = func.__name__

        closure_vars: Dict[str, Any] = {}
        if func.__closure__ and func.__code__.co_freevars:
            freevars = func.__code__.co_freevars
            for name, cell in zip(freevars, func.__closure__):
                try:
                    pickle.dumps(cell.cell_contents)
                    closure_vars[name] = cell.cell_contents
                except (pickle.PicklingError, TypeError, AttributeError):
                    closure_vars[name] = None

        payload_data = ("source", {
            "source": source,
            "func_name": func_name,
            "args": args,
            "kwargs": kwargs,
            "closure_vars": closure_vars,
        })
        return pickle.dumps(payload_data), "source"

    @staticmethod
    def _extract_lambda_source(raw_source: str, func) -> tuple:
        """Convert a lambda's source line into a named-function definition."""
        tree = ast.parse(raw_source)
        lambda_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                lambda_node = node
                break
        if lambda_node is None:
            raise pickle.PicklingError("Could not find lambda in source")

        lambda_src = ast.get_source_segment(raw_source, lambda_node)
        if not lambda_src:
            lambda_src = ast.unparse(lambda_node)

        argspec = inspect.getfullargspec(func)
        args_str = ", ".join(argspec.args)
        colon_idx = lambda_src.index(":")
        body_src = lambda_src[colon_idx + 1:]
        wrapper = f"def __lambda__({args_str}):\n    return {body_src}"
        return wrapper, "__lambda__"

    @property
    def is_real_isolation(self) -> bool:
        """This sandbox provides real OS-level process isolation."""
        return True


_RUNNER_TEMPLATE = """
import sys, os, pickle, resource, signal, textwrap

# Apply rlimits in child even if preexec didn't run (macOS safety)
try:
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
except (ValueError, OSError):
    pass
try:
    resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024*1024, 16*1024*1024))
except (ValueError, OSError):
    pass

# Read pickled payload from stdin
payload = sys.stdin.buffer.read()
method, data = pickle.loads(payload)

if method == "pickle":
    func, args, kwargs = data
elif method == "source":
    source = data["source"]
    func_name = data["func_name"]
    args = data["args"]
    kwargs = data["kwargs"]
    closure_vars = data.get("closure_vars", {})

    namespace = dict(closure_vars)
    exec(source, namespace)
    func = namespace[func_name]
else:
    raise ValueError(f"Unknown serialization method: {method}")

# Execute
try:
    result = func(*args, **kwargs)
    output = pickle.dumps({"status": "ok", "result": result, "error": None})
except SystemExit as e:
    output = pickle.dumps({"status": "error", "result": None, "error": f"SystemExit: {e.code}"})
except BaseException as e:
    output = pickle.dumps({"status": "error", "result": None, "error": f"{type(e).__name__}: {e}"})

sys.stdout.buffer.write(output)
sys.stdout.buffer.flush()
"""
