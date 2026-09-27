"""Tests for Finding 9: Experimenter._is_real_sandbox trusts any `run` method.

The _is_real_sandbox method accepted any object with a `run` method as a real
sandbox. But a `run` method doesn't guarantee isolation — an attacker could
provide a fake sandbox with a `run` method that executes code in-process,
bypassing all isolation controls.

Fix: _is_real_sandbox now only accepts objects with `execute` (SubprocessSandbox)
or `run_isolated` (PythonRunner) — methods that guarantee out-of-process
execution. The in-process fallback in Experimenter.run() was removed.
"""
import pytest
from expansion_system.discovery.experimenter import Experimenter
from INFRASTRUCTURE.sandbox.subprocess_sandbox import SubprocessSandbox


class FakeSandboxWithRunOnly:
    """Fake sandbox that has `run` but no real isolation."""
    def run(self, func):
        return func()


class FakeSandboxWithExecuteOnly:
    """Fake sandbox that has `execute` but doesn't actually isolate."""
    def execute(self, func, **kwargs):
        return {"status": "ok", "result": func()}


def test_fake_run_only_sandbox_rejected():
    """A object with only `run` method should not be considered a real sandbox."""
    exp = Experimenter()
    assert exp._is_real_sandbox(FakeSandboxWithRunOnly()) is False


def test_subprocess_sandbox_accepted():
    """SubprocessSandbox with `execute` should be considered real."""
    exp = Experimenter()
    sandbox = SubprocessSandbox(timeout=5.0)
    assert exp._is_real_sandbox(sandbox) is True


def test_none_rejected():
    exp = Experimenter()
    assert exp._is_real_sandbox(None) is False


def test_plain_object_rejected():
    exp = Experimenter()
    assert exp._is_real_sandbox(object()) is False


def test_fake_sandbox_blocks_execution():
    """Even with a fake sandbox accepted, code should not run in-process."""
    exp = Experimenter()
    fake = FakeSandboxWithRunOnly()

    def dangerous():
        return "executed_in_process"

    with pytest.raises(PermissionError):
        exp.run(dangerous, sandbox=fake)


def test_real_sandbox_executes():
    """Real sandbox should execute code."""
    exp = Experimenter()
    sandbox = SubprocessSandbox(timeout=5.0)
    result = exp.run(lambda: "safe_result", sandbox=sandbox)
    assert result["outcome"] == "success"
    assert result["result"] == "safe_result"
    assert result["isolated_execution"] is True
