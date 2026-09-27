"""Tests for Finding 7: AutonomyManager arbitrary code execution.

The AutonomyManager.act() method accepted any callable with approved=True
and executed it without source-code inspection. An approved action could call
os.system, open files, pickle.load, etc.

Fix: Added `_validate_action_source()` that inspects the callable's source via
`inspect.getsource()` and rejects actions containing dangerous patterns
(os.system, subprocess, __import__, eval, exec, open, pickle, etc.).
"""
import os
import pytest

from executive_system.autonomy_manager import AutonomyManager
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


def test_autonomy_safe_action_allowed():
    guard = PermissionGuard()
    result = AutonomyManager(guard).act(lambda: "done", approved=True)
    assert result == "done"


def test_autonomy_requires_approval():
    guard = PermissionGuard()
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(lambda: "done")


def test_autonomy_blocks_os_system():
    guard = PermissionGuard()
    def dangerous():
        os.system("echo pwned > /tmp/f7_blocked.txt")
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)
    assert not os.path.exists("/tmp/f7_blocked.txt")


def test_autonomy_blocks_subprocess():
    guard = PermissionGuard()
    def dangerous():
        import subprocess
        subprocess.run(["echo", "pwned"])
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)


def test_autonomy_blocks_open():
    guard = PermissionGuard()
    def dangerous():
        open("/tmp/f7_open_blocked.txt", "w")
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)


def test_autonomy_blocks_pickle():
    guard = PermissionGuard()
    def dangerous():
        import pickle
        pickle.loads(b"")
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)


def test_autonomy_blocks_eval():
    guard = PermissionGuard()
    def dangerous():
        eval("__import__('os')")
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)


def test_autonomy_blocks_getattr():
    guard = PermissionGuard()
    def dangerous():
        getattr(os, "system")
    with pytest.raises(PermissionError):
        AutonomyManager(guard).act(dangerous, approved=True)


def test_autonomy_blocks_non_callable():
    guard = PermissionGuard()
    with pytest.raises((TypeError, PermissionError, ValueError)):
        AutonomyManager(guard).act("not_callable", approved=True)
