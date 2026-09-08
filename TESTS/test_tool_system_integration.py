"""End-to-end integration tests: ToolSystem -> PythonRunner -> OS sandbox.

These tests prove that the OS-level sandbox is actually wired into the
production execution path, not just a standalone helper.

For every attack vector the test asserts:
1. ``ToolSystem.execute('python', code=..., approved=True)`` does not
   create the attacker's file in /tmp
2. The result either returns a status indicating rejection, OR raises
   PermissionError, OR (for legitimate code) returns a sandbox result
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from executive_system.tools.tool_system import ToolSystem  # noqa: E402


_PWNED = [
    "/tmp/_e2e_attack_file",
    "/tmp/_e2e_id_out",
    "/tmp/_e2e_typewalk",
    "/tmp/_e2e_subprocess",
    "/tmp/_e2e_popen",
]


def _cleanup() -> None:
    for p in _PWNED:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


class TestToolSystemPythonIntegration(unittest.TestCase):
    def setUp(self) -> None:
        _cleanup()
        self.ts = ToolSystem()

    def tearDown(self) -> None:
        _cleanup()

    # ---- Legitimate code (must succeed) ----

    def test_legitimate_code_runs_via_isolated_path(self) -> None:
        r = self.ts.execute("python", code="result = sum(range(10))", approved=True)
        # Now production path uses run_isolated which returns a status dict
        self.assertEqual(r["status"], "ok")
        self.assertIn("__SANDBOX_OK__", r["output"])

    # ---- Attack vectors (must be blocked) ----

    def test_typewalk_blocked(self) -> None:
        exploit = (
            "for cls in ().__class__.__bases__[0].__subclasses__():\n"
            "    if cls.__name__ == '_wrap_close':\n"
            "        cls.__init__.__globals__.get('system')"
            "('id > /tmp/_e2e_typewalk 2>&1')\n"
            "        break\n"
        )
        with self.assertRaises(PermissionError):
            self.ts.execute("python", code=exploit, approved=True)
        self.assertFalse(os.path.exists("/tmp/_e2e_typewalk"),
                         "type-walk exploit escaped OS+app sandbox")

    def test_os_system_blocked(self) -> None:
        with self.assertRaises(PermissionError):
            self.ts.execute("python",
                            code="os.system('id > /tmp/_e2e_id_out')",
                            approved=True)
        self.assertFalse(os.path.exists("/tmp/_e2e_id_out"),
                         "os.system escaped OS+app sandbox")

    def test_open_blocked(self) -> None:
        with self.assertRaises(PermissionError):
            self.ts.execute("python",
                            code="open('/tmp/_e2e_attack_file', 'w').write('pwned')",
                            approved=True)
        self.assertFalse(os.path.exists("/tmp/_e2e_attack_file"),
                         "open() escaped OS+app sandbox")

    def test_subprocess_blocked(self) -> None:
        with self.assertRaises(PermissionError):
            self.ts.execute("python",
                            code="import subprocess; subprocess.run(['id'])",
                            approved=True)
        self.assertFalse(os.path.exists("/tmp/_e2e_subprocess"),
                         "subprocess escaped OS+app sandbox")

    def test_popen_blocked(self) -> None:
        with self.assertRaises(PermissionError):
            self.ts.execute("python",
                            code="import os; os.popen('id > /tmp/_e2e_popen')",
                            approved=True)
        self.assertFalse(os.path.exists("/tmp/_e2e_popen"),
                         "popen escaped OS+app sandbox")

    def test_approved_false_blocked(self) -> None:
        with self.assertRaises(PermissionError):
            self.ts.execute("python", code="print(1)", approved=False)


class TestRunIsolatedContract(unittest.TestCase):
    """The new PythonRunner.run_isolated must apply all 5 layers."""

    def setUp(self) -> None:
        _cleanup()
        from executive_system.tools.python_runner import PythonRunner
        self.runner = PythonRunner()

    def tearDown(self) -> None:
        _cleanup()

    def test_run_isolated_blocks_typewalk_via_ast(self) -> None:
        with self.assertRaises(PermissionError):
            self.runner.run_isolated(
                "().__class__.__bases__[0].__subclasses__()",
                approved=True,
            )

    def test_run_isolated_blocks_os_system_via_literal(self) -> None:
        with self.assertRaises(PermissionError):
            self.runner.run_isolated(
                "os.system('id')",
                approved=True,
            )

    def test_run_isolated_blocks_open_via_builtin(self) -> None:
        with self.assertRaises(PermissionError):
            self.runner.run_isolated(
                "open('/tmp/x', 'w').write('pwned')",
                approved=True,
            )

    def test_run_isolated_blocks_unapproved(self) -> None:
        with self.assertRaises(PermissionError):
            self.runner.run_isolated("print(1)", approved=False)

    def test_run_isolated_legitimate_code_succeeds(self) -> None:
        r = self.runner.run_isolated(
            "print(sum(range(10)))",
            approved=True,
        )
        self.assertEqual(r["status"], "ok")
        self.assertIn("45", r["output"])


class TestProductionPathNoBypass(unittest.TestCase):
    """ToolSystem must NOT silently fall back to in-process execution."""

    def test_runner_without_run_isolated_is_rejected(self) -> None:
        """A Python runner without run_isolated must NOT be silently used.

        This guards against a future regression where someone replaces the
        default PythonRunner with a stub that lacks the OS-level sandbox
        method. The production path must refuse to execute, not fall back
        to the in-process run() (which has only 4 application-level gates
        and no OS-level isolation).
        """
        class FakeRunner:
            """Mimics a PythonRunner without run_isolated."""

            def run(self, code, approved=False):
                return {"insecure": True, "code": code}

        ts = ToolSystem()
        ts.registry.register_tool("python", lambda: FakeRunner())
        with self.assertRaises(RuntimeError) as cm:
            ts.execute("python", code="print(1)", approved=True)
        self.assertIn("run_isolated", str(cm.exception))
        self.assertIn("OS-level sandboxing", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
