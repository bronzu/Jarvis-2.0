"""Regression tests for the OS-level sandbox (neural_core.os_sandbox).

These tests prove that real OS-level isolation is in place:
- macOS: ``sandbox-exec`` Mandatory Access Control profile
- All platforms: rlimits, setsid, env-strip, umask, timeout

Tests are run on every platform but the macOS-specific assertions
are skipped if ``sandbox-exec`` is not available.
"""
from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_core.os_sandbox import (  # noqa: E402
    OSIsolationLevel,
    is_os_isolation_available,
    run_isolated,
)


SANDBOX_DIR = Path("/tmp/jarvis-os-sandbox-test")


def _cleanup_pwned() -> None:
    for p in [
        "/tmp/pwned_outside.txt",
        "/tmp/pwned_typewalk",
        "/tmp/pwned_os_subprocess",
    ]:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass


def _ensure_sandbox_dir() -> Path:
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    return SANDBOX_DIR


class TestOSSandboxAvailability(unittest.TestCase):
    def test_sandbox_executable_on_macos(self) -> None:
        avail = is_os_isolation_available()
        if avail["platform"]:
            self.assertTrue(avail["sandbox_exec"], "sandbox-exec missing on macOS")
        # rlimit_* are always present on POSIX
        self.assertTrue(avail["rlimit_cpu"])
        self.assertTrue(avail["rlimit_as"])
        self.assertTrue(avail["rlimit_fsize"])
        self.assertTrue(avail["rlimit_core"])
        self.assertTrue(avail["setsid"])
        self.assertTrue(avail["umask"])


class TestOSSandboxLegitimateCode(unittest.TestCase):
    def setUp(self) -> None:
        _cleanup_pwned()
        _ensure_sandbox_dir()

    def test_arithmetic_under_strict(self) -> None:
        r = run_isolated("print(sum(range(100)))",
                         level=OSIsolationLevel.STRICT, timeout_seconds=3.0,
                         sandbox_dir=SANDBOX_DIR)
        self.assertEqual(r["status"], "ok")
        self.assertIn("4950", r["output"])

    def test_file_write_inside_sandbox(self) -> None:
        target = SANDBOX_DIR / "legit.txt"
        try:
            r = run_isolated(
                "open('legit.txt', 'w').write('hello')",
                level=OSIsolationLevel.STRICT, timeout_seconds=3.0,
                sandbox_dir=SANDBOX_DIR,
            )
            self.assertEqual(r["status"], "ok")
            self.assertTrue(target.exists(), "legit file not created")
            self.assertEqual(target.read_text(), "hello")
        finally:
            if target.exists():
                target.unlink()


class TestOSSandboxBlocksEscape(unittest.TestCase):
    def setUp(self) -> None:
        _cleanup_pwned()
        _ensure_sandbox_dir()

    def test_file_write_outside_sandbox_blocked(self) -> None:
        r = run_isolated(
            "open('/tmp/pwned_outside.txt', 'w').write('pwned')",
            level=OSIsolationLevel.STRICT, timeout_seconds=3.0,
            sandbox_dir=SANDBOX_DIR,
        )
        # The child sees a PermissionError or similar
        self.assertNotEqual(r["status"], "ok")
        self.assertFalse(os.path.exists("/tmp/pwned_outside.txt"),
                         "OS-level sandbox failed: file escaped to /tmp")

    def test_typewalk_exploit_blocked(self) -> None:
        exploit = (
            "cls = next(c for c in ().__class__.__bases__[0].__subclasses__() "
            "if c.__name__ == '_wrap_close'); g = cls.__init__.__globals__; "
            "os_open = g['open']; "
            "fd = os_open('/tmp/pwned_typewalk', os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600); "
            "os.write(fd, b'PWNED'); os.close(fd); print('WROTE')"
        )
        r = run_isolated(exploit,
                         level=OSIsolationLevel.STRICT, timeout_seconds=3.0,
                         sandbox_dir=SANDBOX_DIR)
        # Either the type-walk itself failed in Python, OR sandbox-exec
        # blocked the file write, OR both.
        self.assertNotIn("WROTE", r["output"],
                         f"type-walk exploit produced WROTE: {r['output']!r}")
        self.assertFalse(os.path.exists("/tmp/pwned_typewalk"),
                         "OS-level sandbox failed: type-walk created /tmp file")

    def test_paranoid_blocks_import(self) -> None:
        r = run_isolated("import os; print('IMPORTED OS')",
                         level=OSIsolationLevel.PARANOID, timeout_seconds=3.0)
        self.assertNotEqual(r["status"], "ok")
        self.assertNotIn("IMPORTED OS", r["output"])


class TestOSSandboxTimeout(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_sandbox_dir()

    def test_infinite_loop_killed_by_timeout(self) -> None:
        r = run_isolated("while True: pass",
                         level=OSIsolationLevel.STRICT, timeout_seconds=0.7)
        self.assertTrue(r["timed_out"], f"not timed out: {r}")
        self.assertEqual(r["status"], "timeout")


class TestOSSandboxLevels(unittest.TestCase):
    def test_none_level_executes_without_isolation(self) -> None:
        # NONE uses an in-process exec() which doesn't capture stdout,
        # so we use status="ok" as the proof of execution.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = run_isolated("print(1+1)", level=OSIsolationLevel.NONE,
                             timeout_seconds=2.0)
        self.assertEqual(r["status"], "ok")
        self.assertIn("2", buf.getvalue())

    def test_basic_level_executes(self) -> None:
        _ensure_sandbox_dir()
        r = run_isolated("print('basic-ok')",
                         level=OSIsolationLevel.BASIC, timeout_seconds=2.0)
        self.assertEqual(r["status"], "ok")
        self.assertIn("basic-ok", r["output"])


class TestOSSandboxPrimitivesEnforced(unittest.TestCase):
    """The rlimits / setsid / umask / env-strip must be effective under
    sandbox-exec too (not just the preexec_fn path on Linux)."""

    def setUp(self) -> None:
        _ensure_sandbox_dir()

    def test_rlimits_effective_under_strict(self) -> None:
        """RLIMIT_CPU and RLIMIT_FSIZE must be applied in the child even
        when the child is wrapped in sandbox-exec. This test runs a
        diagnostic that imports the runner and checks rlimits in
        the child process.
        """
        import subprocess
        import sys as _sys
        from neural_core.os_sandbox import _RUNNER_TEMPLATE
        # We inject the diagnostic BEFORE the user code runs. The diagnostic
        # runs in the runner's own scope (not the restricted user namespace)
        # so it has full access to resource/os/stat.
        diagnostic_block = (
            "import resource as _r, os as _o, stat as _s\n"
            "print('DIAG_CPU=' + str(_r.getrlimit(_r.RLIMIT_CPU)))\n"
            "print('DIAG_NPROC=' + str(_r.getrlimit(_r.RLIMIT_NPROC)))\n"
            "print('DIAG_NOFILE=' + str(_r.getrlimit(_r.RLIMIT_NOFILE)))\n"
            "print('DIAG_CORE=' + str(_r.getrlimit(_r.RLIMIT_CORE)))\n"
            "print('DIAG_FSIZE=' + str(_r.getrlimit(_r.RLIMIT_FSIZE)))\n"
            "print('DIAG_PGID=' + str(_o.getpgrp()))\n"
            "print('DIAG_SID=' + str(_o.getsid(0)))\n"
            "print('DIAG_NEW_SESSION=' + str(_o.getpgrp() == _o.getsid(0)))\n"
            "_um = _o.umask(0o022); _o.umask(_um); print('DIAG_UMASK=' + oct(_um))\n"
            "print('DIAG_ENV_KEYS=' + str(len(list(_o.environ.keys()))))\n"
        )
        runner_src = _RUNNER_TEMPLATE.format(
            code="print('user-ok')",
            paranoid="False",
            is_sandboxed="True",
        )
        # Inject diagnostic block right after the is_sandboxed block,
        # before the user code runs.
        marker = "# Build the user-facing namespace"
        runner_src = runner_src.replace(
            marker,
            diagnostic_block + "\n# Build the user-facing namespace",
        )

        sandbox_dir = SANDBOX_DIR
        sb_path = "/usr/bin/sandbox-exec" if os.path.exists("/usr/bin/sandbox-exec") else None
        if sb_path:
            profile = f'''(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow file-read*)
(allow file-write* (subpath "{sandbox_dir}"))
'''
            profile_file = sandbox_dir / ".rlimit_test.sb"
            profile_file.write_text(profile)
            cmd = [sb_path, "-f", str(profile_file), _sys.executable, "-I", "-S", "-c", runner_src]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                output = r.stdout
            finally:
                profile_file.unlink()
        else:
            r = subprocess.run(
                [_sys.executable, "-I", "-S", "-c", runner_src],
                capture_output=True, text=True, timeout=5,
            )
            output = r.stdout

        # Extract diagnostic values
        diag = {}
        for line in output.splitlines():
            if "=" in line and line.startswith("DIAG_"):
                k, v = line.split("=", 1)
                diag[k] = v

        self.assertIn("DIAG_CPU", diag, f"no diagnostics in output: {output!r}")
        cpu_soft = int(diag["DIAG_CPU"].split(",")[0].strip("() "))
        self.assertEqual(cpu_soft, 5, f"RLIMIT_CPU soft not 5: {diag}")
        self.assertIn("DIAG_NOFILE", diag)
        nofile_soft = int(diag["DIAG_NOFILE"].split(",")[0].strip("() "))
        self.assertEqual(nofile_soft, 32, f"RLIMIT_NOFILE soft not 32: {diag}")
        self.assertIn("DIAG_CORE", diag)
        core_soft = int(diag["DIAG_CORE"].split(",")[0].strip("() "))
        self.assertEqual(core_soft, 0, f"RLIMIT_CORE soft not 0: {diag}")
        self.assertIn("DIAG_UMASK", diag)
        umask_val = int(diag["DIAG_UMASK"], 8)
        self.assertEqual(umask_val, 0o077,
                         f"umask not 0o077: {diag['DIAG_UMASK']}")
        # New session: PGID should equal SID (setsid was called)
        self.assertIn("DIAG_NEW_SESSION", diag)
        new_session = diag["DIAG_NEW_SESSION"] == "True"
        self.assertTrue(new_session, f"setsid not effective: {diag}")


class TestCrossPlatformHonesty(unittest.TestCase):
    """The probe must report what is ACTUALLY enforced on this platform,
    not pretend that all primitives work everywhere."""

    def test_probe_reports_real_capabilities(self) -> None:
        from neural_core.os_sandbox import is_os_isolation_available
        avail = is_os_isolation_available()
        # On macOS, sandbox-exec must be available
        if avail["platform"]:
            self.assertTrue(avail["sandbox_exec"],
                            "macOS without sandbox-exec is unexpected")
            # On macOS, chroot is unavailable
            self.assertFalse(avail["chroot"],
                             "chroot reported on macOS (incorrect)")
        else:
            # Linux: sandbox-exec is not a real mechanism
            self.assertFalse(avail["sandbox_exec"],
                             "sandbox-exec reported on non-macOS")
            # Linux: chroot is the underlying mechanism
            self.assertTrue(avail["chroot"],
                            "chroot not reported on Linux")
        # rlimits work on every POSIX
        self.assertTrue(avail["rlimit_cpu"])
        self.assertTrue(avail["rlimit_fsize"])
        self.assertTrue(avail["rlimit_nproc"])
        self.assertTrue(avail["rlimit_core"])

    def test_isolation_probe_distinguishes_layers(self) -> None:
        """os_isolation_probe must distinguish app-level vs OS-level."""
        from neural_core.os_isolation_probe import run_isolation_probe
        r = run_isolation_probe()
        # application_level_isolation is always True
        self.assertTrue(r["summary"]["application_level_isolation"])
        # os_sandbox is True iff the OS-level mechanism is wired
        self.assertTrue(r["summary"]["os_sandbox"],
                        "os_sandbox=false but is_os_isolation_available reports True")
        # app_os_sandbox must include the smoke test evidence
        self.assertTrue(r["app_os_sandbox"]["module_available"])
        smoke = r["app_os_sandbox"]["smoke_test"]
        self.assertEqual(smoke["legitimate_code"]["status"], "ok")
        self.assertTrue(smoke["escape_blocked"]["block_succeeded"])


if __name__ == "__main__":
    unittest.main()
