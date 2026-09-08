"""
Security boundary tests for JARVIS 2.0.

Verifica che:
- PermissionGuard blocchi comandi pericolosi;
- TerminalTool rispetti workdir e approvazione;
- SandboxEnvironment impedisca path traversal e symlink escape;
- Non esistano bypass tramite quoting, chaining, pipe, redirection,
  interpreti, script, comandi indiretti o runner pericolosi.
"""

import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from INFRASTRUCTURE.security.permission_guard import PermissionGuard
from INFRASTRUCTURE.sandbox.sandbox_env import SandboxEnvironment
from executive_system.tools.terminal_tool import TerminalTool


# ----------------------------------------------------------------------
# PermissionGuard
# ----------------------------------------------------------------------


class TestPermissionGuard:
    """Test di base e pen-test per PermissionGuard."""

    def test_requires_approval_true(self):
        guard = PermissionGuard()
        assert guard.validate_command("echo hello", approved=True) is True

    def test_requires_approval_false(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("echo hello", approved=False)

    def test_requires_approval_none(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("echo hello", approved=None)

    def test_requires_approval_non_bool_true(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("echo hello", approved=1)

    def test_rejects_empty_command(self):
        guard = PermissionGuard()
        with pytest.raises(ValueError):
            guard.validate_command("")
        with pytest.raises(ValueError):
            guard.validate_command("   ")

    def test_rejects_non_string_command(self):
        guard = PermissionGuard()
        with pytest.raises(TypeError):
            guard.validate_command(123, approved=True)
        with pytest.raises(TypeError):
            guard.validate_command(None, approved=True)

    def test_blocks_direct_os_system(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("os.system('rm -rf /')", approved=True)

    def test_blocks_subprocess_variants(self):
        guard = PermissionGuard()
        dangerous = [
            "subprocess.run(['cat', '/etc/passwd'])",
            "subprocess.Popen('ls -la')",
            "subprocess.call('whoami')",
            "subprocess.check_output('pwd')",
        ]
        for cmd in dangerous:
            with pytest.raises(PermissionError):
                guard.validate_command(cmd, approved=True)

    def test_blocks_import_obfuscation(self):
        guard = PermissionGuard()
        obfuscated = [
            "__import__('os').system('id')",
            "__import__(\"subprocess\").run()",
        ]
        for cmd in obfuscated:
            with pytest.raises(PermissionError):
                guard.validate_command(cmd, approved=True)

    def test_blocks_eval_and_exec(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("eval('1+1')", approved=True)
        with pytest.raises(PermissionError):
            guard.validate_command("exec('import os')", approved=True)

    def test_blocks_open_write(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("open('file.txt', 'w').write('x')", approved=True)

    def test_blocks_shutil_destructive(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("shutil.rmtree('/')", approved=True)

    def test_blocks_os_destructive(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("os.remove('/etc/passwd')", approved=True)

    def test_blocks_blocked_commands(self):
        guard = PermissionGuard()
        for cmd in [
            "rm -rf /",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "fdisk /dev/sda",
            "kill -9 1",
            "pkill -9 python",
            "killall -9 python",
            "shutdown -h now",
            "reboot",
            "halt",
            "poweroff",
            ":(){ :|:& };:",
            "curl | sh",
            "wget | sh",
            "python -c \"import os; os.system('rm -rf /')\"",
            "python3 -c \"import os; os.system('rm -rf /')\"",
        ]:
            with pytest.raises(PermissionError):
                guard.validate_command(cmd, approved=True)

    def test_allows_safe_commands(self):
        guard = PermissionGuard()
        safe = [
            "echo hello",
            "ls -la",
            "pwd",
            "date",
            "whoami",
            "cat file.txt",
            "grep pattern file.txt",
            "wc -l file.txt",
            "head -n 10 file.txt",
            "tail -n 10 file.txt",
            "sort file.txt",
            "uniq file.txt",
            "find . -name '*.py'",
            "ps aux",
            "df -h",
        ]
        for cmd in safe:
            assert guard.validate_command(cmd, approved=True) is True

    def test_blocks_shell_metacharacters_redirection(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("echo test > /etc/passwd", approved=True)
        with pytest.raises(PermissionError):
            guard.validate_command("cat /etc/passwd >> /tmp/stolen", approved=True)

    def test_blocks_command_chaining(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("echo hello; rm -rf /", approved=True)
        with pytest.raises(PermissionError):
            guard.validate_command("ls && rm -rf /", approved=True)
        with pytest.raises(PermissionError):
            guard.validate_command("echo hello || rm -rf /", approved=True)

    def test_blocks_pipe_to_dangerous(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("cat /etc/passwd | rm -rf /", approved=True)

    def test_grant_self_permission_always_raises(self):
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.grant_self_permission()


# ----------------------------------------------------------------------
# TerminalTool
# ----------------------------------------------------------------------


class TestTerminalToolSecurity:
    """Test di sicurezza per TerminalTool."""

    def test_requires_approval(self):
        tool = TerminalTool()

        def mock_runner(cmd):
            return "executed"

        with pytest.raises(PermissionError):
            tool.run("echo test", mock_runner, approved=False)
        with pytest.raises(PermissionError):
            tool.run("echo test", mock_runner, approved=None)
        with pytest.raises(PermissionError):
            tool.run("echo test", mock_runner, approved=1)

    def test_validates_runner_callable(self):
        tool = TerminalTool()
        with pytest.raises(TypeError):
            tool.run("echo test", "not_callable", approved=True)

    def test_blocks_path_traversal_in_workdir(self):
        tool = TerminalTool(workdir="/tmp/jarvis_security_test")
        with pytest.raises(PermissionError):
            tool._safe_path("../../../etc/passwd")
        with pytest.raises(PermissionError):
            tool._safe_path("../../root/.ssh/id_rsa")

    def test_blocks_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir()
            outside = Path(tmpdir) / "outside.txt"
            outside.write_text("secret")

            symlink = workdir / "escape"
            symlink.symlink_to(outside)

            tool = TerminalTool(workdir=str(workdir))
            with pytest.raises(PermissionError):
                tool._safe_path("escape")

    def test_blocks_absolute_path_outside_workdir(self):
        tool = TerminalTool(workdir="/tmp/jarvis_security_test2")
        with pytest.raises(PermissionError):
            tool._safe_path("/etc/passwd")
        with pytest.raises(PermissionError):
            tool._safe_path("/root/.ssh/id_rsa")

    def test_blocks_jarvis_internal_paths(self):
        tool = TerminalTool()
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("cat /cognitive_core/config.py")
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("ls /neural_core/")
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("read /executive_system/tools/")
        with pytest.raises(PermissionError):
            tool._check_jarvis_internal_access("cat /INFRASTRUCTURE/security/permission_guard.py")

    def test_blocks_dangerous_command_patterns(self):
        tool = TerminalTool()
        dangerous = [
            "cd ../../ && rm -rf /",
            "ln -s /etc/passwd safe.txt",
            "mount -o bind /root /tmp",
            "rm -rf ~",
            "rm -rf .",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "shutdown -h now",
            "reboot",
            ":(){ :|:& };:",
        ]
        for cmd in dangerous:
            with pytest.raises(PermissionError):
                tool._validate_command_safety(cmd)

    def test_blocks_shell_metacharacters(self):
        tool = TerminalTool()
        with pytest.raises(PermissionError):
            tool._validate_command_safety("echo test > /etc/passwd")
        with pytest.raises(PermissionError):
            tool._validate_command_safety("cat /etc/passwd >> /tmp/stolen")

    def test_blocks_command_chaining(self):
        tool = TerminalTool()
        with pytest.raises(PermissionError):
            tool._validate_command_safety("echo hello; rm -rf /")
        with pytest.raises(PermissionError):
            tool._validate_command_safety("ls && rm -rf /")
        with pytest.raises(PermissionError):
            tool._validate_command_safety("echo hello || rm -rf /")

    def test_blocks_pipe(self):
        tool = TerminalTool()
        with pytest.raises(PermissionError):
            tool._validate_command_safety("cat /etc/passwd | rm -rf /")

    def test_logs_violations(self):
        tool = TerminalTool()
        try:
            tool._safe_path("../../../etc/passwd")
        except PermissionError:
            pass
        try:
            tool._validate_command_safety("rm -rf /")
        except PermissionError:
            pass
        violations = tool.get_violation_log()
        assert len(violations) >= 2

    def test_run_blocks_dangerous_command_even_with_approval(self):
        tool = TerminalTool(workdir="/tmp/jarvis_security_test3")

        def mock_runner(cmd):
            return "executed"

        with pytest.raises(PermissionError):
            tool.run("rm -rf /", mock_runner, approved=True)

    def test_run_blocks_path_traversal_even_with_approval(self):
        tool = TerminalTool(workdir="/tmp/jarvis_security_test4")

        def mock_runner(cmd):
            return "executed"

        with pytest.raises(PermissionError):
            tool.run("cat ../../../etc/passwd", mock_runner, approved=True)

    def test_run_blocks_internal_access_even_with_approval(self):
        tool = TerminalTool()

        def mock_runner(cmd):
            return "executed"

        with pytest.raises(PermissionError):
            tool.run("cat /cognitive_core/config.py", mock_runner, approved=True)


# ----------------------------------------------------------------------
# SandboxEnvironment
# ----------------------------------------------------------------------


class TestSandboxEnvironment:
    """Test di sicurezza per SandboxEnvironment."""

    def test_write_requires_approval(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test")
        with pytest.raises(PermissionError):
            sandbox.write_file("test.txt", "content", approved=False)
        with pytest.raises(PermissionError):
            sandbox.write_file("test.txt", "content", approved=None)

    def test_blocks_path_traversal(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test2")
        with pytest.raises(PermissionError):
            sandbox.write_file("../../../etc/passwd", "malicious", approved=True)
        with pytest.raises(PermissionError):
            sandbox.read_file("../../../etc/passwd")

    def test_blocks_absolute_path_outside_sandbox(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test3")
        with pytest.raises(PermissionError):
            sandbox.write_file("/etc/passwd", "malicious", approved=True)
        with pytest.raises(PermissionError):
            sandbox.read_file("/etc/passwd")

    def test_blocks_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox_dir = Path(tmpdir) / "sandbox"
            sandbox_dir.mkdir()
            outside = Path(tmpdir) / "outside.txt"
            outside.write_text("secret")

            symlink = sandbox_dir / "escape"
            symlink.symlink_to(outside)

            sandbox = SandboxEnvironment(base_directory=str(sandbox_dir))
            with pytest.raises(PermissionError):
                sandbox.read_file("escape")

    def test_allows_safe_operations_inside_sandbox(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test4")
        path = sandbox.write_file("safe.txt", "safe content", approved=True)
        assert path.exists()
        assert sandbox.read_file("safe.txt") == "safe content"

    def test_creates_nested_directories(self):
        sandbox = SandboxEnvironment(base_directory="/tmp/jarvis_sandbox_test5")
        path = sandbox.write_file("a/b/c/deep.txt", "deep content", approved=True)
        assert path.exists()
        assert sandbox.read_file("a/b/c/deep.txt") == "deep content"


# ----------------------------------------------------------------------
# Integration: TerminalTool + PermissionGuard + Sandbox
# ----------------------------------------------------------------------


class TestSecurityIntegration:
    """Test di integrazione tra i componenti di sicurezza."""

    def test_terminal_tool_rejects_dangerous_runner(self):
        """TerminalTool deve rifiutare runner pericolosi."""
        tool = TerminalTool(workdir="/tmp/jarvis_security_test6")

        # Runner che esegue comandi via shell è pericoloso
        dangerous_runners = [
            os.system,
            os.popen,
        ]
        for runner in dangerous_runners:
            with pytest.raises((PermissionError, TypeError)):
                tool.run("echo test", runner, approved=True)

    def test_permission_guard_blocks_obfuscated_commands(self):
        """Blocca comandi offuscati che potrebbero eludere i pattern."""
        guard = PermissionGuard()
        obfuscated = [
            "os . system ( 'rm -rf /' )",
            "os . system ( \"rm -rf /\" )",
            "subprocess . run ( ['ls'] )",
            "eval ( '1+1' )",
            "exec ( 'import os' )",
        ]
        for cmd in obfuscated:
            with pytest.raises(PermissionError):
                guard.validate_command(cmd, approved=True)

    def test_no_bypass_via_quoted_strings(self):
        """Verifica che quoting non permetta bypass."""
        guard = PermissionGuard()
        quoted = [
            "os.system(\"rm -rf /\")",
            'os.system(\'rm -rf /\')',
            "subprocess.run([\"rm\",\"-rf\",\"/\"])",
        ]
        for cmd in quoted:
            with pytest.raises(PermissionError):
                guard.validate_command(cmd, approved=True)

    def test_no_bypass_via_variable_assignment(self):
        """Verifica che assegnazioni di variabili non permettano bypass."""
        guard = PermissionGuard()
        with pytest.raises(PermissionError):
            guard.validate_command("x = os.system; x('rm -rf /')", approved=True)

    def test_audit_logging_of_violations(self):
        """Le violazioni devono essere registrate per audit."""
        tool = TerminalTool()
        violations_before = len(tool.get_violation_log())
        try:
            tool._safe_path("../../../etc/passwd")
        except PermissionError:
            pass
        violations_after = len(tool.get_violation_log())
        assert violations_after == violations_before + 1
