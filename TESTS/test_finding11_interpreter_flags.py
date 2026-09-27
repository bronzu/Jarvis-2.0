"""Tests for Finding 11: PermissionGuard blocks interpreter inline-code flags.

The BLOCKED_INTERPRETER_FLAGS set only included `-c` and `--command`, but many
interpreters have their own inline-code flags:
- node -e (JavaScript inline)
- perl -e (Perl inline)
- ruby -e (Ruby inline)
- php -r (PHP inline)

This allowed attackers to inject arbitrary code via these interpreters.
Fixed by adding -e, --execute, -r, --run to BLOCKED_INTERPRETER_FLAGS.
"""
import pytest
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


def test_blocks_python_c():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("python -c 'print(1)'", approved=True)


def test_blocks_python3_c():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("python3 -c 'print(1)'", approved=True)


def test_blocks_node_e():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('node -e "console.log(1)"', approved=True)


def test_blocks_perl_e():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('perl -e "print 1"', approved=True)


def test_blocks_ruby_e():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('ruby -e "puts 1"', approved=True)


def test_blocks_php_r():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('php -r "echo 1"', approved=True)


def test_blocks_python_long_command():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("python --command 'print(1)'", approved=True)


def test_allows_normal_command():
    g = PermissionGuard()
    result = g.validate_command("ls -la", approved=True)
    assert result is True


def test_allows_python_script_without_inline_flag():
    """python script.py (without -c/-e) is allowed."""
    g = PermissionGuard()
    result = g.validate_command("python script.py", approved=True)
    assert result is True


def test_blocks_bash_c():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('bash -c "id"', approved=True)


def test_blocks_sh_c():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("sh -c id", approved=True)


def test_blocks_zsh_c():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command('zsh -c "id"', approved=True)


def test_blocks_awk_in_terminal_tool():
    """awk is blocked by TerminalTool._validate_command_safety."""
    from executive_system.tools.terminal_tool import TerminalTool
    t = TerminalTool()
    with pytest.raises(PermissionError):
        t._validate_command_safety("awk 'BEGIN{print 1}'")


def test_blocks_sed_in_terminal_tool():
    """sed is blocked by TerminalTool._validate_command_safety."""
    from executive_system.tools.terminal_tool import TerminalTool
    t = TerminalTool()
    with pytest.raises(PermissionError):
        t._validate_command_safety("sed 's/x/y/'")
