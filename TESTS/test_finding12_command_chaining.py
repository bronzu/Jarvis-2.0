"""Tests for Finding 12: PermissionGuard allows command chaining operators.

The DANGEROUS_PATTERNS list only blocked pipes (`|`) but not other shell
command chaining operators:
- `;` (sequence) — e.g. `ls;id` runs `id` after `ls`
- `&&` (logical AND) — e.g. `ls && id` runs `id` if `ls` succeeds
- `||` (logical OR) — e.g. `ls || id` runs `id` if `ls` fails
- backticks — e.g. `` `id` `` executes `id` in subshell
- `$(...)` — e.g. `$(id)` executes `id` in subshell

This allowed attackers to chain arbitrary commands after an allowed command,
bypassing the allowlist (e.g. `echo hello; node -e 'evil'` or `ls; rm -rf /`).

Fix: Added `;`, `&&`, `||`, backticks, and `$(...)` to DANGEROUS_PATTERNS.
"""
import pytest
from INFRASTRUCTURE.security.permission_guard import PermissionGuard


def test_blocks_semicolon_chaining():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls;id", approved=True)


def test_blocks_semicolon_chaining_with_space():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls ; id", approved=True)


def test_blocks_and_chaining():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls && id", approved=True)


def test_blocks_or_chaining():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls || id", approved=True)


def test_blocks_backtick_substitution():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("echo `id`", approved=True)


def test_blocks_paren_substitution():
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("echo $(id)", approved=True)


def test_blocks_pipe_already_blocked():
    """Pipe was already blocked; verify it still is."""
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls | id", approved=True)


def test_allows_simple_command():
    g = PermissionGuard()
    result = g.validate_command("ls -la", approved=True)
    assert result is True


def test_allows_echo():
    g = PermissionGuard()
    result = g.validate_command("echo hello", approved=True)
    assert result is True


def test_allows_command_with_args():
    g = PermissionGuard()
    result = g.validate_command("grep pattern file.txt", approved=True)
    assert result is True


def test_blocks_destructive_chain():
    """ls; rm -rf / should be blocked even though ls is allowed."""
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("ls; rm -rf /", approved=True)


def test_blocks_reversed_destructive_chain():
    """rm -rf /; ls should be blocked by blocked commands check."""
    g = PermissionGuard()
    with pytest.raises(PermissionError):
        g.validate_command("rm -rf /", approved=True)
