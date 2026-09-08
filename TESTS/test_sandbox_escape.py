"""
Security regression tests for the PythonRunner sandbox.

These tests verify that the documented escape vectors used against the
previous `{"__builtins__": {abs, sum, len}}`-only sandbox no longer work
after the AST + literal-pattern defense layers were added.
"""

import os
import sys
import pytest

from executive_system.tools.python_runner import PythonRunner


def _exploit_files_present():
    return [p for p in (
        "/tmp/sandbox_escape_proof",
        "/tmp/pwned_chain",
        "/tmp/full_chain_escape",
        "/tmp/no_redir_escape",
    ) if os.path.exists(p)]


@pytest.fixture
def cleanup_exploit_files():
    for f in _exploit_files_present():
        os.unlink(f)
    yield
    for f in _exploit_files_present():
        if os.path.exists(f):
            os.unlink(f)


def test_runner_blocks_type_walk_escape(cleanup_exploit_files):
    """The classic ().__class__.__bases__[0].__subclasses__() escape."""
    runner = PythonRunner()
    code = """
for cls in ().__class__.__bases__[0].__subclasses__():
    if cls.__name__ == '_wrap_close':
        cls.__init__.__globals__.get('system')('id')
        break
"""
    with pytest.raises(PermissionError):
        runner.run(code, approved=True)
    assert not _exploit_files_present()


def test_runner_blocks_os_system_literal(cleanup_exploit_files):
    runner = PythonRunner()
    with pytest.raises(PermissionError):
        runner.run("import os; os.system('id')", approved=True)


def test_runner_blocks_open_with_w(cleanup_exploit_files):
    runner = PythonRunner()
    with pytest.raises(PermissionError):
        runner.run("open('/tmp/foo', 'w').write('x')", approved=True)


def test_runner_blocks_eval_exec_compile(cleanup_exploit_files):
    runner = PythonRunner()
    for code in ("eval('1+1')", "exec('print(1)')", "compile('1', '<x>', 'eval')"):
        with pytest.raises(PermissionError):
            runner.run(code, approved=True)


def test_runner_blocks_getattr_builtin_access(cleanup_exploit_files):
    runner = PythonRunner()
    with pytest.raises(PermissionError):
        runner.run("getattr(__builtins__, '__import__')('os')", approved=True)


def test_runner_blocks_attribute_introspection(cleanup_exploit_files):
    """Accessing __class__/__bases__/__subclasses__/__globals__ is denied."""
    runner = PythonRunner()
    for code in (
        "().__class__",
        "().__bases__",
        "().__subclasses__()",
        "().__class__.__mro__",
    ):
        with pytest.raises(PermissionError):
            runner.run(code, approved=True)


def test_runner_allows_legitimate_arithmetic():
    runner = PythonRunner()
    result = runner.run("x = 1+1", approved=True)
    assert result.get("x") == 2


def test_runner_allows_abs_sum_len():
    runner = PythonRunner()
    # Assign to a variable so the value appears in the returned scope dict
    assert runner.run("a = abs(-5)", approved=True).get("a") == 5
    assert runner.run("s = sum([1,2,3])", approved=True).get("s") == 6
    assert runner.run("l = len([1,2,3])", approved=True).get("l") == 3


def test_runner_blocks_when_not_approved():
    runner = PythonRunner()
    with pytest.raises(PermissionError):
        runner.run("x = 1+1", approved=False)
