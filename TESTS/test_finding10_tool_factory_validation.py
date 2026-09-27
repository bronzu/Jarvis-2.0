"""Tests for Finding 10: ToolFactory.validate_generated substring matching bypass.

The validate_generated method used substring matching (`pattern in code`) to
detect dangerous patterns. This can be bypassed by string concatenation:
  `__imp` + `ort__` doesn't contain `__import__` as a substring.

Fix: Added AST-based validation that detects dangerous imports, function calls,
and __builtins__ access regardless of string obfuscation.
"""
import pytest
from expansion_system.tool_factory.factory import ToolFactory


def test_validate_blocks_dunder_import():
    f = ToolFactory()
    code = "import os\n\ndef run(): os.system('id')"
    r = f.validate_generated(code)
    assert r["valid"] is False


def test_validate_blocks_str_concat_import():
    f = ToolFactory()
    code = """class T:
    def run(self):
        getattr(__builtins__, '__imp'+'ort__')('os').system('id')"""
    r = f.validate_generated(code)
    assert r["valid"] is False
    assert "getattr" in r["error"]


def test_validate_blocks_subscript_builtins():
    f = ToolFactory()
    code = """class T:
    def run(self):
        __builtins__['__imp'+'ort__']('os').system('id')"""
    r = f.validate_generated(code)
    assert r["valid"] is False
    assert "__builtins__" in r["error"]


def test_validate_allows_safe_code():
    f = ToolFactory()
    code = """from typing import Any
class SafeTool:
    def compute(self, text: str):
        return 42"""
    r = f.validate_generated(code)
    assert r["valid"] is True


def test_validate_allows_hashlib_import():
    f = ToolFactory()
    code = """import hashlib
class SafeTool:
    def compute(self, text: str):
        return hashlib.sha256(text.encode()).hexdigest()"""
    r = f.validate_generated(code)
    assert r["valid"] is True


def test_validate_blocks_eval():
    f = ToolFactory()
    code = """class T:
    def run(self):
        eval('1+1')"""
    r = f.validate_generated(code)
    assert r["valid"] is False


def test_validate_blocks_subprocess_import():
    f = ToolFactory()
    code = """import subprocess
class T:
    def run(self):
        return subprocess.run(['ls'])"""
    r = f.validate_generated(code)
    assert r["valid"] is False


def test_validate_blocks_os_import():
    f = ToolFactory()
    code = """import os
class T:
    def run(self):
        return os.system('id')"""
    r = f.validate_generated(code)
    assert r["valid"] is False


def test_validate_rejects_syntax_error():
    f = ToolFactory()
    r = f.validate_generated("def broken(")
    assert r["valid"] is False
    assert "SyntaxError" in r["error"]
