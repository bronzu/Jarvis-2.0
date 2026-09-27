"""Tests for Finding 2: AST gate bypass via str.format().

The payload "{0.__class__.__bases__[0].__subclasses__}".format([]) passes both
the AST check and the literal re-scan because:
1. The AST sees only a string Constant - no Attribute access node for __class__
2. The literal re-scan does not search inside string literal contents

Fix: ban .format() on string constants in the AST gate, and scan string
literal contents for denied attribute names in the literal re-scan.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executive_system.tools.python_runner import (
    _ast_has_denied,
    _literal_has_denied,
    PythonRunner,
)


class TestASTFormatBypass(unittest.TestCase):
    def test_payload_ast_rejected(self):
        """The format-string bypass payload must be rejected by _ast_has_denied."""
        payload = '"{0.__class__.__bases__[0].__subclasses__}".format([])'
        err = _ast_has_denied(payload)
        self.assertNotEqual(err, "", f"AST check should reject format-string bypass, got: {err!r}")

    def test_payload_literal_rejected(self):
        """The format-string bypass payload must be rejected by _literal_has_denied."""
        payload = '"{0.__class__.__bases__[0].__subclasses__}".format([])'
        err = _literal_has_denied(payload)
        self.assertNotEqual(err, "", f"Literal check should reject, got: {err!r}")

    def test_payload_python_runner_rejects(self):
        """PythonRunner.run must raise PermissionError for the bypass payload."""
        payload = '"{0.__class__.__bases__[0].__subclasses__}".format([])'
        runner = PythonRunner()
        with self.assertRaises(PermissionError):
            runner.run(payload, approved=True)

    def test_percent_format_bypass_rejected(self):
        """% -formatting on string literal with denied attrs must be rejected."""
        payload = '"%s" % "__class__"'
        err = _ast_has_denied(payload)
        self.assertNotEqual(err, "")

    def test_literal_scans_for_denied_attrs(self):
        """_literal_has_denied must find denied attribute names in string literals."""
        for attr in ["__class__", "__bases__", "__subclasses__", "__globals__",
                      "__builtins__", "__import__"]:
            code = f's = "{attr}"'
            err = _literal_has_denied(code)
            self.assertNotEqual(err, "", f"Should detect {attr} in string literal")

    def test_legitimate_code_still_passes(self):
        """Legitimate Python code must not be falsely blocked."""
        legit_cases = [
            'result = sum(range(10))',
            'x = 1 + 2',
            'name = "hello"; msg = "world"',
            'items = ["a", "b", "c"]',
            'msg = "hello".upper()',
            'print("hello world")',
            'total = 0\nfor i in range(5):\n    total += i',
        ]
        for code in legit_cases:
            ast_err = _ast_has_denied(code)
            lit_err = _literal_has_denied(code)
            self.assertEqual(ast_err, "", f"AST false-positive for {code!r}: {ast_err}")
            self.assertEqual(lit_err, "", f"Literal false-positive for {code!r}: {lit_err}")

    def test_run_isolated_also_blocks_payload(self):
        """PythonRunner.run_isolated must also reject the bypass payload."""
        payload = '"{0.__class__.__bases__[0].__subclasses__}".format([])'
        runner = PythonRunner()
        with self.assertRaises(PermissionError):
            runner.run_isolated(payload, approved=True)


if __name__ == "__main__":
    unittest.main()
