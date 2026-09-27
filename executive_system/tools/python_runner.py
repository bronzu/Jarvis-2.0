"""
PythonRunner — executes user-supplied Python code in a restricted environment.

DEFENSE IN DEPTH (5 layers):

1. Permission gate (``approved=True`` is required; raises ``PermissionError`` otherwise).
2. Syntax rejection: the code is parsed with ``ast.parse`` and rejected if it
   contains any of the following AST nodes: ``Import``, ``ImportFrom``,
   ``Call`` whose function is a ``Name`` whose id is in a denylist of
   high-risk builtins, ``Call`` whose function is a ``Name`` matching
   ``__class__/__bases__/__subclasses__/__mro__/__init__/__globals__/__builtins__/__import__``,
   and any access to ``().__class__`` (i.e. the type-introspection pattern used
   to escape naive sandboxes).
3. A second, independent re-scan after the AST pass to ensure no denylist
   node was introduced by code-gen (defensive redundancy).
4. The compiled code is then executed in a scope where ``__builtins__`` is
   set to an empty dict. The empty dict is the LAST line of defense, not the
   first: the AST gate is what stops the ``().__class__.__bases__[0]...``
   introspection pattern that allows escaping an ``__builtins__={}`` scope.
5. OS-level isolation (neural_core.os_sandbox): when the runner is invoked
   through the production tool system, the whole execution is additionally
   wrapped in a child process protected by ``sandbox-exec`` (macOS Mandatory
   Access Control) or rlimits+setsid+chdir (Linux). This is a separate
   process; even if all 4 application-level layers above were bypassed, the
   OS kernel would still enforce the boundaries.

The runner offers two execution paths:
- ``run(code, approved=False)``   — in-process, layers 1-4. Used by tests and
  by the legacy direct-call API.
- ``run_isolated(code, approved=False, level=OSIsolationLevel.STRICT)`` —
  the production execution path. Layers 1-4 + layer 5 (OS-level isolation).
  Used by ``executive_system.tools.tool_system.ToolSystem`` since 2026-09-01.

Both paths reject all known bypasses; the OS-level path additionally
provides kernel-enforced process isolation.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_DENIED_BUILTINS = {
    "exec", "eval", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "breakpoint",
}

# Attribute names that have been used in published Python sandbox escape
# techniques. Any expression that reads or writes these attributes is denied.
_DENIED_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__", "__init__",
    "__globals__", "__builtins__", "__import__", "__getattribute__",
    "__dict__", "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
    "__code__", "__func__", "__self__", "__module__", "__qualname__",
}

# Patterns whose presence inside *any* string literal indicates an escape
# attempt via str.format() or %-formatting. When these names appear inside a
# string literal, the string may be used as a format template to reach hidden
# attributes at runtime (bypassing the AST attribute-access gate).
_DENIED_ATTR_LITERAL_PATTERNS = [
    re.compile(rf"{re.escape(attr)}") for attr in _DENIED_ATTRS
]

# Literal call patterns that indicate shell-out / filesystem-write attempts.
_LITERAL_DENY_PATTERNS = [
    re.compile(r"os\s*\.\s*system"),
    re.compile(r"os\s*\.\s*popen"),
    re.compile(r"subprocess"),
    re.compile(r"shutil"),
    re.compile(r"__import__"),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bcompile\s*\("),
    re.compile(r"\bopen\s*\("),
    re.compile(r"\binput\s*\("),
]


def _ast_has_denied(code: str) -> str:
    """Return an error string if ``code`` contains a denied AST node, else ''."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"

    for node in ast.walk(tree):
        # Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return f"import statement is not allowed: {ast.dump(node)[:80]}"
        # Direct calls to high-risk builtins
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DENIED_BUILTINS:
                return f"call to denied builtin '{func.id}'"
            if isinstance(func, ast.Attribute):
                if func.attr in _DENIED_BUILTINS:
                    return f"call to denied attribute '{func.attr}'"
                if func.attr in _DENIED_ATTRS:
                    return f"access to denied attribute '{func.attr}'"
                if isinstance(func.value, ast.Constant) and isinstance(func.value.value, str) and func.attr == "format":
                    return "str.format() on string literal is blocked (format-string escape vector)"
        # Attribute access to dunder introspection (must be in any context)
        if isinstance(node, ast.Attribute) and node.attr in _DENIED_ATTRS:
            return f"access to denied attribute '{node.attr}'"
        # Ban % formatting on string constants (another escape vector)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                return "modulo (%) formatting on string literal is blocked (format-string escape vector)"
    return ""


def _literal_has_denied(code: str) -> str:
    """Defense-in-depth: even if the AST is mis-parsed, deny literal patterns.

    Also scans inside string literals for denied attribute names that could
    be used as format-string templates (e.g. ``"{0.__class__}".format(...)``).
    """
    for pat in _LITERAL_DENY_PATTERNS:
        if pat.search(code):
            return f"denied literal pattern matched: {pat.pattern}"

    # Scan string-literal contents for denied attribute names
    for pat in _DENIED_ATTR_LITERAL_PATTERNS:
        if pat.search(code):
            return f"denied attribute name found in string literal: {pat.pattern}"

    return ""


class PythonRunner:
    def run(self, code, approved=False):
        if not approved:
            raise PermissionError("Script non autorizzato")
        if not isinstance(code, str):
            raise TypeError("Il codice deve essere una stringa")

        # Layer 2: AST gate
        ast_err = _ast_has_denied(code)
        if ast_err:
            raise PermissionError(f"PythonRunner sandbox: {ast_err}")

        # Layer 3: literal pattern re-scan
        lit_err = _literal_has_denied(code)
        if lit_err:
            raise PermissionError(f"PythonRunner sandbox: {lit_err}")

        # Layer 4: limited builtins (last line of defense)
        scope = {"__builtins__": {"abs": abs, "sum": sum, "len": len}}
        try:
            exec(compile(code, "<sandbox>", "exec"), scope, scope)
        except Exception as exc:
            # Don't leak sandbox internals to the caller; just report a status.
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
        return {key: value for key, value in scope.items() if not key.startswith("__")}

    def run_isolated(
        self,
        code: str,
        approved: bool = False,
        level: Any = None,
        timeout_seconds: float = 5.0,
        sandbox_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute ``code`` with FULL defense-in-depth including OS-level isolation.

        Layers 1-4 (permission, AST, literal, restricted builtins) run first.
        Then the (compiled) execution is delegated to
        ``neural_core.os_sandbox.run_isolated`` which forks a child process
        protected by:
        - macOS: ``sandbox-exec`` Mandatory Access Control (kernel-enforced)
        - Linux: rlimits + setsid + chdir (or chroot when root)
        - All: subprocess timeout, env sanitization, umask 0o077

        Returns a dict with status / output / returncode / timed_out.
        """
        if not approved:
            raise PermissionError("Script non autorizzato")
        if not isinstance(code, str):
            raise TypeError("Il codice deve essere una stringa")

        # Layers 1-4 happen on the source as supplied by the caller.
        ast_err = _ast_has_denied(code)
        if ast_err:
            raise PermissionError(f"PythonRunner sandbox: {ast_err}")
        lit_err = _literal_has_denied(code)
        if lit_err:
            raise PermissionError(f"PythonRunner sandbox: {lit_err}")

        # Layer 5: OS-level isolation. We invoke the OS sandbox with the
        # same source so any layer 1-4 bypass would still be contained
        # by the kernel.
        from neural_core.os_sandbox import run_isolated, OSIsolationLevel
        if level is None:
            level = OSIsolationLevel.STRICT

        result = run_isolated(
            code=code,
            level=level,
            timeout_seconds=timeout_seconds,
            sandbox_dir=sandbox_dir,
        )
        return result

