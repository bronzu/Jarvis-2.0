"""
AST-based test quality auditor.

Identifies candidate test patterns that may be weak:
  - bare ``assert x``
  - ``assert x is not None``
  - ``assert x >= 0``
  - ``assert result`` (truthiness-only)
  - existence/truthiness asserts
  - tests with only ``mock``/patching and no behavioral assertion
  - tests that don't verify failure behavior
  - tests with no assertions

Produces a structured report with file/line references. Does NOT auto-classify
candidates as wrong; the user must review.
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List


WEAK_ASSERT_CATEGORIES = {
    "bare_truthiness": lambda node: (
        isinstance(node.test, ast.Name)
        and isinstance(node.test.ctx, ast.Load)
    ),
    "compare_to_none": lambda node: (
        isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], (ast.Is, ast.IsNot))
        and isinstance(node.test.left, ast.Name)
        and (
            (isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value is None)
            or (isinstance(node.test.comparators[0], ast.NameConstant) and node.test.comparators[0].value is None)
        )
    ),
    "compare_zero": lambda node: (
        isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.GtE)
        and (
            (isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == 0)
            or (isinstance(node.test.comparators[0], ast.NameConstant) and node.test.comparators[0].value == 0)
        )
    ),
    "single_string_assert": lambda node: (
        isinstance(node.test, ast.Constant)
        and isinstance(node.test.value, str)
    ),
}


def _is_test_function(node: ast.AST) -> bool:
    if not isinstance(node, ast.FunctionDef):
        return False
    if not node.name.startswith("test_"):
        return False
    return True


def _function_has_assert(node: ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Call):
            func = sub.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name and name.startswith("assert") and name[len("assert"):] in (
                "Equal", "NotEqual", "True", "False", "Is", "IsNot",
                "IsNone", "IsNotNone", "In", "NotIn", "IsInstance",
                "NotIsInstance", "Greater", "GreaterEqual", "Less", "LessEqual",
                "AlmostEqual", "NotAlmostEqual", "Regex", "NotRegex",
                "CountEqual", "DictEqual", "SetEqual", "ListEqual", "TupleEqual",
            ):
                return True
    return False


def _function_has_mock(node: ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in ("Mock", "MagicMock", "patch", "monkeypatch"):
                return True
    return False


def _function_uses_pytest_raises(node: ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in ("raises", "warns", "fail"):
                return True
    return False


def _function_raises_in_with(node: ast.FunctionDef) -> bool:
    """Detect ``with pytest.raises(...):`` patterns (also a valid assertion)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.With):
            for item in sub.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    func = call.func
                    name = None
                    if isinstance(func, ast.Attribute):
                        name = func.attr
                    elif isinstance(func, ast.Name):
                        name = func.id
                    if name == "raises":
                        return True
    return False


def _function_uses_pytest_skip(node: ast.FunctionDef) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == "skip":
                return True
    return False


def _analyze_function(node: ast.FunctionDef) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    has_assert = _function_has_assert(node)
    has_mock = _function_has_mock(node)
    has_raises = _function_uses_pytest_raises(node) or _function_raises_in_with(node)
    has_skip = _function_uses_pytest_skip(node)

    if not has_assert and not has_raises and not has_skip:
        candidates.append({
            "kind": "no_assert_in_test",
            "line": node.lineno,
            "message": f"Test function '{node.name}' has no assert/raises/skip",
        })

    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            for cat, detector in WEAK_ASSERT_CATEGORIES.items():
                try:
                    if detector(sub):
                        candidates.append({
                            "kind": f"weak_assert:{cat}",
                            "line": sub.lineno,
                            "message": f"Assert at line {sub.lineno} matches weak pattern '{cat}'",
                        })
                except Exception:
                    pass

    if has_mock and not has_raises and not has_assert:
        candidates.append({
            "kind": "mock_only",
            "line": node.lineno,
            "message": "Test uses mocks/patches but has no behavioral assertion or raises",
        })

    return {
        "name": node.name,
        "line": node.lineno,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "has_assert": has_assert,
        "has_mock": has_mock,
        "has_raises": has_raises,
    }


def audit_test_file(path: Path) -> Dict[str, Any]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return {"path": str(path), "error": f"parse error: {e}"}
    tests = []
    for node in ast.walk(tree):
        if _is_test_function(node):
            tests.append(_analyze_function(node))
    return {
        "path": str(path),
        "total_tests": len(tests),
        "tests_with_candidates": sum(1 for t in tests if t["candidate_count"] > 0),
        "tests": tests,
    }


def audit_tests(tests_dir: Path = Path("TESTS")) -> Dict[str, Any]:
    files = sorted(tests_dir.rglob("test_*.py"))
    results = []
    total_tests = 0
    total_candidates = 0
    for path in files:
        result = audit_test_file(path)
        results.append(result)
        if "tests" in result:
            total_tests += result["total_tests"]
            for t in result["tests"]:
                total_candidates += t["candidate_count"]
    return {
        "scanned_files": len(files),
        "total_tests": total_tests,
        "total_candidates": total_candidates,
        "files": results,
    }


if __name__ == "__main__":
    report = audit_tests()
    print(json.dumps({
        "scanned_files": report["scanned_files"],
        "total_tests": report["total_tests"],
        "total_candidates": report["total_candidates"],
        "candidate_breakdown": {},
    }, indent=2))
    candidates_by_kind: Dict[str, int] = {}
    for f in report["files"]:
        for t in f.get("tests", []):
            for c in t["candidates"]:
                candidates_by_kind[c["kind"]] = candidates_by_kind.get(c["kind"], 0) + 1
    print("Candidate kinds:")
    for k, v in sorted(candidates_by_kind.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
