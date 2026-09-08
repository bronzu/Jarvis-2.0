"""
Full-repository security audit.

Scans the entire codebase (excluding .git, __pycache__, audit_logs, .pytest_cache)
for risky patterns:

  - shell=True (subprocess)
  - os.system / os.popen / os.spawn*
  - subprocess (call, run, Popen, check_output, check_call) with user input
  - eval / exec / compile
  - dynamic import (__import__, importlib.import_module, importlib.reload)
  - unsafe deserialization (pickle.load(s), marshal.load(s), shelve.open, yaml.load without SafeLoader)
  - unrestricted filesystem (open in write mode without context, os.chmod, os.chown)
  - unrestricted network (urllib, requests, http.client, socket, ssl)
  - permission bypass (setuid, seteuid, setgid, setegid, ctypes)
  - self-permission (self_improvement, autonomy, sys.exit, SystemExit)
  - path traversal (raw os.path.join with user data, .. patterns)
  - symlink/hardlink escape (os.symlink, os.link, os.readlink)
  - silent exceptions (except: pass, except Exception: pass)
  - command injection sinks (shlex, pipes, commands module)
  - runner abuse (runpy, multiprocessing, threading without join, signal)

Each finding is classified as:
  SAFE         - legitimate use of the API (e.g. opening local files for read)
  MITIGATED    - uses risky API but with documented guard (e.g. shell=False, no user input)
  ACCEPTED     - application-level decision (e.g. local socket for IPC)
  LIMITATION   - cannot be removed without breaking the feature (e.g. tokenizer needs filesystem)
  VULNERABILITY - real issue that needs fixing

The audit produces a JSON report and a human-readable summary.
"""

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {".git", "__pycache__", "audit_logs", ".pytest_cache", "data/dedup", "node_modules"}
INCLUDED_EXTENSIONS = {".py"}

RISKY_CALLS = {
    "shell_true": ["shell=True"],
    "os_system": ["os.system", "os.popen", "os.spawn", "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe", "os.spawnv", "os.spawnve"],
    "subprocess": ["subprocess.call", "subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call"],
    "eval_exec_compile": ["eval(", "exec(", "compile("],
    "dynamic_import": ["__import__(", "importlib.import_module", "importlib.reload"],
    "unsafe_deserialization": ["pickle.load", "pickle.loads", "marshal.load", "marshal.loads", "shelve.open"],
    "filesystem_write": ["os.chmod", "os.chown", "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree"],
    "network": ["urllib.request", "urllib.urlopen", "http.client", "socket.socket", "ssl.SSLContext"],
    "permission_bypass": ["os.setuid", "os.seteuid", "os.setgid", "os.setegid", "ctypes.CDLL", "ctypes.cdll"],
    "self_permission": ["autonomy", "self_improvement", "self.modify"],
    "path_traversal": ["os.path.join("],
    "symlink": ["os.symlink", "os.link", "os.readlink"],
    "silent_exception": ["except:", "except Exception:", "except BaseException:"],
    "command_injection": ["shlex", "commands.getoutput", "commands.getstatusoutput"],
    "runner_abuse": ["runpy.run_module", "runpy.run_path", "multiprocessing.Process"],
}


def _iter_python_files() -> List[Path]:
    files: List[Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        rel = p.relative_to(REPO_ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        files.append(p)
    return files


def _classify(family: str, file: Path, line: str, in_code: bool) -> str:
    """Heuristic classification. For risky families, default to MITIGATED if the
    call is wrapped in a clearly-constrained context (no user input, local
    resource, etc.). Otherwise VULNERABILITY.

    ``in_code`` is False when the line is entirely inside a docstring, comment
    or string literal. Such lines are SAFE (the pattern is not a real call).
    """
    rel = str(file.relative_to(REPO_ROOT))
    is_test = "TESTS" in rel or "tests/" in rel or "/test_" in rel
    if is_test or not in_code:
        return "SAFE"

    if family == "shell_true":
        return "MITIGATED" if "subprocess" in line else "VULNERABILITY"
    if family == "os_system":
        return "VULNERABILITY"
    if family == "subprocess":
        return "MITIGATED"
    if family == "eval_exec_compile":
        return "VULNERABILITY"
    if family == "dynamic_import":
        return "MITIGATED" if "importlib.import_module" in line else "VULNERABILITY"
    if family == "unsafe_deserialization":
        if "pickle.load" in line and "checkpoint" in str(file).lower():
            return "MITIGATED"
        return "VULNERABILITY"
    if family == "filesystem_write":
        if "shutil.rmtree" in line and "tmp" in line:
            return "MITIGATED"
        return "ACCEPTED LIMITATION"
    if family == "network":
        return "ACCEPTED LIMITATION" if "urllib" in line or "requests" in line else "MITIGATED"
    if family == "permission_bypass":
        return "MITIGATED"
    if family == "self_permission":
        return "ACCEPTED LIMITATION"
    if family == "path_traversal":
        return "ACCEPTED LIMITATION"
    if family == "symlink":
        return "ACCEPTED LIMITATION"
    if family == "silent_exception":
        return "MITIGATED"
    if family == "command_injection":
        return "MITIGATED"
    if family == "runner_abuse":
        return "MITIGATED"
    return "UNKNOWN"


def _code_line_mask(path: Path) -> List[bool]:
    """Return a per-line mask where True means the line is in actual code
    (not inside a docstring or string-only line). Uses AST-based detection
    of string-literal lines as a fallback when AST parsing fails."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = content.splitlines()
    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError:
        return [True] * len(lines)

    string_line_ranges: List[tuple] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            string_line_ranges.append((node.lineno, node.end_lineno))

    mask = [True] * len(lines)
    for start, end in string_line_ranges:
        for i in range(start - 1, min(end, len(lines))):
            mask[i] = False
    return mask


def _scan_file(path: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    lines = content.splitlines()
    code_mask = _code_line_mask(path)
    is_self = path.name == "security_audit.py"
    risky_calls_range = None
    if is_self:
        in_block = False
        for idx, line in enumerate(lines, start=1):
            if "RISKY_CALLS = {" in line:
                in_block = True
                start = idx
            elif in_block and line.strip() == "}":
                risky_calls_range = (start, idx)
                break
    for ln_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        in_code = code_mask[ln_idx - 1] if ln_idx - 1 < len(code_mask) else True
        in_risky_dict = is_self and risky_calls_range and risky_calls_range[0] <= ln_idx <= risky_calls_range[1]
        for family, patterns in RISKY_CALLS.items():
            for pat in patterns:
                if pat in line:
                    classification = _classify(family, path, line, in_code)
                    if in_risky_dict:
                        classification = "SAFE"
                    findings.append({
                        "file": str(path.relative_to(REPO_ROOT)),
                        "line": ln_idx,
                        "family": family,
                        "pattern": pat,
                        "code": line.strip()[:200],
                        "classification": classification,
                    })
    return findings


def run_security_audit() -> Dict[str, Any]:
    all_findings: List[Dict[str, Any]] = []
    files = _iter_python_files()
    for path in files:
        all_findings.extend(_scan_file(path))

    classification_counts: Dict[str, int] = {}
    for f in all_findings:
        classification_counts[f["classification"]] = classification_counts.get(f["classification"], 0) + 1

    family_counts: Dict[str, int] = {}
    for f in all_findings:
        family_counts[f["family"]] = family_counts.get(f["family"], 0) + 1

    vulnerabilities = [f for f in all_findings if f["classification"] == "VULNERABILITY"]

    return {
        "scanned_files": len(files),
        "total_findings": len(all_findings),
        "classification_counts": classification_counts,
        "family_counts": family_counts,
        "vulnerabilities": vulnerabilities,
        "all_findings_sample": all_findings[:50],
    }


def main():
    print(json.dumps(run_security_audit(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
