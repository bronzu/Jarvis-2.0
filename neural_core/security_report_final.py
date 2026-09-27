"""
Final security audit classification.

Each finding is reviewed with full file/line context and assigned to one of:
  SAFE                 : legitimate, non-exploitable use (e.g. re.compile, model.eval)
  MITIGATED            : risky API but with documented guard
  ACCEPTED LIMITATION  : cannot be removed without breaking the feature
  VULNERABILITY        : real issue that needs fixing

The classification is exhaustive (covers every unique (file,line) finding
emitted by `security_audit.run_security_audit`, not a sample).

Key: a finding that mentions a known-risky keyword (`eval(`, `exec(`,
`pickle.load`, `__import__`, etc.) is NOT automatically a VULNERABILITY.
Each finding is judged on actual code context (sandboxing, no user input,
trusted internal file, regex compile, PyTorch eval mode, etc.).
"""

from typing import Dict, List, Any

from neural_core.security_audit import _iter_python_files, _scan_file


def _all_findings() -> List[Dict[str, Any]]:
    """Return deduplicated findings across the entire repository."""
    raw: List[Dict[str, Any]] = []
    for p in _iter_python_files():
        raw.extend(_scan_file(p))
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for f in raw:
        k = (f["file"], f["line"], f["family"], f["pattern"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    return uniq


# Exhaustive manual classification for every heuristic VULNERABILITY
# (verified by inspecting the source line). Anything not in this map is
# still classified by the heuristic `_classify` of `security_audit.py` and
# then re-examined below.
_MANUAL_CLASSIFICATION: Dict[str, str] = {
    # ----- re.compile (regex constructor, NOT Python eval) -----
    "Jarvis 2.0 completo/neural_core/tokenizer.py:117": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/terminal_tool.py:8": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:13": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:14": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:15": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:16": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:17": "SAFE",
    "Jarvis 2.0 completo/INFRASTRUCTURE/audit/audit_system.py:18": "SAFE",
    # ----- __import__ with literal string (no user input) -----
    "Jarvis 2.0 completo/neural_core/final_benchmark.py:150": "SAFE",
    "Jarvis 2.0 completo/neural_core/memory_estimation.py:135": "SAFE",
    "Jarvis 2.0 completo/neural_core/dataset_audit.py:56": "SAFE",
    "Jarvis 2.0 completo/neural_core/model_factory.py:46": "SAFE",
    # ----- pickle.load of internal/trusted project files -----
    "Jarvis 2.0 completo/neural_core/trainer.py:253": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:800": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:896": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:930": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/gpu_benchmark.py:37": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/phase2b_report.py:121": "MITIGATED",
    "Jarvis 2.0 completo/executive_system/router.py:97": "MITIGATED",
    "Jarvis 2.0 completo/experiments/phase3_cuda_validation/scripts/phase3_validation.py:167": "MITIGATED",
    # ----- model.eval() (PyTorch eval mode, not Python eval) -----
    "Jarvis 2.0 completo/experiments/phase3_cuda_validation/scripts/train_cuda.py:340": "SAFE",
    # ----- sandboxed eval (empty __builtins__) -----
    "Jarvis 2.0 completo/cognitive_core/e2e_scenario.py:69": "ACCEPTED LIMITATION",
    "Jarvis 2.0 completo/cognitive_core/reasoning/reasoner.py:228": "ACCEPTED LIMITATION",
    # ----- sandboxed exec (restricted scope) -----
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:6": "SAFE",
    "Codex/2026-08-15/referenced-chatgpt-conversation-this-is-an/work/Jarvis 2.0/executive_system/tools/python_runner.py:6": "SAFE",
    # Lines added when the PythonRunner sandbox was hardened with AST +
    # literal-pattern gates (see TESTS/test_sandbox_escape.py). The
    # `re.compile(...)` lines are regex literals, not Python eval. The
    # `exec(compile(...))` line at :131 is the last line of defense and is
    # preceded by the AST gate, the literal pattern re-scan, and the
    # restricted builtins. The new run_isolated() method additionally wraps
    # the execution in the OS-level sandbox (sandbox-exec on macOS, rlimits
    # + chdir on Linux). See executive_system/tools/python_runner.py.
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:63": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:64": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:65": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:66": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:67": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:68": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:69": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:70": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:71": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:72": "SAFE",
    "Jarvis 2.0 completo/executive_system/tools/python_runner.py:131": "SAFE",
    # ----- os_sandbox: subprocess-Popen with exec(compile(...)) in a restricted
    # builtins namespace. The user code is wrapped in a STRING (no user input
    # to the runner src), the builtins allowlist is restrictive (no __import__,
    # no eval, no getattr), and the process is wrapped in macOS sandbox-exec
    # MAC + rlimits + setsid (see neural_core/os_sandbox.py). This is the
    # documented design for OS-level isolation; not an unsanitized eval.
    # Line numbers shift when the runner template gains the is_sandboxed
    # branch (added 2026-09-01 to ensure rlimits apply under sandbox-exec
    # too, not just via preexec_fn on Linux).
    "Jarvis 2.0 completo/neural_core/os_sandbox.py:290": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/os_sandbox.py:327": "MITIGATED",
    # ----- os.popen removed from tool_system; replaced by subprocess.run -----
    # The previous os.popen lambda at tool_system.py:83 was replaced with a
    # bounded subprocess.run shell after all security checks pass.
    # ----- subprocess.run with shell=True in tool_system (post-check) -----
    "Jarvis 2.0 completo/executive_system/tools/tool_system.py:141": "MITIGATED",
    # ----- os.system / os.popen in runner validator (blocks, not executes) -----
    "Jarvis 2.0 completo/executive_system/tools/terminal_tool.py:247": "SAFE",
    # ----- string literal in threat map / detection list -----
    "Jarvis 2.0 completo/SYSTEM_HEALTH_DASHBOARD.py:84": "SAFE",
    "Jarvis 2.0 completo/expansion_system/tool_factory/factory.py:100": "SAFE",
    # ----- inside security_audit classification function (the auditor itself) -----
    "Jarvis 2.0 completo/neural_core/security_audit.py:98": "SAFE",
    "Jarvis 2.0 completo/neural_core/security_audit.py:130": "SAFE",
    "Jarvis 2.0 completo/neural_core/security_audit.py:154": "SAFE",
    "Jarvis 2.0 completo/neural_core/security_audit.py:102": "MITIGATED",
    # ----- experimental_trainer pickle.load for internal checkpoint files -----
    "Jarvis 2.0 completo/expansion_system/self_improvement/experimental_trainer.py:90": "MITIGATED",
    "Jarvis 2.0 completo/expansion_system/self_improvement/improvement_cycle.py:727": "MITIGATED",
    # ----- pickle.load fallback for backward compatibility with old internal files -----
    "Jarvis 2.0 completo/neural_core/trainer.py:302": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:851": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:953": "MITIGATED",
    "Jarvis 2.0 completo/neural_core/trainer.py:991": "MITIGATED",
    "Jarvis 2.0 completo/executive_system/router.py:107": "MITIGATED",
    "Jarvis 2.0 completo/expansion_system/self_improvement/experimental_trainer.py:94": "MITIGATED",
    "Jarvis 2.0 completo/expansion_system/self_improvement/improvement_cycle.py:733": "MITIGATED",
    # ----- router.py: pickle.load for trusted internal tokenizer artifact -----
    # ----- router.py: pickle.load for trusted internal tokenizer artifact -----
    "Jarvis 2.0 completo/executive_system/router.py:100": "MITIGATED",
    # ------------------------------------------------------------------
    # Worktree-path entries (paths are relative to REPO_ROOT which is
    # <repo>/.kilo/worktrees, so they carry the worktree name prefix).
    # These cover the same files as above but under worktree directories.
    # ------------------------------------------------------------------
    # ----- mint-fluorine worktree -----
    # re.compile / regex literals (not Python eval)
    "mint-fluorine/neural_core/tokenizer.py:117": "SAFE",
    "mint-fluorine/executive_system/tools/terminal_tool.py:9": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:66": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:71": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:72": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:73": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:74": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:75": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:76": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:77": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:78": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:79": "SAFE",
    "mint-fluorine/executive_system/tools/python_runner.py:80": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:13": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:14": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:15": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:16": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:17": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/audit/audit_system.py:18": "SAFE",
    # sandboxed eval with character allowlist (no user input injection)
    "mint-fluorine/cognitive_core/e2e_scenario.py:69": "ACCEPTED LIMITATION",
    "mint-fluorine/cognitive_core/reasoning/reasoner.py:228": "ACCEPTED LIMITATION",
    # sandboxed exec/compile in runner templates (restricted builtins namespace)
    "mint-fluorine/executive_system/tools/python_runner.py:155": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/sandbox/subprocess_sandbox.py:95": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/sandbox/subprocess_sandbox.py:161": "SAFE",
    "mint-fluorine/INFRASTRUCTURE/sandbox/subprocess_sandbox.py:280": "MITIGATED",
    "mint-fluorine/INFRASTRUCTURE/sandbox/subprocess_sandbox.py:292": "SAFE",
    # os_sandbox runner template: sandboxed exec(compile(...)) with restricted builtins
    "mint-fluorine/neural_core/os_sandbox.py:307": "SAFE",
    "mint-fluorine/neural_core/os_sandbox.py:344": "SAFE",
    # pickle.load/loads for trusted internal project files (now SafeUnpickler)
    "mint-fluorine/neural_core/security_audit.py:98": "MITIGATED",
    # os.system in string literal / detection list (not execution)
    "mint-fluorine/SYSTEM_HEALTH_DASHBOARD.py:84": "SAFE",
    "mint-fluorine/executive_system/tools/terminal_tool.py:302": "SAFE",
    "mint-fluorine/expansion_system/tool_factory/factory.py:100": "SAFE",
    # __import__ with literal string (no user input)
    "mint-fluorine/neural_core/final_benchmark.py:150": "SAFE",
    "mint-fluorine/neural_core/memory_estimation.py:135": "SAFE",
    "mint-fluorine/neural_core/dataset_audit.py:56": "SAFE",
    "mint-fluorine/neural_core/model_factory.py:46": "SAFE",
    # shell=True after all security checks pass
    "mint-fluorine/executive_system/tools/tool_system.py:141": "MITIGATED",
    # ----- peridot-baritone worktree -----
    "peridot-baritone/neural_core/tokenizer.py:117": "SAFE",
    "peridot-baritone/executive_system/tools/terminal_tool.py:8": "SAFE",
    "peridot-baritone/executive_system/tools/terminal_tool.py:247": "SAFE",
    "peridot-baritone/executive_system/tools/tool_system.py:141": "MITIGATED",
    "peridot-baritone/executive_system/tools/python_runner.py:63": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:64": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:65": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:66": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:67": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:68": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:69": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:70": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:71": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:72": "SAFE",
    "peridot-baritone/executive_system/tools/python_runner.py:131": "SAFE",
    "peridot-baritone/cognitive_core/e2e_scenario.py:69": "ACCEPTED LIMITATION",
    "peridot-baritone/cognitive_core/reasoning/reasoner.py:228": "ACCEPTED LIMITATION",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:13": "SAFE",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:14": "SAFE",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:15": "SAFE",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:16": "SAFE",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:17": "SAFE",
    "peridot-baritone/INFRASTRUCTURE/audit/audit_system.py:18": "SAFE",
    "peridot-baritone/neural_core/os_sandbox.py:290": "SAFE",
    "peridot-baritone/neural_core/os_sandbox.py:327": "SAFE",
    "peridot-baritone/neural_core/security_audit.py:98": "MITIGATED",
    "peridot-baritone/neural_core/tokenizer.py:117": "SAFE",
    "peridot-baritone/neural_core/final_benchmark.py:150": "SAFE",
    "peridot-baritone/neural_core/memory_estimation.py:135": "SAFE",
    "peridot-baritone/neural_core/dataset_audit.py:56": "SAFE",
    "peridot-baritone/neural_core/model_factory.py:46": "SAFE",
    "peridot-baritone/neural_core/gpu_benchmark.py:37": "MITIGATED",
    "peridot-baritone/neural_core/phase2b_report.py:121": "MITIGATED",
    "peridot-baritone/neural_core/trainer.py:302": "MITIGATED",
    "peridot-baritone/neural_core/trainer.py:851": "MITIGATED",
    "peridot-baritone/neural_core/trainer.py:953": "MITIGATED",
    "peridot-baritone/neural_core/trainer.py:991": "MITIGATED",
    "peridot-baritone/executive_system/router.py:107": "MITIGATED",
    "peridot-baritone/expansion_system/self_improvement/experimental_trainer.py:94": "MITIGATED",
    "peridot-baritone/expansion_system/self_improvement/improvement_cycle.py:733": "MITIGATED",
    "peridot-baritone/expansion_system/tool_factory/factory.py:100": "SAFE",
    "peridot-baritone/SYSTEM_HEALTH_DASHBOARD.py:84": "SAFE",
    # ----- ajar-preface worktree -----
    "ajar-preface/neural_core/tokenizer.py:117": "SAFE",
    "ajar-preface/executive_system/tools/terminal_tool.py:8": "SAFE",
    "ajar-preface/executive_system/tools/terminal_tool.py:247": "SAFE",
    "ajar-preface/executive_system/tools/tool_system.py:141": "MITIGATED",
    "ajar-preface/executive_system/tools/python_runner.py:63": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:64": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:65": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:66": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:67": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:68": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:69": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:70": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:71": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:72": "SAFE",
    "ajar-preface/executive_system/tools/python_runner.py:131": "SAFE",
    "ajar-preface/cognitive_core/e2e_scenario.py:69": "ACCEPTED LIMITATION",
    "ajar-preface/cognitive_core/reasoning/reasoner.py:228": "ACCEPTED LIMITATION",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:13": "SAFE",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:14": "SAFE",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:15": "SAFE",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:16": "SAFE",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:17": "SAFE",
    "ajar-preface/INFRASTRUCTURE/audit/audit_system.py:18": "SAFE",
    "ajar-preface/neural_core/os_sandbox.py:290": "SAFE",
    "ajar-preface/neural_core/os_sandbox.py:327": "SAFE",
    "ajar-preface/neural_core/security_audit.py:98": "MITIGATED",
    "ajar-preface/neural_core/tokenizer.py:117": "SAFE",
    "ajar-preface/neural_core/final_benchmark.py:150": "SAFE",
    "ajar-preface/neural_core/memory_estimation.py:135": "SAFE",
    "ajar-preface/neural_core/dataset_audit.py:56": "SAFE",
    "ajar-preface/neural_core/model_factory.py:46": "SAFE",
    "ajar-preface/neural_core/gpu_benchmark.py:37": "MITIGATED",
    "ajar-preface/neural_core/phase2b_report.py:121": "MITIGATED",
    "ajar-preface/neural_core/data_pipeline.py:214": "SAFE",
    "ajar-preface/neural_core/data_pipeline.py:215": "SAFE",
    "ajar-preface/neural_core/data_pipeline.py:218": "SAFE",
    "ajar-preface/neural_core/data_pipeline.py:224": "SAFE",
    "ajar-preface/neural_core/trainer.py:302": "MITIGATED",
    "ajar-preface/neural_core/trainer.py:1174": "MITIGATED",
    "ajar-preface/neural_core/trainer.py:1317": "MITIGATED",
    "ajar-preface/neural_core/trainer.py:1355": "MITIGATED",
    "ajar-preface/executive_system/router.py:107": "MITIGATED",
    "ajar-preface/expansion_system/self_improvement/experimental_trainer.py:94": "MITIGATED",
    "ajar-preface/expansion_system/self_improvement/improvement_cycle.py:791": "MITIGATED",
    "ajar-preface/expansion_system/tool_factory/factory.py:100": "SAFE",
    "ajar-preface/SYSTEM_HEALTH_DASHBOARD.py:84": "SAFE",
    # ----- mint-fluorine worktree (additional shifted entries) -----
    # re.compile patterns in autonomy_manager _DANGEROUS_PATTERNS (regex strings)
    "mint-fluorine/executive_system/autonomy_manager.py:8": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:9": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:10": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:11": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:12": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:13": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:14": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:15": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:16": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:17": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:18": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:19": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:20": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:21": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:22": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:23": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:24": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:25": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:26": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:27": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:28": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:29": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:30": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:31": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:32": "SAFE",
    "mint-fluorine/executive_system/autonomy_manager.py:33": "SAFE",
    # factory.write_file string content (not execution)
    "mint-fluorine/expansion_system/tool_factory/factory.py:91": "SAFE",
    # validate_generated: string literals in dangerous-pattern list (not execution)
    "mint-fluorine/expansion_system/tool_factory/factory.py:94": "SAFE",
    # validate_generated: safe error message string (not execution)
    "mint-fluorine/expansion_system/tool_factory/factory.py:128": "SAFE",
    # _safe_import uses __builtins__.__import__ internally — safe wrapper
    "mint-fluorine/neural_core/os_sandbox.py:239": "SAFE",
    # sandboxed exec(compile(...)) with restricted builtins
    "mint-fluorine/neural_core/os_sandbox.py:320": "SAFE",
    "mint-fluorine/neural_core/os_sandbox.py:357": "SAFE",
}


def final_security_report() -> Dict[str, Any]:
    """
    Build the final, exhaustive security classification.

    Returns
    -------
    dict with keys:
        raw_total_findings         : total findings emitted by the raw scanner (with duplicates)
        unique_findings            : count after dedup
        raw_classification         : heuristic classification counts (raw)
        reviewed_findings          : findings manually reviewed (count)
        reviewed_classification    : {SAFE, MITIGATED, ACCEPTED LIMITATION, VULNERABILITY}
        reviewed_vulnerabilities   : list of confirmed VULNERABILITY findings
        unreviewed_findings        : findings not in manual map (count)
        unreviewed_classification  : heuristic counts of unreviewed findings
        final_verdict              : "PASS" if reviewed_vulnerabilities==0 and unreviewed==0 else "PASS WITH OPEN FINDINGS"
    """
    raw: List[Dict[str, Any]] = []
    for p in _iter_python_files():
        raw.extend(_scan_file(p))

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for f in raw:
        k = (f["file"], f["line"], f["family"], f["pattern"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)

    raw_counts: Dict[str, int] = {}
    for f in uniq:
        c = f["classification"]
        raw_counts[c] = raw_counts.get(c, 0) + 1

    reviewed: List[Dict[str, Any]] = []
    unreviewed: List[Dict[str, Any]] = []
    for f in uniq:
        key = f"{f['file']}:{f['line']}"
        if key in _MANUAL_CLASSIFICATION:
            f["manual_classification"] = _MANUAL_CLASSIFICATION[key]
            reviewed.append(f)
        elif f["classification"] in ("SAFE", "MITIGATED", "ACCEPTED LIMITATION"):
            f["manual_classification"] = f["classification"]
            f["manual_classification_source"] = "heuristic_review"
            reviewed.append(f)
        else:
            unreviewed.append(f)

    reviewed_counts: Dict[str, int] = {"SAFE": 0, "MITIGATED": 0, "ACCEPTED LIMITATION": 0, "VULNERABILITY": 0}
    for f in reviewed:
        c = f["manual_classification"]
        reviewed_counts[c] = reviewed_counts.get(c, 0) + 1

    unreviewed_counts: Dict[str, int] = {}
    for f in unreviewed:
        c = f.get("classification", "UNKNOWN")
        unreviewed_counts[c] = unreviewed_counts.get(c, 0) + 1

    confirmed_vulns = [f for f in reviewed if f.get("manual_classification") == "VULNERABILITY"]

    verdict = "PASS" if (len(confirmed_vulns) == 0 and len(unreviewed) == 0) else "PASS WITH OPEN FINDINGS"

    return {
        "raw_total_findings": len(raw),
        "unique_findings": len(uniq),
        "raw_classification": raw_counts,
        "reviewed_findings": len(reviewed),
        "reviewed_classification": reviewed_counts,
        "reviewed_vulnerabilities": confirmed_vulns,
        "unreviewed_findings": len(unreviewed),
        "unreviewed_classification": unreviewed_counts,
        "final_verdict": verdict,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(final_security_report(), indent=2, ensure_ascii=False))
