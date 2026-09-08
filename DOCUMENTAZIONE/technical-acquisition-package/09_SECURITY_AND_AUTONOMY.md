# Security and Autonomy

## 1. Security Architecture

### 1.1 Defense-in-Depth Layers

| Layer | Component | Mechanism | Coverage |
|---|---|---|---|
| L1 | PermissionGuard | Regex blacklist (21 patterns, 17 blocked commands) | Command validation |
| L2 | SecurityLayer | Wraps PermissionGuard + AuditSystem | Unified security interface |
| L3 | AuditSystem | Structured event logging with 40+ event types | Audit trail |
| L4 | NetworkPolicy | Host allowlist/blocked, protocol/port policy | Network boundaries |
| L5 | SandboxEnvironment | Path confinement, resource limits | Filesystem boundaries |
| L6 | OSSandbox | macOS `sandbox-exec` MAC + Linux `chroot`/`rlimits` | OS-level process isolation |

### 1.2 PermissionGuard Dangerous Patterns
```python
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/?(?!\s)",       # rm -rf /
    r"rm\s+-rf\s+~",              # rm -rf ~
    r"chmod\s+-R\s+777",          # recursive chmod 777
    r"mkfs",                      # format filesystem
    r"dd\s+if=.*of=/dev/sd",      # disk wipe
    r":\(\)\s*\{.*\};: ",         # fork bomb
    r"\$\(.*\)\s*\|.*sh",         # command substitution pipe
    # ... 14 more patterns
]
```

### 1.3 PythonRunner 5-Layer Defense
1. **Permission gate** — `approved=True` required
2. **AST rejection** — blocks `Import`, `ImportFrom`, high-risk builtins (`exec`, `eval`, `compile`, `open`, `input`, `__import__`, etc.), and dunder introspection (`__class__`, `__bases__`, `__subclasses__`, etc.)
3. **Literal re-scan** — independent regex scan for `os.system`, `subprocess`, `shutil`, etc.
4. **Restricted builtins** — `__builtins__ = {"abs": abs, "sum": sum, "len": len}`
5. **OS isolation** — `os_sandbox.run_isolated()` with kernel-enforced boundaries

### 1.4 Secret Redaction
AuditSystem redacts sensitive patterns before logging:
- `api_key`, `password`, `token`, `Bearer`, `sk-...` (OpenAI keys)
- Patterns defined in `audit_system.py`

---

## 2. Security Audit Results

### 2.1 Latest Audit (`FINAL_CLOSURE_AUDIT.json`)
```json
{
  "total_findings": 315,
  "vulnerabilities": 0,
  "safe": 298,
  "mitigated": 17,
  "critical": 0,
  "high": 0,
  "medium": 0,
  "low": 0
}
```

**Status:** 0 confirmed vulnerabilities, 0 unreviewed findings.

### 2.2 Critical Finding (Fixed)
During the closure audit, a **real sandbox escape vulnerability** was discovered and fixed:
- **Exploit:** `().__class__.__bases__[0].__subclasses__()` + `_wrap_close` type walk to reach `os.system`
- **Fix:** 4-layer defense in `python_runner.py` (permission, AST, literal, builtins) + OS sandbox integration
- **Regression tests:** 9 tests in `test_sandbox_escape.py`
- **Mutation testing:** Confirms tests catch real bypasses

### 2.3 Security Test Coverage
| Test Category | Tests | What They Verify |
|---|---|---|
| Permission guard | 23 | Dangerous commands blocked, approval flow, runner validation |
| Sandbox escape | 9 | Type walk, os.system, open, eval, getattr, attribute introspection |
| OS sandbox | 9 | rlimits, chroot, sandbox-exec, timeout, env sanitization |
| Layered security | 10 | Full defense-in-depth chain |
| Autonomy guardrails | ~10 | Autonomy manager enforces boundaries |

---

## 3. Autonomy Analysis

### 3.1 Autonomy Configuration
```yaml
autonomy:
  enabled: false
  requirements:
    goal_manager: true
    planner: true
    executor: true
  restrictions:
    modify_permissions: false
    bypass_security: false
```

**Status:** Autonomy is **disabled by default**. Even if enabled, it cannot modify permissions or bypass security.

### 3.2 Self-Improvement Configuration
```yaml
self_improvement:
  enabled: false
  mode:
    controlled_proposal_only: true
  requirements:
    sandbox: true
    benchmark: true
    tests_required: true
    training: true
  triggers:
    manual: true
    periodic: false
    weakness_based: true
  limits:
    max_candidates_per_cycle: 3
    max_concurrent_cycles: 1
    cycle_timeout_seconds: 300
  thresholds:
    min_improvement: 0.01
    critical_regression_block: true
  approval:
    require_user_approval: true
    auto_deploy: false
```

**Status:** Self-improvement is **disabled by default**. It requires `approved=True` to run, cannot auto-deploy, and blocks critical regressions.

### 3.3 Device Control Configuration
```yaml
vision:
  enabled: false
  screenshot_enabled: false
  ocr_enabled: false
```

**Status:** Vision, screenshot, and OCR are **disabled by default**.

---

## 4. Sandbox Analysis

### 4.1 Application-Level vs OS-Level
| Aspect | Application-Level | OS-Level |
|---|---|---|
| Path validation | `SandboxEnvironment._safe_path()` | `os.chdir()` / `chroot()` |
| Command filtering | `PermissionGuard` regex | `sandbox-exec` MAC profile |
| Resource limits | None | `rlimit` (CPU, AS, FSIZE, NPROC, NOFILE, CORE) |
| Process isolation | None | `setsid()` (new session) |
| Environment sanitization | None | Strip to PATH/LANG/LC_ALL/TERM |
| Kernel enforcement | NO | YES (macOS) / PARTIAL (Linux) |

### 4.2 macOS Sandbox Profile
```
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*)
(allow file-write* (require-all (subpath "/path/to/sandbox")))
```

**Enforcement:** XNU kernel Mandatory Access Control. Cannot be bypassed from user space.

### 4.3 Linux Isolation
- **As root:** `chroot()` + `rlimits` + `setsid` + `umask 0o077`
- **As non-root:** `chdir()` + `rlimits` + `setsid` + `umask 0o077`
- **Limitation:** Non-root `chdir` does not provide filesystem isolation — only process-level isolation.

### 4.4 Smoke Test Results
| Test | Result |
|---|---|
| Legitimate code runs | PASS |
| Escape attempt blocked | PASS |
| rlimits effective | PASS |
| setsid effective | PASS |
| umask applied | PASS |
| env sanitized | PASS |

---

## 5. Security Risks and Mitigations

| Risk | Severity | Mitigation | Status |
|---|---|---|---|
| Sandbox escape via type walk | CRITICAL (pre-fix) | 5-layer defense + OS sandbox | FIXED |
| Command injection | HIGH | PermissionGuard + AST gate | MITIGATED |
| Path traversal | HIGH | `_safe_path()` + sandbox | MITIGATED |
| Resource exhaustion | MEDIUM | rlimits + timeout | MITIGATED |
| Secret leakage in logs | MEDIUM | AuditSystem redaction | MITIGATED |
| Unauthorized self-modification | HIGH | `enabled: false` + approval required | MITIGATED |
| Network data exfiltration | MEDIUM | NetworkPolicy + allow_https config | CONFIGURABLE |
| OS-level escape (non-root Linux) | MEDIUM | Documented limitation | ACCEPTED |

---

## 6. Remaining Security Concerns

1. **Blacklist-based approach** — PermissionGuard uses a blacklist. New attack vectors may not be covered.
2. **No static analysis of generated code** — Custom runners are not analyzed statically; only the AST gate and literal re-scan protect against injection.
3. **Non-root Linux isolation is weak** — `chdir` does not provide filesystem isolation. Containerization is recommended for production.
4. **No runtime integrity monitoring** — No attestation or runtime verification of the sandbox environment.
5. **Audit logs could be tampered with** — No write-once or signed logging.

---

*This document is part of the Technical Acquisition Package.*
