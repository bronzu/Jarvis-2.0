# Technical Risk Register

## 1. Technical Risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|---|
| TR-001 | Model is small (2.7M params) trained on limited corpus (33 MB) | HIGH | CERTAIN | Cannot compete with production LLMs | Retrain on larger dataset; consider transfer learning | ACCEPTED |
| TR-002 | No large-scale language training | HIGH | CERTAIN | Limited conversational ability | Requires significant training investment | OPEN |
| TR-003 | Stale documentation (test counts, integration claims) | MEDIUM | CERTAIN | Misleads acquirer during due diligence | Update all documentation before disclosure | OPEN |
| TR-004 | Dataset overlap warnings in tests | MEDIUM | POSSIBLE | Training data integrity compromised | Investigate and regenerate dataset if needed | OPEN |
| TR-005 | Full test suite not verified (timeout) | MEDIUM | POSSIBLE | Undetected test failures | Run full suite in CI environment | OPEN |
| TR-006 | Single-process architecture | MEDIUM | CERTAIN | Cannot scale to multi-user | Requires distributed architecture redesign | ACCEPTED |
| TR-007 | No production deployment | MEDIUM | CERTAIN | Not ready for production use | Requires containerization, orchestration, monitoring | OPEN |
| TR-008 | OS sandbox portability (macOS-only `sandbox-exec`) | MEDIUM | CERTAIN | Limited deployment platforms | Containerization recommended for Linux/Windows | ACCEPTED |
| TR-009 | Non-root Linux isolation is weak (`chdir` only) | MEDIUM | CERTAIN | Filesystem escape possible on Linux | Deploy in containers or as root | ACCEPTED |
| TR-010 | No external users or real-world validation | MEDIUM | CERTAIN | Unknown production behavior | Beta testing, A/B evaluation | OPEN |
| TR-011 | Blacklist-based security | MEDIUM | POSSIBLE | New attack vectors not covered | Add allowlist-based validation where possible | OPEN |
| TR-012 | No static analysis of generated code | LOW | POSSIBLE | Code injection via custom runners | Add static analysis to runner factory | OPEN |
| TR-013 | Audit logs not write-once or signed | LOW | POSSIBLE | Log tampering | Implement signed audit trail | OPEN |
| TR-014 | No runtime integrity monitoring | LOW | POSSIBLE | Sandbox environment tampering | Implement attestation | OPEN |
| TR-015 | Performance benchmarks are small-scale | LOW | CERTAIN | Unknown production performance | Run benchmarks at production scale | OPEN |

---

## 2. Product Risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|---|
| PR-001 | Missing language training | HIGH | CERTAIN | System cannot engage in natural conversation | Retrain on conversational dataset | OPEN |
| PR-002 | Limited real-world validation | HIGH | CERTAIN | Unknown edge cases and failure modes | Extensive beta testing | OPEN |
| PR-003 | No external user base | MEDIUM | CERTAIN | No product-market fit evidence | User acquisition and feedback loops | OPEN |
| PR-004 | Understanding is rule-based, not neural | MEDIUM | CERTAIN | Limited intent/entity extraction accuracy | Integrate neural NER/intent classifier | OPEN |
| PR-005 | No Bayesian/probabilistic reasoning | MEDIUM | CERTAIN | Cannot handle uncertainty formally | Add probabilistic inference layer | OPEN |
| PR-006 | No formal planning (PDDL) | MEDIUM | CERTAIN | Limited complex task planning | Integrate PDDL planner or enhance existing | OPEN |
| PR-007 | Vision/voice disabled | LOW | CERTAIN | No multi-modal interaction | Re-enable when models are ready | ACCEPTED |

---

## 3. IP Risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|---|
| IP-001 | Dataset provenance unclear | HIGH | POSSIBLE | Copyright infringement, model ownership disputes | Audit all dataset sources, secure licenses | OPEN |
| IP-002 | No LICENSE file in repository | HIGH | CERTAIN | Unclear IP ownership terms | Draft and add LICENSE file | OPEN |
| IP-003 | Transformer architecture may overlap with patents | MEDIUM | POSSIBLE | Patent infringement claims | Freedom-to-operate analysis | OPEN |
| IP-004 | OS sandbox approach may overlap with existing patents | MEDIUM | POSSIBLE | Patent infringement claims | Freedom-to-operate analysis | OPEN |
| IP-005 | No invention disclosures filed | MEDIUM | CERTAIN | Loss of patent eligibility | File provisional patents before public disclosure | OPEN |
| IP-006 | Git history not available | LOW | CERTAIN | Cannot trace individual contributions | Use file modification times and documentation | ACCEPTED |

---

## 4. Security Risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|---|
| SR-001 | Sandbox escape (new vector) | HIGH | POSSIBLE | Arbitrary code execution | Continuous security audit, fuzzing | OPEN |
| SR-002 | Blacklist bypass | MEDIUM | POSSIBLE | Dangerous command execution | Add allowlist validation, static analysis | OPEN |
| SR-003 | Log tampering | MEDIUM | POSSIBLE | Audit trail compromise | Signed, write-once logging | OPEN |
| SR-004 | Unauthorized self-modification | HIGH | UNLIKELY | System compromise | `enabled: false` + approval required | MITIGATED |
| SR-005 | Network data exfiltration | MEDIUM | POSSIBLE | Data leak | NetworkPolicy + TLS verification | CONFIGURABLE |
| SR-006 | Resource exhaustion | MEDIUM | POSSIBLE | DoS via CPU/memory/disk | rlimits + timeout + monitoring | MITIGATED |

---

## 5. Acquisition Risks

| ID | Risk | Severity | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|---|
| AR-001 | Key-person dependency | HIGH | HIGH | Loss of critical knowledge | Document architecture, train team | OPEN |
| AR-002 | Undocumented knowledge | HIGH | HIGH | Incomplete understanding | Comprehensive documentation sprint | OPEN |
| AR-003 | Build reproducibility | MEDIUM | POSSIBLE | Cannot reproduce training/inference | Document environment, dependencies, seeds | OPEN |
| AR-004 | Infrastructure dependency | MEDIUM | POSSIBLE | Cannot run without specific hardware | Containerize, document requirements | OPEN |
| AR-005 | Missing credentials | MEDIUM | POSSIBLE | Cannot access services | Credential transfer process | OPEN |
| AR-006 | Missing deployment instructions | MEDIUM | CERTAIN | Cannot deploy to production | Write deployment guide, Dockerfiles | OPEN |
| AR-007 | Test suite incomplete verification | MEDIUM | POSSIBLE | Undetected regressions | Full CI/CD pipeline with test execution | OPEN |

---

## 6. Risk Heat Map

```
        HIGH IMPACT
             |
    TR-001   |   TR-002
    PR-001   |   PR-002
    IP-001   |   AR-001
    IP-002   |   AR-002
             |
-------------+------------- MEDIUM IMPACT
             |
    TR-003   |   TR-005
    TR-006   |   TR-008
    PR-004   |   PR-006
    IP-003   |   SR-002
    AR-003   |   AR-004
             |
-------------+------------- LOW IMPACT
             |
    TR-012   |   TR-015
    SR-003   |
    AR-005   |
             |
        LOW SEVERITY           HIGH SEVERITY
```

---

*This document is part of the Technical Acquisition Package.*
