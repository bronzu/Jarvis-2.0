# Restricted Disclosure Map

## 1. Disclosure Classification Framework

| Level | Definition | Audience | When to Disclose |
|---|---|---|---|
| **PUBLIC** | Information that can be shown in first contact without NDA | Potential acquirer initial contact | First meeting / teaser materials |
| **NDA-ONLY** | Information requiring signed NDA | Technical evaluators after NDA | Technical overview calls, architecture diagrams |
| **DUE-DILIGENCE ONLY** | Sensitive information for technical due diligence | Technical due diligence team | During formal due diligence phase |
| **TRADE SECRET** | Information that should not be disclosed without contractual need | Legal counsel, acquisition team | Only under specific contractual provisions |

---

## 2. PUBLIC / Safe to Disclose

| Information | Rationale | Evidence |
|---|---|---|
| **System name and purpose** | "JARVIS 2.0 is an AI agent system" — descriptive, not secret | `jarvis.py` header |
| **Technology stack** | Python, NumPy, PyTorch (optional) — standard technologies | `requirements.txt` |
| **High-level architecture** | Multi-layer cognitive agent architecture — standard pattern | `02_ARCHITECTURE.md` |
| **Test count (approximate)** | "Over 1,000 tests" — can be verified by running tests | `pytest --collect-only` |
| **Security posture summary** | "0 confirmed vulnerabilities, comprehensive testing" — can be verified | `FINAL_CLOSURE_AUDIT.json` |
| **Model size and type** | "MicroTransformer, 2.7M parameters, NumPy-based" — factual | `network.py`, `model.pkl` |
| **No external API dependency** | "No runtime dependency on OpenAI/Ollama" — verifiable | Code analysis |
| **OS sandbox capability** | "Kernel-enforced isolation on macOS and Linux" — verifiable | `os_sandbox.py` |
| **Self-improvement disabled** | "Disabled by configuration" — verifiable | `config.yaml` |
| **Device control disabled** | "Vision/screenshot/OCR disabled" — verifiable | `config.yaml` |

**Recommended first-contact materials:**
- `00_EXECUTIVE_SUMMARY.md` (redacted version)
- High-level architecture diagram
- Security posture summary
- Test count and pass rate summary

---

## 3. NDA-ONLY

| Information | Rationale | Evidence |
|---|---|---|
| **Detailed component inventory** | Reveals implementation depth and scope | `03_COMPONENT_INVENTORY.md` |
| **Test quality assessment** | Reveals testing rigor and gaps | `04_VERIFICATION_AND_TEST_AUDIT.md` |
| **Capability matrix** | Reveals what is implemented vs claimed | `05_CAPABILITY_MATRIX.md` |
| **Benchmark results** | Reveals performance characteristics | `06_BENCHMARKS.md` |
| **Differentiation analysis** | Reveals potential IP and competitive advantages | `07_DIFFERENTIATION_AND_IP.md` |
| **Dependency list** | Reveals supply chain and license risks | `08_DEPENDENCIES_AND_PROVENANCE.md` |
| **Security architecture details** | Reveals defense-in-depth mechanisms | `09_SECURITY_AND_AUTONOMY.md` |
| **Risk register** | Reveals known weaknesses and gaps | `10_TECHNICAL_RISK_REGISTER.md` |
| **Demo protocol** | Reveals how to demonstrate capabilities | `13_DEMO_PROTOCOL.md` |

**Recommended NDA-phase materials:**
- Full Technical Acquisition Package (excluding trade secrets)
- Architecture deep-dive presentation
- Security architecture review
- Test suite walkthrough
- Benchmark demonstration

---

## 4. DUE-DILIGENCE ONLY

| Information | Rationale | Evidence |
|---|---|---|
| **Complete source code** | Core IP — must be protected | All `.py` files |
| **Complete test suite** | Reveals testing rigor and edge cases | `TESTS/` directory |
| **Training data** | Reveals model training quality and dataset provenance | `neural_core/data/` |
| **Model weights** | Reveals model capability and quality | `model.pkl`, `best_model.json` |
| **Benchmark artifacts** | Reveals performance data | `FINAL_BENCHMARK.json`, `audit_logs/` |
| **Security audit reports** | Reveals security posture and findings | `FINAL_CLOSURE_AUDIT.json`, `FINAL_ACCEPTANCE_AUDIT.json` |
| **Configuration files** | Reveals system behavior and defaults | `config.yaml` |
| **Deployment instructions** | Reveals infrastructure requirements | To be created |
| **Git history** | Reveals development history and contributor information | Not available (not a git repo) |
| **Dataset provenance** | Reveals data sourcing and licensing | `provenance_manifest.json` |

**Recommended due-diligence materials:**
- Complete source code repository
- Full test suite with CI configuration
- Training data with provenance documentation
- Model weights and checkpoint lineage
- Security audit reports (all versions)
- Benchmark results and raw artifacts
- Deployment guide and infrastructure requirements
- Contributor agreements and IP assignments

---

## 5. TRADE SECRET / Do Not Disclose Initially

| Information | Rationale | Evidence |
|---|---|---|
| **Specific AST gate patterns** | Reveals exact bypass prevention mechanisms | `python_runner.py` `_DENIED_ATTRS` set |
| **OS sandbox profile templates** | Reveals exact MAC profile and rlimit values | `os_sandbox.py` `_sandbox_profile()` |
| **Exact model hyperparameters** | Reveals training recipe and architecture details | `network.py`, `model_config.py` |
| **Training data exact content** | Reveals proprietary corpus | `full_corpus.txt` |
| **Exact permission guard patterns** | Reveals exact blacklist regex patterns | `permission_guard.py` `DANGEROUS_PATTERNS` |
| **Weakness detection heuristics** | Reveals self-analysis algorithms | `weakness_detector.py` |
| **Recovery policy strategies** | Reveals exact recovery logic | `recovery.py` |
| **Benchmark raw numbers** | Reveals performance characteristics not publicly available | `FINAL_BENCHMARK.json`, `audit_logs/` |

**When to disclose:**
- Only under specific contractual provisions (e.g., "source code escrow," "technology transfer agreement")
- Only to parties with legitimate need-to-know (e.g., integration engineers, security auditors)
- Only after legal review of disclosure scope and NDAs

---

## 6. Disclosure Strategy

### Phase 1: Initial Contact (Public)
- Share `00_EXECUTIVE_SUMMARY.md` (redacted)
- Provide high-level architecture diagram
- Share security posture summary
- Schedule technical overview call

### Phase 2: Technical Discussion (NDA)
- Sign mutual NDA
- Share full Technical Acquisition Package (except trade secrets)
- Provide architecture deep-dive
- Demonstrate capabilities via `13_DEMO_PROTOCOL.md`
- Discuss integration requirements

### Phase 3: Due Diligence (Due-Diligence Only)
- Execute `11_ACQUISITION_DUE_DILIGENCE_CHECKLIST.md`
- Provide complete source code, tests, benchmarks
- Facilitate security audit by external firm
- Provide dataset with provenance documentation
- Negotiate definitive terms

### Phase 4: Transaction (Trade Secret as Needed)
- Disclose trade secrets only under specific contractual provisions
- Provide source code escrow if required
- Facilitate technology transfer with legal oversight

---

## 7. Redaction Guidelines

When sharing documents:
- Remove specific regex patterns from `permission_guard.py`
- Remove exact AST deny lists from `python_runner.py`
- Remove exact rlimit values from `os_sandbox.py`
- Remove exact model hyperparameters from public architecture diagrams
- Summarize dataset size without revealing exact content
- Summarize benchmark results without revealing raw numbers

When sharing code:
- Provide obfuscated or summarized versions for public/NDA phases
- Provide complete source only under due-diligence agreements
- Never share model weights or training data without contractual protection

---

*This document is part of the Technical Acquisition Package.*
