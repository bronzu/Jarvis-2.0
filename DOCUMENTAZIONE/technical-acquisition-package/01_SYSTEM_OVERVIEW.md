# System Overview

## 1. High-Level Description

JARVIS 2.0 is a **proprietary AI agent system** implemented in pure Python with minimal external dependencies (`numpy`, `requests`, `beautifulsoup4`, `pyyaml`, `pytest`). The system is designed around a **cognitive architecture** that processes user input through multiple reasoning, memory, and execution layers before producing an action or response.

### Core Principle
The system is **model-agnostic in architecture** but currently runs on a proprietary `MicroTransformer` (2.7M parameters, NumPy-based) trained on an internal corpus. It has **no runtime dependency on external LLM APIs**.

### Execution Model
```
User Input
    ↓
ExecutiveRouter (entry point)
    ↓
AgentRuntime (task loop orchestrator)
    ↓
CognitiveCoordinator (cognitive pipeline)
    ↓
[Understanding → Reasoning → Planning → Memory → Knowledge → Tools → World Model]
    ↓
Action Execution (with permission guard + sandbox)
    ↓
Response
```

---

## 2. System Boundaries

### Included in this repository
- Neural core (model, tokenizer, trainer, benchmarks)
- Cognitive core (coordinator, runtime, memory, reasoning, world model)
- Executive system (router, tools, goal management, autonomy)
- Expansion system (discovery, self-analysis, self-improvement — disabled)
- Infrastructure (security, audit, logging, network policy, sandbox)
- Interface layer (voice, CLI)
- Tests (1,031 tests)
- Documentation (Italian and English)

### NOT included
- Large-scale pretrained language model
- External API credentials
- Production deployment configuration (Docker, K8s, etc.)
- CI/CD pipelines
- Monitoring/observability at scale
- Distributed execution framework

---

## 3. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.x | No type hints enforced at runtime |
| Numerical | NumPy | Pure NumPy for CPU model |
| GPU | PyTorch (MPS) | Optional, for parity testing only |
| Configuration | YAML | Single `config.yaml` |
| Persistence | JSON, pickle, SQLite | `state.db`, `unified_memory_db.json`, `model.pkl` |
| Testing | pytest | 1,031 tests |
| Security | Application-level + OS-level | PermissionGuard + sandbox-exec/chroot |
| Network | requests, BeautifulSoup | Gated by NetworkPolicy |
| Logging | Python logging + custom AuditSystem | Structured events with secret redaction |

---

## 4. Key Characteristics

### Strengths
- **Self-contained**: No external API dependencies for core operation
- **Security-focused**: Real OS-level sandboxing, AST-based code validation, comprehensive permission model
- **Architecturally complete**: Full cognitive loop with task lifecycle, state management, recovery
- **Tested**: 1,031 tests covering neural core, security, memory, reasoning, tools, runtime
- **Reproducible**: Deterministic training, serialized tokenizer, checkpoint with lineage

### Limitations
- **Small model**: 2.7M parameters, trained on 33 MB corpus
- **Limited language capability**: No large-scale conversational training
- **No production deployment**: Single-process, no orchestration
- **Stale documentation**: Test counts and integration claims are outdated
- **Limited performance data**: Benchmarks exist but are small-scale

---

## 5. Use Cases (Demonstrated)

| Use Case | Status | Evidence |
|---|---|---|
| Command-line assistant | **DEMONSTRATED** | `jarvis.py` interactive CLI |
| File operations in sandbox | **DEMONSTRATED** | `test_e2e_jarvis.py::test_filesystem_tool_execution_and_verification` |
| Safe code execution | **DEMONSTRATED** | `PythonRunner` with AST gate + OS sandbox |
| Permission enforcement | **DEMONSTRATED** | 44+ security tests, permission guard blocks dangerous commands |
| Memory persistence | **DEMONSTRATED** | `UnifiedMemory` with disk persistence |
| Identity recognition | **DEMONSTRATED** | `test_identity_question_routed_to_identity_system` |
| Reasoning (rule-based) | **DEMONSTRATED** | Reasoner with deductive/inductive/classification |
| Planning (basic) | **DEMONSTRATED** | Planner with goal decomposition |
| World modeling | **DEMONSTRATED** | Entity tracking, causal relations, simulation |

---

## 6. Use Cases (Not Demonstrated)

| Use Case | Status | Reason |
|---|---|---|
| Large-scale conversation | **NOT DEMONSTRATED** | Model trained on small corpus |
| Multi-user/multi-session | **NOT DEMONSTRATED** | Single-process, no session isolation beyond task_id |
| Distributed execution | **NOT IMPLEMENTED** | No network transparency layer |
| Real-time voice interaction | **DISABLED** | Voice interface exists but not integrated |
| Vision/screenshot/OCR | **DISABLED** | Config flags set to false |
| Self-modification | **DISABLED** | Self-improvement requires explicit approval |
| Long-horizon autonomy | **NOT DEMONSTRATED** | Runtime loop exists but limited iteration testing |

---

*This document is part of the Technical Acquisition Package.*
