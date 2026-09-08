# Acquisition Due Diligence Checklist

## 1. Code Quality

- [ ] **Run full test suite** — Execute all 1,031 tests in CI environment and verify 100% pass rate
- [ ] **Verify test counts** — Confirm actual test count matches documentation
- [ ] **Run static analysis** — Execute bandit, pylint, mypy on all source files
- [ ] **Measure code coverage** — Install pytest-cov and verify coverage percentages
- [ ] **Check for dead code** — Run vulture or similar tool to identify unused code
- [ ] **Verify no secrets in code** — Scan for API keys, passwords, tokens
- [ ] **Check dependency licenses** — Verify all dependencies are compatible with acquisition terms

## 2. Architecture

- [ ] **Verify AgentRuntime integration** — Confirm `jarvis.py` → `ExecutiveRouter` → `AgentRuntime` path works end-to-end
- [ ] **Map all dependencies** — Generate complete dependency graph of all modules
- [ ] **Identify circular dependencies** — Check for import cycles
- [ ] **Verify module boundaries** — Confirm cognitive, executive, neural, infrastructure layers are properly separated
- [ ] **Check for hidden coupling** — Identify modules that depend on implementation details of other modules
- [ ] **Document extension points** — Identify where new capabilities can be added without modifying core

## 3. Neural Core

- [ ] **Verify model training** — Reproduce training from scratch using `train.py`
- [ ] **Verify model loading** — Confirm `model.pkl` loads correctly and produces expected outputs
- [ ] **Verify gradient correctness** — Run `test_gradient_check.py` and review numerical tolerances
- [ ] **Verify GPU parity** — Run `test_gpu_parity.py` on target hardware
- [ ] **Measure inference latency** — Benchmark end-to-end inference on production hardware
- [ ] **Measure training throughput** — Benchmark full training run on production corpus
- [ ] **Verify checkpoint compatibility** — Test save/load round-trip across versions
- [ ] **Verify tokenizer correctness** — Run full tokenizer test suite
- [ ] **Verify dataset integrity** — Confirm 0 overlap, 0 duplicates, correct splits

## 4. Security

- [ ] **Re-run security audit** — Execute `security_audit.py` and `security_report_final.py` independently
- [ ] **Independent penetration test** — Engage external security firm for penetration testing
- [ ] **Fuzz security gates** — Fuzz `PermissionGuard`, `PythonRunner`, `SandboxEnvironment`
- [ ] **Verify sandbox escape tests** — Confirm all 9 sandbox escape tests pass
- [ ] **Verify layered security tests** — Confirm all 10 layered security tests pass
- [ ] **Test on target OS** — Verify OS sandbox works on target deployment OS
- [ ] **Review audit log integrity** — Verify logs cannot be tampered with
- [ ] **Check for hardcoded credentials** — Scan entire codebase

## 5. Memory and State

- [ ] **Verify memory persistence** — Test `UnifiedMemory` disk persistence across restarts
- [ ] **Verify LRU eviction** — Test `EpisodicMemory` eviction under memory pressure
- [ ] **Verify TTL expiration** — Test memory expiration after TTL
- [ ] **Verify memory search** — Test `search_semantic()` returns correct results
- [ ] **Verify state serialization** — Test `AgentState` round-trip through JSON/pickle
- [ ] **Verify checkpoint/resume** — Test full checkpoint/resume cycle

## 6. Tools and Execution

- [ ] **Verify filesystem tool** — Test file CRUD within sandbox boundaries
- [ ] **Verify terminal tool** — Test command execution with dangerous patterns blocked
- [ ] **Verify Python runner** — Test code execution with approval, AST gate, OS sandbox
- [ ] **Verify web tools** — Test `InternetTool` and `WebBrowserTool` with real network (if enabled)
- [ ] **Verify tool selection** — Test `ToolSelector` chooses correct tool for capability
- [ ] **Verify capability registry** — Test `CapabilityRegistry` tracks available tools

## 7. Integration

- [ ] **Verify E2E pipeline** — Run `cognitive_core/e2e_scenario.py` and inspect output
- [ ] **Verify CLI path** — Run `jarvis.py` interactively and test common commands
- [ ] **Verify neural model integration** — Test `ExecutiveRouter.generate()` with neural model
- [ ] **Verify memory integration** — Test memory persistence across requests
- [ ] **Verify security integration** — Test permission enforcement end-to-end

## 8. Data and Provenance

- [ ] **Verify dataset sources** — Document original sources of `full_corpus.txt`
- [ ] **Verify dataset licenses** — Confirm all data is properly licensed
- [ ] **Verify model provenance** — Confirm model weights were trained from scratch
- [ ] **Verify checkpoint lineage** — Confirm all checkpoints include complete lineage
- [ ] **Verify reproducibility** — Run training twice and confirm bitwise identical results

## 9. Documentation

- [ ] **Update test counts** — Correct all documentation to reflect 1,031 tests
- [ ] **Update integration status** — Correct documentation to reflect AgentRuntime is integrated
- [ ] **Document dataset overlap** — Explain and resolve dataset overlap warnings
- [ ] **Document architecture** — Create up-to-date architecture diagrams
- [ ] **Document deployment** — Create deployment guide with hardware/software requirements
- [ ] **Document API** — Create API reference for all public interfaces
- [ ] **Document security model** — Create security architecture document

## 10. Legal and IP

- [ ] **Add LICENSE file** — Determine and add appropriate license
- [ ] **File invention disclosures** — Document Class A and B IP for patent consideration
- [ ] **Freedom-to-operate analysis** — Engage patent counsel for transformer and sandbox patents
- [ ] **Verify contributor agreements** — Confirm all contributors assigned IP rights
- [ ] **Verify third-party licenses** — Confirm all dependencies have compatible licenses
- [ ] **Create data room index** — Organize all documents, code, tests, benchmarks for due diligence

---

*This document is part of the Technical Acquisition Package.*
