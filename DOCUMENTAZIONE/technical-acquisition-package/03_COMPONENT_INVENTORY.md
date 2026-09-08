# Component Inventory

## 1. Neural Core (`neural_core/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| MicroTransformer | `network.py` | 785 | IMPLEMENTED | Production-ready | NumPy only |
| TorchMicroTransformer | `torch_model.py` | 381 | IMPLEMENTED | Testing | PyTorch (optional) |
| Tokenizer | `tokenizer.py` | 395 | IMPLEMENTED | Production-ready | None |
| Trainer | `trainer.py` | 948 | IMPLEMENTED | Production-ready | NumPy |
| NextTokenModel | `trainer.py` | ~200 | IMPLEMENTED | Production-ready | NumPy |
| Loss (CrossEntropy) | `loss.py` | ~100 | IMPLEMENTED | Production-ready | NumPy |
| Optimizer (Adam) | `optimizer.py` | ~150 | IMPLEMENTED | Production-ready | NumPy |
| Dataset | `dataset.py` | ~400 | IMPLEMENTED | Production-ready | None |
| Model Interface | `model_interface.py` | 362 | IMPLEMENTED | Production-ready | NumPy, tokenizer |
| Embeddings | `embeddings.py` | ~100 | IMPLEMENTED | Production-ready | NumPy |
| Benchmarks | `benchmarks.py`, `final_benchmark.py`, `gpu_benchmark.py`, `gpu_benchmark_extended.py` | ~500 | IMPLEMENTED | Experimental | NumPy, PyTorch |
| OS Sandbox | `os_sandbox.py` | 510 | IMPLEMENTED | Production-ready | macOS/Linux OS primitives |
| OS Isolation Probe | `os_isolation_probe.py` | 229 | IMPLEMENTED | Production-ready | OS primitives |
| External LLM Provider | `providers/external_llm_provider.py` | 253 | IMPLEMENTED (DISABLED) | N/A | requests (optional) |
| Security Audit | `security_audit.py`, `security_report_final.py` | ~400 | IMPLEMENTED | Audit-only | AST, regex |
| Test Quality Audit | `test_quality_audit.py` | ~200 | IMPLEMENTED | Audit-only | AST |
| Dataset Audit | `dataset_audit.py`, `dataset_audit_extended.py` | ~300 | IMPLEMENTED | Audit-only | None |
| Provenance Pipeline | `provenance_pipeline.py` | ~150 | IMPLEMENTED | Audit-only | None |
| Reproducibility | `reproducibility.py` | ~100 | IMPLEMENTED | Experimental | None |
| Checkpoint | `checkpoint.py`, `checkpoint_lineage.py`, `checkpoint_metadata.py` | ~300 | IMPLEMENTED | Production-ready | pickle, JSON |
| Baseline Integrity | `baseline_integrity.py` | ~100 | IMPLEMENTED | Experimental | None |

---

## 2. Cognitive Core (`cognitive_core/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| Coordinator | `coordinator.py` | 527 | IMPLEMENTED | Production-ready | All cognitive subsystems |
| AgentRuntime | `runtime/agent_runtime.py` | 815 | IMPLEMENTED | Production-ready | CognitiveCoordinator, ActionExecutor |
| TaskLifecycle | `runtime/task_lifecycle.py` | ~200 | IMPLEMENTED | Production-ready | None |
| AgentState | `runtime/agent_state.py` | ~200 | IMPLEMENTED | Production-ready | None |
| Recovery | `runtime/recovery.py` | ~150 | IMPLEMENTED | Experimental | None |
| Action | `runtime/action.py` | ~150 | IMPLEMENTED | Production-ready | None |
| Verification | `runtime/verification.py` | ~100 | IMPLEMENTED | Production-ready | None |
| Understanding | `understanding.py` | 227 | IMPLEMENTED | Production-ready | None (rule-based) |
| Reasoner | `reasoning/reasoner.py` | 374 | IMPLEMENTED | Production-ready | Decomposer, Hypothesis |
| Decomposer | `reasoning/decomposer.py` | ~100 | IMPLEMENTED | Production-ready | None |
| Hypothesis | `reasoning/hypothesis.py` | ~100 | IMPLEMENTED | Production-ready | None |
| Planner | `planner.py` | 281 | IMPLEMENTED | Production-ready | Reasoner, Memory |
| WorldModel | `world_model/world_model.py` | 183 | IMPLEMENTED | Production-ready | None |
| Simulator | `world_model/simulator.py` | ~100 | IMPLEMENTED | Production-ready | WorldModel |
| WorkingMemory | `memory/working_memory.py` | 17 | IMPLEMENTED | Production-ready | None |
| EpisodicMemory | `memory/episodic_memory.py` | ~100 | IMPLEMENTED | Production-ready | None |
| SemanticMemory | `memory/semantic_memory.py` | 13 | IMPLEMENTED | Production-ready | None |
| ProceduralMemory | `memory/procedural_memory.py` | 11 | IMPLEMENTED | Production-ready | None |
| UnifiedMemory | `memory/unified_memory.py` | 152 | IMPLEMENTED | Production-ready | All memory types |
| MemoryManager | `memory/memory_manager.py` | ~100 | IMPLEMENTED | Production-ready | Episodic, disk |
| ContextManager | `context_manager.py` | ~50 | IMPLEMENTED | Production-ready | None |
| KnowledgeGraph | `knowledge/knowledge_graph.py` | ~150 | IMPLEMENTED | Production-ready | None |
| VectorMemory | `knowledge/vector_memory.py` | ~120 | IMPLEMENTED | Production-ready | NumPy |
| VisionSystem | `vision/vision_system.py` | ~100 | IMPLEMENTED (DISABLED) | N/A | None |
| E2EScenario | `e2e_scenario.py` | 227 | IMPLEMENTED | Test-only | All cognitive subsystems |
| IdentityResolver | `identity/identity_resolver.py` | ~50 | IMPLEMENTED | Production-ready | None |

---

## 3. Executive System (`executive_system/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| Router | `router.py` | 323 | IMPLEMENTED | Production-ready | All executive subsystems |
| ActionExecutor | `action_executor.py` | 66 | IMPLEMENTED | Production-ready | PermissionGuard, AuditSystem |
| CapabilityRegistry | `capability_registry.py` | ~100 | IMPLEMENTED | Production-ready | None |
| ToolSelector | `tool_selector.py` | ~100 | IMPLEMENTED | Partial | CapabilityRegistry |
| AutonomyManager | `autonomy_manager.py` | ~150 | IMPLEMENTED | Production-ready | PermissionGuard |
| GoalManager | `goal_manager.py` | ~200 | IMPLEMENTED | Production-ready | None |
| ObservationFeedback | `observation_feedback.py` | ~100 | IMPLEMENTED | Production-ready | None |
| ToolSystem | `tools/tool_system.py` | ~150 | IMPLEMENTED | Production-ready | SecurityLayer |
| FilesystemTool | `tools/filesystem_tool.py` | ~100 | IMPLEMENTED | Production-ready | SandboxEnvironment |
| TerminalTool | `tools/terminal_tool.py` | ~150 | IMPLEMENTED | Production-ready | PermissionGuard |
| PythonRunner | `tools/python_runner.py` | 187 | IMPLEMENTED | Production-ready | AST, OS sandbox |
| WebBrowserTool | `tools/web_browser_tool.py` | ~100 | IMPLEMENTED | Production-ready | BeautifulSoup |
| InternetTool | `tools/internet_tool.py` | ~100 | IMPLEMENTED | Production-ready | requests, NetworkPolicy |

---

## 4. Expansion System (`expansion_system/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| DiscoveryEngine | `discovery/discovery_engine.py` | 161 | IMPLEMENTED | Experimental | None |
| Experimenter | `discovery/experimenter.py` | ~100 | IMPLEMENTED | Experimental | None |
| WeaknessDetector | `self_analysis/weakness_detector.py` | 157 | IMPLEMENTED | Experimental | None |
| PerformanceMonitor | `self_analysis/performance_monitor.py` | ~100 | IMPLEMENTED | Experimental | None |
| ImprovementCycle | `self_improvement/improvement_cycle.py` | 780 | IMPLEMENTED (DISABLED) | Experimental | All expansion subsystems |
| CandidateBuilder | `self_improvement/candidate_builder.py` | ~100 | IMPLEMENTED | Experimental | None |
| BenchmarkComparator | `self_improvement/benchmark_comparator.py` | ~100 | IMPLEMENTED | Experimental | None |
| SystemBenchmark | `self_improvement/system_benchmark.py` | ~100 | IMPLEMENTED | Experimental | None |
| TrainingStrategy | `self_improvement/training_strategy.py` | ~100 | IMPLEMENTED | Experimental | None |
| ExperimentalTrainer | `self_improvement/experimental_trainer.py` | ~150 | IMPLEMENTED | Experimental | None |
| ToolFactory | `tool_factory/factory.py` | ~150 | IMPLEMENTED | Experimental | None |
| Installer | `tool_factory/installer.py` | ~100 | IMPLEMENTED | Experimental | None |

---

## 5. Infrastructure (`INFRASTRUCTURE/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| PermissionGuard | `security/permission_guard.py` | ~300 | IMPLEMENTED | Production-ready | None |
| SecurityLayer | `security/security_layer.py` | ~150 | IMPLEMENTED | Production-ready | PermissionGuard, AuditSystem |
| AuditSystem | `audit/audit_system.py` | ~400 | IMPLEMENTED | Production-ready | None |
| NetworkPolicy | `network_policy.py` | ~150 | IMPLEMENTED | Production-ready | None |
| SandboxEnvironment | `sandbox/sandbox_env.py` | ~150 | IMPLEMENTED | Production-ready | None |
| Logging | `logging.py` | ~100 | IMPLEMENTED | Production-ready | Python logging |

---

## 6. Interface Layer (`interface_layer/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| VoiceInterface | `voice_interface.py` | ~100 | IMPLEMENTED | Experimental | ExecutiveRouter |

---

## 7. CORE System (`CORE/`)

| Component | File | LOC | Status | Maturity | Dependencies |
|---|---|---|---|---|---|
| Identity | `system/identity.py` | ~100 | IMPLEMENTED | Production-ready | None |
| Configuration | `config/configurazione.py` | ~100 | IMPLEMENTED | Production-ready | PyYAML |

---

## 8. Test Suite (`TESTS/`)

| Category | Files | Tests (collected) | Status |
|---|---|---|---|
| Neural Core | `test_neural_core/*.py` | 356 | PASS |
| Security | `test_security.py`, `test_layered_security.py`, `test_sandbox_escape.py` | 42 | PASS |
| OS Sandbox | `test_os_sandbox.py` | 9 | PASS |
| Agent Runtime | `test_agent_runtime.py`, `test_runtime_integration.py` | 45 | PASS |
| E2E Integration | `test_e2e_jarvis.py`, `test_cognitive_core_integration.py`, `test_deep_integration_audit.py`, `test_e2e_integration_audit.py` | ~20 | PASS |
| Memory | `test_memory_cycle.py`, `test_memory_manager.py`, `test_memory_knowledge.py`, `test_vector_memory.py` | ~40 | PASS |
| Reasoning | `test_reasoning.py`, `test_reasoning_extended.py` | ~20 | PASS |
| World Model | `test_world_model.py`, `test_world_model_extended.py` | ~10 | PASS |
| Tools | `test_tools.py`, `test_tool_system_integration.py`, `test_terminal_tool.py`, `test_filesystem_tool.py`, `test_python_runner.py`, `test_internet_tool.py` | ~30 | PASS |
| Self-Improvement | `test_self_improvement.py`, `test_self_improvement_comprehensive.py`, `test_self_improvement_disabled.py` | ~15 | PASS |
| Device Control | `test_device_control_disabled.py` | 5 | PASS |
| Discovery | `test_discovery.py`, `test_discovery_extended.py` | ~10 | PASS |
| Benchmarks | `test_benchmarks.py` | 5 | PASS |
| Reproducibility | `test_reproducibility.py` | 1 | PASS |
| Config | `test_config/test_configurazione.py` | 3 | PASS |
| Logging | `test_logging/test_logging.py` | 1 | PASS |

**Total collected: 1,031** (documentation claims 688-863 — stale)

---

*This document is part of the Technical Acquisition Package.*
