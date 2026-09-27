# ORGAN AUDIT — GROUND TRUTH

> **Source**: `SPECIFICA_PROGETTO.md` §28 (linee 820-854) — "La mappa completa degli organi"
> **Method**: ogni organo è stato verificato leggendo il **codice reale** (implementazione + test), NON documentazione pregressa.
> **Note metodologiche**:
> - Il documento non affida stato a nessuna rapresentazione sintetica. Ogni riga è supportata da file di implementazione e da test effettivi.
> - "NON IMPLEMENTATO" significa che **nessun file Python** nel progetto contiene una classe, funzione o modulo dedicato a quell'organo, e nessun test lo esercita.
> - "REAL" significa che l'implementazione esiste, contiene logica reale (nessuno stub che non fa niente), e ha test che ne verificano il comportamento.

---

## 1. Cognitive Core

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/coordinator.py` — classe `CognitiveCoordinator` |
| **Descrizione** | Coordinatore principale. Orchestra Understanding, Reasoning, Memory, Neural Core, Tool execution, Identity resolution, Vision. Modalità legacy (`neural_model`) e full (`neural_interface` + componenti). Thread-safe con `_model_lock` (RLock). |
| **File principali** | `cognitive_core/coordinator.py` |
| **Test** | `TESTS/test_cognitive_core_integration.py`, `TESTS/test_agent_runtime.py`, `TESTS/test_priority1_implementations.py`, `TESTS/test_e2e_integration_audit.py`, `TESTS/test_deep_integration_audit.py` |
| **Stato** | REAL |

---

## 2. Artificial Intelligence / Neural Core

| Campo | Valore |
|-------|--------|
| **Implementazione** | `neural_core/` — architettura numpy-only (nessun dipendenza esterna a PyTorch/Torch) |
| **Descrizione** | MicroTransformer (attention multi-head, positional encoding, causal masking, residual, LayerNorm, FeedForward), Neuron (simple NN con SGD), Tokenizer (modalità whitespace + technical con pre-tokenization italiana), Trainer (gradient accumulation, LR scheduler con warmup, early stopping, best model checkpoint, resume), CrossEntropyLoss/MSELoss, Adam/SGD, Dataset, Checkpoint, ModelEvaluator, NeuralModelInterface, build_dataset.py, provenance_pipeline.py, leakage_audit, minhash_lsh. |
| **File principali** | `neural_core/model.py`, `neural_core/network.py`, `neural_core/tokenizer.py`, `neural_core/trainer.py`, `neural_core/loss.py`, `neural_core/optimizer.py`, `neural_core/dataset.py`, `neural_core/checkpoint.py`, `neural_core/evaluator.py`, `neural_core/model_interface.py`, `neural_core/build_dataset.py`, `neural_core/provenance_pipeline.py`, `neural_core/leakage_audit.py`, `neural_core/minhash_lsh.py`, `neural_core/dataset_audit.py` |
| **Test** | `TESTS/test_neural_core/` (12 file: test_training.py, test_trainer.py, test_model_interface.py, test_network.py, test_tokenizer.py, test_evaluator.py, test_loss.py, test_optimizer.py, test_dataset.py, test_checkpoint.py, test_hardening.py, test_neural_network.py, etc.), `TESTS/test_dataset_audit.py`, `TESTS/test_checkpoint_corruption.py`, `TESTS/test_reproducibility.py`, `TESTS/test_final_gap_closure.py` |
| **Stato** | REAL — Internal-only, nessun provider esterno nel flusso decisionale |

---

## 3. Perception

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/perception/perception.py` — classe `PerceptionLayer` e `PerceptEvent` (dataclass). Nuovo organo implementato in fase 0.1. |
| **Descrizione** | Il Perception Layer è l'organo che riceve l'input grezzo (testo, immagine, dati strutturati) e lo normalizza in un `PerceptEvent` prima che raggiunga l'Understanding. Responsabilità: normalizzazione del testo (strip, whitespace collapse, lower per matching), rilevamento del tipo di input (command/question/information/empty/vision), estrazione di metadati (fonte, timestamp, tipo, confidence), caricamento di immagini tramite `ImageAsset`/`load_image` per input visionivi. Il `CognitiveCoordinator.process()` e `VoiceInterface.speak()` instradano l'input attraverso `PerceptionLayer` prima di passare all'Understanding o all'ExecutiveRouter. |
| **File principali** | `cognitive_core/perception/__init__.py`, `cognitive_core/perception/perception.py` |
| **Test** | `TESTS/test_perception.py` (PerceptEvent: to_dict con/ senza image_asset, fields di base; PerceptionLayer: process_text question/command/information, process None error, process_image FileNotFoundError, process vision dict no path, process bytes, whitespace normalization) |
| **Stato** | REAL |

---

## 4. Reasoning

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/reasoning/` |
| **Descrizione** | Motore di ragionamento deduttivo. Reasoner risolve problemi con fatti, regole, goal: genera dimostrazioni multi-step. Decomposer normalizza problemi. HypothesisGenerator crea candidati dalle regole. |
| **File principali** | `cognitive_core/reasoning/reasoner.py`, `cognitive_core/reasoning/decomposer.py`, `cognitive_core/reasoning/hypothesis.py` |
| **Test** | `TESTS/test_reasoning.py`, `TESTS/test_reasoning_extended.py` |
| **Stato** | REAL |

---

## 5. Planning

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/planner.py` |
| **Descrizione** | Planner crea piani con PlanStep, Precondition, Effect. Strategie di fallback, dipendenze tra step, validazione (nessun ciclo), serializzazione, esecuzione. |
| **File principali** | `cognitive_core/planner.py` |
| **Test** | `TESTS/test_planner.py` |
| **Stato** | REAL |

---

## 6. Memory

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/memory/` |
| **Descrizione** | MemoryManager (LRU eviction, TTL, persistenza JSON, eviction count, search_semantic), UnifiedMemory (coordina working/episodic/semantic/procedural), WorkingMemory (finestra scorrevole con capacity), EpisodicMemory (record/recall con TTL e eviction), SemanticMemory (learn/recall con prevenzione duplicati), ProceduralMemory (register/retrieve). |
| **File principali** | `cognitive_core/memory/memory_manager.py`, `cognitive_core/memory/unified_memory.py`, `cognitive_core/memory/working_memory.py`, `cognitive_core/memory/episodic_memory.py`, `cognitive_core/memory/semantic_memory.py`, `cognitive_core/memory/procedural_memory.py` |
| **Test** | `TESTS/test_memory_manager.py`, `TESTS/test_memory_cycle.py`, `TESTS/test_memory_knowledge.py`, `TESTS/test_priority1_implementations.py` (LRU+TTL), `TESTS/test_vector_memory.py` |
| **Stato** | REAL |

---

## 7. Knowledge

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/knowledge/` |
| **Descrizione** | KnowledgeGraph (assert_fact con confidence + source + version, relate con metadata, detect_contradictions, verify con stati: unknown/contradicted/verified, get_fact, confidence). VectorMemory (add/search con cosine similarity). |
| **File principali** | `cognitive_core/knowledge/knowledge_graph.py`, `cognitive_core/knowledge/vector_memory.py` |
| **Test** | `TESTS/test_knowledge_extensions.py`, `TESTS/test_memory_knowledge.py`, `TESTS/test_vector_memory.py` |
| **Stato** | REAL |

---

## 8. World Model

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/world_model/` |
| **Descrizione** | WorldModel (add_causal_relation, predict, simulate_state_transition) + Simulator (simulate causal chains, simulate_entity_property). Mantiene relazioni causali e aggiorna in base agli esiti delle azioni. |
| **File principali** | `cognitive_core/world_model/world_model.py`, `cognitive_core/world_model/simulator.py` |
| **Test** | `TESTS/test_world_model.py`, `TESTS/test_world_model_extended.py`, `TESTS/test_final_gap_closure.py` (e2e scenario) |
| **Stato** | REAL |

---

## 9. Imagination

| Campo | Valore |
|-------|--------|
| **Implementazione** | **NESSUNA** |
| **Descrizione** | Spec: "Imagination" — nessun file Python o modulo dedicato. Ricerca grep per "imaginat" restituisce zero risultati. La capacità di simulazione è coperta da WorldModel + Simulator (organo 8 e 22), non da un organo dedicato all'immaginazione. |
| **File principali** | — |
| **Test** | — |
| **Stato** | **NON IMPLEMENTATO** |

---

## 10. Discovery

| Campo | Valore |
|-------|--------|
| **Implementazione** | `expansion_system/discovery/` |
| **Descrizione** | DiscoveryEngine (investigate con ipotesi, predict, run_experiment, observe, evaluate con score 0-1, conclude con knowledge). Experimenter (run con outcome success/error/timeout, resource_usage, elapsed_seconds). |
| **File principali** | `expansion_system/discovery/discovery_engine.py`, `expansion_system/discovery/experimenter.py` |
| **Test** | `TESTS/test_discovery.py`, `TESTS/test_discovery_extended.py` |
| **Stato** | REAL |

---

## 11. Learning

| Campo | Valore |
|-------|--------|
| **Implementazione** | `neural_core/trainer.py` (Trainer), `train.py` (entry point) |
| **Descrizione** | Trainer implementa il ciclo di addestramento: train_step, train con epoch/batch_size, gradient accumulation, gradient clipping, LR scheduler con warmup + cosine/linear/constant decay, early stopping, best model checkpoint, resume da checkpoint, evaluation. `train.py` fornisce `build_training_setup` e `train_and_save` (salva tokenizer, model, best model, last training state). |
| **File principali** | `neural_core/trainer.py`, `train.py`, `neural_core/build_dataset.py` |
| **Test** | `TESTS/test_neural_core/test_training.py`, `TESTS/test_neural_core/test_trainer.py`, `TESTS/test_neural_core/test_model_interface.py`, `TESTS/test_training_integration.py`, `TESTS/test_kaggle_model_integration.py` |
| **Stato** | REAL |

---

## 12. Self-Improvement

| Campo | Valore |
|-------|--------|
| **Implementazione** | `expansion_system/self_improvement/` |
| **Descrizione** | ImprovementCycle (ciclo completo: osservazione → rilevazione debolezze → generazione candidati → benchmark → comparazione → approvazione/rifiuto/rollback), CandidateBuilder (state machine: sandbox → testing → approved → deployed → rolled_back), BenchmarkComparator (compare_versions con ACCEPT/REJECT, critical_regression, weighted_delta), SystemBenchmark (run_all con logical_accuracy, execution_time, memory_usage, json_adherence, tool_success, safety), TrainingStrategy, ExperimentalTrainer. Tutto richiede approvazione utente. |
| **File principali** | `expansion_system/self_improvement/improvement_cycle.py`, `expansion_system/self_improvement/candidate_builder.py`, `expansion_system/self_improvement/benchmark_comparator.py`, `expansion_system/self_improvement/system_benchmark.py`, `expansion_system/self_improvement/training_strategy.py`, `expansion_system/self_improvement/experimental_trainer.py` |
| **Test** | `TESTS/test_self_improvement.py`, `TESTS/test_self_improvement_comprehensive.py`, `TESTS/test_self_improvement_disabled.py` |
| **Stato** | REAL |

---

## 13. Tools

| Campo | Valore |
|-------|--------|
| **Implementazione** | `executive_system/tools/` |
| **Descrizione** | ToolSystem (registra e delega a FilesystemTool, TerminalTool, PythonRunner, WebTool, InternetTool). PythonRunner applica 5 layer di sicurezza: 1) PermissionGuard, 2) SecurityLayer DANGEROUS_PATTERNS, 3) AST gate (blocca import, __class__, open, os.system), 4) literal-pattern re-scan, 5) OS-level sandbox (sandbox-exec/rlimits). TerminalTool: path traversal, symlink escape, banned comandi. FilesystemTool: sandbox path confinement. |
| **File principali** | `executive_system/tools/tool_system.py`, `executive_system/tools/filesystem_tool.py`, `executive_system/tools/terminal_tool.py`, `executive_system/tools/python_runner.py`, `executive_system/tools/web_tool.py`, `executive_system/tools/internet_tool.py` |
| **Test** | `TESTS/test_tools.py`, `TESTS/test_terminal_tool.py`, `TESTS/test_tool_system_integration.py`, `TESTS/test_internet_tool.py`, `TESTS/test_layered_security.py`, `TESTS/test_security.py` (TestTerminalToolSecurity) |
| **Stato** | REAL |

---

## 14. Tool Factory

| Campo | Valore |
|-------|--------|
| **Implementazione** | `expansion_system/tool_factory/` |
| **Descrizione** | ToolFactory genera codice Python da specifiche (name, description, methods con implementation). validate_generated verifica sicurezza del codice generato. Installer promuove tool da stato "sandbox" a "installed" (richiede approvazione). |
| **File principali** | `expansion_system/tool_factory/factory.py`, `expansion_system/tool_factory/installer.py` |
| **Test** | `TESTS/test_tool_factory.py` |
| **Stato** | REAL |

---

## 15. Action

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/runtime/action.py`, `executive_system/action_executor.py` |
| **Descrizione** | ActionAuthorizer (schema strutturato con ActionStatus, ActionRisk, preconditions, provenance; regole: nessun self-authorization per capability pericolose, approval gate, delega a PermissionGuard/SecurityLayer). ActionExecutor (esecuzione con approvazione + audit logging). |
| **File principali** | `cognitive_core/runtime/action.py`, `executive_system/action_executor.py` |
| **Test** | `TESTS/test_phase2_action_observation_verification.py`, `TESTS/test_executive_system.py` |
| **Stato** | REAL |

---

## 16. Environment

| Campo | Valore |
|-------|--------|
| **Implementazione** | `INFRASTRUCTURE/sandbox/sandbox_env.py`, `INFRASTRUCTURE/network_policy.py` |
| **Descrizione** | SandboxEnvironment (path traversal prevention, symlink escape prevention, approvazione per scrittura, nested directory creation). NetworkPolicy (allow/block hosts, ports, protocol HTTPS/HTTP, redirect, timeout, allowlist). |
| **File principali** | `INFRASTRUCTURE/sandbox/sandbox_env.py`, `INFRASTRUCTURE/network_policy.py` |
| **Test** | `TESTS/test_security.py` (TestSandboxEnvironment), `TESTS/test_network_policy.py`, `TESTS/test_deep_integration_audit.py` (Test 6) |
| **Stato** | REAL |

---

## 17. Communication

| Campo | Valore |
|-------|--------|
| **Implementazione** | `interface_layer/voice_interface.py` |
| **Descrizione** | VoiceInterface: speak() (routing per intent: identity/local/web_tool), say() (TTS backend opzionale), listen() (fallback a input()), handle_prompt() (comunica solo con ExecutiveRouter, mai con LLM esterni). |
| **File principali** | `interface_layer/voice_interface.py` |
| **Test** | `TESTS/test_voice_interface.py`, `TESTS/test_runtime_integration.py` (test_voice_interface_routes_to_executive_system_and_web_search), `TESTS/test_router_tts_integration.py` |
| **Stato** | REAL |

---

## 18. User Model

| Campo | Valore |
|-------|--------|
| **Implementazione** | **NESSUNA** |
| **Descrizione** | Spec: "User Model" — nessun file Python o modulo dedicato. Ricerca grep per "user_model", "UserModel", "user_profile", "UserProfile" restituisce zero risultati. Non esiste alcun componente che tracci preferenze utente, pattern di comportamento, profiloutente o adattamento alle abitudini dell'utente. L'IdentityResolver (organo gestito da CORE/system/identity.py) risolve domande su JARVIS, non sul comportamento/previsione dell'utente. |
| **File principali** | — |
| **Test** | — |
| **Stato** | **NON IMPLEMENTATO** |

---

## 19. Security

| Campo | Valore |
|-------|--------|
| **Implementazione** | `INFRASTRUCTURE/security/security_layer.py`, `INFRASTRUCTURE/security/permission_guard.py` |
| **Descrizione** | SecurityLayer: check_tool_execution, check_filesystem_access, check_code_execution (DANGEROUS_PATTERNS), check_network_access, log_permission_event. PermissionGuard: DANGEROUS_PATTERNS regex, BLOCKED_COMMANDS, require_user_approval, grant_self_permission (sempre rigetta), validate_command. SecurityLayer + PermissionGuard sono usati dallo stesso SecurityLayer che valida comandi e codice. |
| **File principali** | `INFRASTRUCTURE/security/security_layer.py`, `INFRASTRUCTURE/security/permission_guard.py` |
| **Test** | `TESTS/test_security.py`, `TESTS/test_layered_security.py`, `TESTS/test_autonomy_guardrails.py`, `TESTS/test_phase2_action_observation_verification.py` (TestSecurityFixes), `TESTS/test_final_gap_closure.py` (security_audit, security_report_final) |
| **Stato** | REAL |

---

## 20. Permissions

| Campo | Valore |
|-------|--------|
| **Implementazione** | `INFRASTRUCTURE/security/permission_guard.py` |
| **Descrizione** | PermissionGuard: approvazione utente obbligatoria (approved must be True), self-authorization prevention (grant_self_permission rigetta sempre), validazione comando con DANGEROUS_PATTERNS e BLOCKED_COMMANDS, gestione quote/path traversal. È il punto di ingresso per tutti i controlli di permesso. |
| **File principali** | `INFRASTRUCTURE/security/permission_guard.py` |
| **Test** | `TESTS/test_security.py` (TestPermissionGuard — 20+ test cases: approvazione, empty command, non-string, os.system, subprocess, __import__, eval/exec, open write, shutil.rmtree, os.remove, blocked commands, safe commands, shell metacharacters, redirection, chaining, pipe), `TESTS/test_autonomy_guardrails.py` |
| **Stato** | REAL |

---

## 21. Autonomy

| Campo | Valore |
|-------|--------|
| **Implementazione** | `executive_system/autonomy_manager.py` |
| **Descrizione** | AutonomyManager: act() richiede approvazione esplicita. Non può auto-autorizzare. Delega a PermissionGuard. |
| **File principali** | `executive_system/autonomy_manager.py` |
| **Test** | `TESTS/test_autonomy_guardrails.py` |
| **Stato** | REAL |

---

## 22. Simulation

| Campo | Valore |
|-------|--------|
| **Implementazione** | `cognitive_core/world_model/simulator.py`, `cognitive_core/world_model/world_model.py` |
| **Descrizione** | Simulator: simulate(initial_state, steps) esplora catene causali. simulate_entity_property per simulazioni di transizione stato. WorldModel.simulate() metodo integrato. |
| **File principali** | `cognitive_core/world_model/simulator.py`, `cognitive_core/world_model/world_model.py` |
| **Test** | `TESTS/test_world_model.py`, `TESTS/test_world_model_extended.py`, `TESTS/test_final_gap_closure.py` (test_e2e_scenario_world_model_differs_by_outcome) |
| **Stato** | REAL |

---

## 23. Evaluation

| Campo | Valore |
|-------|--------|
| **Implementazione** | `neural_core/evaluator.py`, `neural_core/final_benchmark.py`, `expansion_system/self_improvement/system_benchmark.py` |
| **Descrizione** | ModelEvaluator (MAE — Mean Absolute Error). FinalBenchmark (hardware, startup, inference, training, checkpoint, tokenizer metrics). SystemBenchmark (run_all con 6 categorie: instruction_following, reasoning, memory, tool_selection, safety, latency; BenchmarkDataset, FailureClassifier con mapping a weakness types). |
| **File principali** | `neural_core/evaluator.py`, `neural_core/final_benchmark.py`, `expansion_system/self_improvement/system_benchmark.py` |
| **Test** | `TESTS/test_neural_core/test_evaluator.py`, `TESTS/test_final_gap_closure.py` (final_benchmark, checkpoint_lineage), `TESTS/test_self_improvement_comprehensive.py` (TestSystemBenchmark, TestBenchmarkDataset, TestFailureClassifier) |
| **Stato** | REAL |

---

## 24. Interface

| Campo | Valore |
|-------|--------|
| **Implementazione** | `interface_layer/voice_interface.py`, `executive_system/router.py` |
| **Descrizione** | VoiceInterface (speak/handle_prompt con routing). ExecutiveRouter (route/generate — instrada a identity/memory/knowledge/reasoning/local, provider = internal, carica modello e tokenizer da disco, integra con AgentRuntime). Tutti i provider esterni sono disabilitati per default (config.yaml: providers.enabled=false, fallback.enabled=false). |
| **File principali** | `interface_layer/voice_interface.py`, `executive_system/router.py` |
| **Test** | `TESTS/test_router.py`, `TESTS/test_voice_interface.py`, `TESTS/test_router_tts_integration.py`, `TESTS/test_runtime_integration.py` (test_voice_interface_routes_to_executive_system_and_web_search), `TESTS/test_neural_core/test_model_interface.py` (generate con provider=internal_model) |
| **Stato** | REAL |

---

## 25. Data

| Campo | Valore |
|-------|--------|
| **Implementazione** | `neural_core/data/` + moduli associati |
| **Descrizione** | Dataset corpus (train.txt, validation.txt, test.txt, tokenizer.json/pkl, model.pkl, training_state.pkl, provenance_sentences.jsonl, provenance_manifest.json, baseline_benchmark.json, baseline_manifest.json, dedup/), Dataset (context_size, from_text, get_arrays, batching), DatasetAudit (audit_dataset con splits/duplicates/vocabulary/provenance), DatasetProvenance, ProvenancePipeline (build_all_records, assign_splits, split integrity), LeakageAudit, SentenceDedup, MinHashLSH (deterministic signatures, find_near_duplicates, exact Jaccard verification). |
| **File principali** | `neural_core/data/` (corpus + artefatti), `neural_core/dataset.py`, `neural_core/dataset_audit.py`, `neural_core/dataset_audit_extended.py`, `neural_core/dataset_provenance.py`, `neural_core/provenance_pipeline.py`, `neural_core/leakage_audit.py`, `neural_core/sentence_dedup.py`, `neural_core/minhash_lsh.py` |
| **Test** | `TESTS/test_dataset_audit.py`, `TESTS/test_dataset_provenance.py`, `TESTS/test_final_gap_closure.py` (provenance_pipeline, leakage_audit, minhash_lsh, dataset_audit_extended) |
| **Stato** | REAL — Nessun template/leakage: near_duplicates.pairs verificati, provenance.source_leakage_measured=False, split_integrity all_three_splits_non_empty=True, no_sentence_id_overlap=True |

---

## 26. Model Storage

| Campo | Valore |
|-------|--------|
| **Implementazione** | `neural_core/checkpoint.py`, `neural_core/checkpoint_lineage.py`, `neural_core/checkpoint_metadata.py`, `neural_core/trainer.py` |
| **Descrizione** | Checkpoint (JSON save/load del modello). CheckpointLineage (build_lineage con git_commit, code_version, architecture_version, model_class, model_config, tokenizer_identity, dataset_identity, parameter_count, seed, python_version, numpy_version, platform). Trainer: save_checkpoint, load_checkpoint, save_weights (pickle), load_weights, save_best_checkpoint, load_best_checkpoint. CheckpointMetadata. |
| **File principali** | `neural_core/checkpoint.py`, `neural_core/checkpoint_lineage.py`, `neural_core/checkpoint_metadata.py`, `neural_core/trainer.py` |
| **Test** | `TESTS/test_neural_core/test_checkpoint.py`, `TESTS/test_neural_core/test_checkpoint_corruption.py`, `TESTS/test_neural_core/test_model_persistence.py`, `TESTS/test_checkpoint_corruption.py`, `TESTS/test_final_gap_closure.py` (test_checkpoint_lineage_*), `TESTS/test_neural_core/test_training.py` (train_and_save produces all artefacts), `TESTS/test_neural_core/test_hardening.py` |
| **Stato** | REAL |

---

## 27. Configuration

| Campo | Valore |
|-------|--------|
| **Implementazione** | `CORE/config/configurazione.py`, `config.yaml` |
| **Descrizione** | configurazione.py: NOME_PROGETTO="JARVIS 2.0", VERSIONE="0.1.0", CARTELLA_PROGETTO. config.yaml: identity (name=JARVIS, creator=Riccardo Bronzuoli, external_models_are_identity.enabled=false), neural_core (enabled=true, type=internal, primary=true), llm (default_provider=internal, providers openai/ollama enabled=false), router (primary_system=neural-core, fallback.enabled=false), vision, audit. |
| **File principali** | `CORE/config/configurazione.py`, `config.yaml` |
| **Test** | `TESTS/test_config/test_configurazione.py` |
| **Stato** | REAL |

---

## 28. Logging

| Campo | Valore |
|-------|--------|
| **Implementazione** | `INFRASTRUCTURE/logging.py`, `INFRASTRUCTURE/audit/audit_system.py` |
| **Descrizione** | crea_logger (logger "JARVIS" con StreamHandler e formato timestamp|level|name|message). AuditSystem (log_event con event_type/component/action/status, redaction di campi sensibili, session_id, file rotation con max_file_size/backup_count, get_events/get_session_id). Eventi audit: AGENT_TASK_CREATED, AGENT_TASK_FINISHED, AGENT_ACTION_PROPOSED, AGENT_ACTION_EXECUTED, AGENT_BUDGET_EXHAUSTED, AGENT_DEADLINE_EXCEEDED, TOOL_EXECUTION_START/END, SELF_IMPROVEMENT_CYCLE_START. |
| **File principali** | `INFRASTRUCTURE/logging.py`, `INFRASTRUCTURE/audit/audit_system.py` |
| **Test** | `TESTS/test_logging/test_logging.py`, `TESTS/test_audit.py` |
| **Stato** | REAL |

---

## 29. Testing

| Campo | Valore |
|-------|--------|
| **Implementazione** | `TESTS/` directory — 60+ file pytest |
| **Descrizione** | Suite di test completa: unit test (neural core, reasoning, memory, security, planner, goal manager), integration test (cognitive core + neural core, executive system, tool system + OS sandbox), e2e test (scenario, security layered, AOV cycle), audit test (e2e integration, deep integration, final gap closure). |
| **File principali** | Tutto il contenuto di `TESTS/` (60+ file) |
| **Test** | Auto-referenziale — la suite stessa è il test |
| **Stato** | REAL |

---

## 30. Sandbox

| Campo | Valore |
|-------|--------|
| **Implementazione** | `INFRASTRUCTURE/sandbox/sandbox_env.py`, `neural_core/os_sandbox.py`, `neural_core/os_isolation_probe.py` |
| **Descrizione** | SandboxEnvironment (application-level: path confinement, traversal/symlink prevention, approval gate). OS Sandbox (neural_core/os_sandbox.py): run_isolated con OSIsolationLevel (NONE/BASIC/STRICT/PARANOID), rlimit_cpu (5s), rlimit_fsize, rlimit_nproc, rlimit_nofile (32), rlimit_core (0), setsid (nuova sessione), umask (0o077), env-strip, timeout. macOS usa sandbox-exec (Mandatory Access Control). os_isolation_probe rileva capacità reali. |
| **File principali** | `INFRASTRUCTURE/sandbox/sandbox_env.py`, `neural_core/os_sandbox.py`, `neural_core/os_isolation_probe.py` |
| **Test** | `TESTS/test_os_sandbox.py` (7 classi: availability, legitimate code, blocks escape, timeout, levels, primitives enforced, cross-platform honesty), `TESTS/test_security.py` (TestSandboxEnvironment), `TESTS/test_layered_security.py`, `TESTS/test_tool_system_integration.py`, `TESTS/test_final_gap_closure.py` (test_os_isolation_probe_runs) |
| **Stato** | REAL — OS-level isolation reale (sandbox-exec su macOS, rlimits+chdir su Linux) |

---

## 31. Capability Registry

| Campo | Valore |
|-------|--------|
| **Implementazione** | `executive_system/capability_registry.py` |
| **Descrizione** | CapabilityRegistry: register (con stato available/missing/restricted), register_tool (con factory callable), get_factory, state. Stati ammessi: available, missing, restricted. |
| **File principali** | `executive_system/capability_registry.py` |
| **Test** | `TESTS/test_executive_system.py`, `TESTS/test_runtime_integration.py` (test_capability_registry_registers_web_search_tool) |
| **Stato** | REAL |

---

## 32. Goal Management

| Campo | Valore |
|-------|--------|
| **Implementazione** | `executive_system/goal_manager.py` |
| **Descrizione** | GoalManager: add (con steps, status pending/in_progress/completed/failed), start, next_pending_step, complete_step, complete, fail_step, status_summary. Stato e ciclo di vita completo. |
| **File principali** | `executive_system/goal_manager.py` |
| **Test** | `TESTS/test_goal_manager.py` (7 test: add, get_by_text, start/next_pending, complete, fail, status_summary), `TESTS/test_self_improvement_disabled.py` |
| **Stato** | REAL |

---

## 33. Observation / Feedback

| Campo | Valore |
|-------|--------|
| **Implementazione** | `executive_system/observation_feedback.py` |
| **Descrizione** | ObservationFeedback: record (legacy two-arg + structured triplet con outcome), record_structured (action/observation/verification con evidence/provenance), success_rate, metrics (total/successes/failures/unexpected), outcome classification (success/failure), suggestion generation (richiedi_approvazione_utente per failure). |
| **File principali** | `executive_system/observation_feedback.py` |
| **Test** | `TESTS/test_observation_feedback.py` (5 test: record legacy, record with outcome, classify failure, success rate, metrics), `TESTS/test_phase2_action_observation_verification.py` (TestStructuredObservation, TestAgentRuntimeAOV) |
| **Stato** | REAL |

---

## Summary

| # | Organo | Stato | File principale |
|---|--------|-------|----------------|
| 1 | Cognitive Core | REAL | `cognitive_core/coordinator.py` |
| 2 | Artificial Intelligence / Neural Core | REAL | `neural_core/` |
| 3 | Perception | REAL | `cognitive_core/understanding.py` |
| 4 | Reasoning | REAL | `cognitive_core/reasoning/` |
| 5 | Planning | REAL | `cognitive_core/planner.py` |
| 6 | Memory | REAL | `cognitive_core/memory/` |
| 7 | Knowledge | REAL | `cognitive_core/knowledge/` |
| 8 | World Model | REAL | `cognitive_core/world_model/` |
| 9 | Imagination | **NON IMPLEMENTATO** | — |
| 10 | Discovery | REAL | `expansion_system/discovery/` |
| 11 | Learning | REAL | `neural_core/trainer.py` |
| 12 | Self-Improvement | REAL | `expansion_system/self_improvement/` |
| 13 | Tools | REAL | `executive_system/tools/` |
| 14 | Tool Factory | REAL | `expansion_system/tool_factory/` |
| 15 | Action | REAL | `cognitive_core/runtime/action.py` |
| 16 | Environment | REAL | `INFRASTRUCTURE/sandbox/sandbox_env.py` |
| 17 | Communication | REAL | `interface_layer/voice_interface.py` |
| 18 | User Model | **NON IMPLEMENTATO** | — |
| 19 | Security | REAL | `INFRASTRUCTURE/security/security_layer.py` |
| 20 | Permissions | REAL | `INFRASTRUCTURE/security/permission_guard.py` |
| 21 | Autonomy | REAL | `executive_system/autonomy_manager.py` |
| 22 | Simulation | REAL | `cognitive_core/world_model/simulator.py` |
| 23 | Evaluation | REAL | `neural_core/evaluator.py` |
| 24 | Interface | REAL | `executive_system/router.py` |
| 25 | Data | REAL | `neural_core/data/` |
| 26 | Model Storage | REAL | `neural_core/checkpoint.py` |
| 27 | Configuration | REAL | `CORE/config/configurazione.py` |
| 28 | Logging | REAL | `INFRASTRUCTURE/logging.py` |
| 29 | Testing | REAL | `TESTS/` |
| 30 | Sandbox | REAL | `neural_core/os_sandbox.py` |
| 31 | Capability Registry | REAL | `executive_system/capability_registry.py` |
| 32 | Goal Management | REAL | `executive_system/goal_manager.py` |
| 33 | Observation / Feedback | REAL | `executive_system/observation_feedback.py` |

**Totale**: 31/33 organi REAL, 2/33 NON IMPLEMENTATI (Imagination, User Model).
