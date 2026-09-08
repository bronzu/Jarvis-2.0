# Dependencies and Provenance

## 1. Runtime Dependencies

### 1.1 Python Packages (`requirements.txt`)
```
numpy
requests
beautifulsoup4
pyyaml
pytest
```

| Package | Version (observed) | License | Purpose | Risk |
|---|---|---|---|---|
| numpy | 1.26.4 | BSD | Numerical computing | LOW — permissive license, widely used |
| requests | — | Apache 2.0 | HTTP client | LOW — permissive license |
| beautifulsoup4 | — | MIT | HTML parsing | LOW — permissive license |
| pyyaml | — | MIT | YAML parsing | LOW — permissive license |
| pytest | — | MIT | Testing | LOW — permissive license |

### 1.2 Optional/Development Dependencies
- **PyTorch** — used in `torch_model.py` and `gpu_benchmark_extended.py` for GPU parity testing. NOT in `requirements.txt`. Not required for production.
- **pytest plugins** — `pytest-randomly` mentioned in documentation but not installed (no internet).

---

## 2. External Service Dependencies

### 2.1 External LLM APIs
| Service | Config | Production Path | Status |
|---|---|---|---|
| OpenAI | `llm.providers.openai.enabled: false` | NOT CALLED | DISABLED |
| Ollama | `llm.providers.ollama.enabled: false` | NOT CALLED | DISABLED |

**Evidence:**
- `neural_core/providers/external_llm_provider.py` — exists but not imported by production code
- `executive_system/router.py` — `_decide_route()` routes to identity/memory/knowledge/reasoning/local only
- `test_runtime_integration.py` — asserts `ExternalLLMProvider().is_enabled("openai") is False`

### 2.2 Network Dependencies
| Service | Purpose | Gated By | Status |
|---|---|---|---|
| Web search (via `requests`) | InternetTool, WebBrowserTool | NetworkPolicy | CONFIGURABLE |
| OpenAI API | ExternalLLMProvider (optional) | Config flag | DISABLED |
| Ollama API | ExternalLLMProvider (optional) | Config flag | DISABLED |

---

## 3. OS-Level Dependencies

| Dependency | Platform | Purpose | Status |
|---|---|---|---|
| `sandbox-exec` | macOS | Kernel-enforced MAC isolation | REQUIRED for full OS sandbox |
| `chroot` | Linux (root) | Filesystem isolation | OPTIONAL (falls back to chdir) |
| `rlimit` | Linux/macOS | Resource limits | REQUIRED |
| `setsid` | Linux/macOS | Process group isolation | REQUIRED |

**Portability Risk:** `sandbox-exec` is macOS-only. Linux uses `chroot` (requires root for full isolation). Windows is unsupported.

---

## 4. Dataset Provenance

### 4.1 Dataset Files
| File | Size | Records | Source | License | Risk |
|---|---|---|---|---|---|
| `full_corpus.txt` | 33.4 MB | 30,361 | Unknown | Unknown | **HIGH** — provenance unclear |
| `corpus_addestramento.txt` | 1.5 KB | 22 lines | Unknown | Unknown | **MEDIUM** — small, likely example data |
| `train.txt` | 4.95 MB | — | Derived from `full_corpus.txt` | Unknown | **HIGH** — inherits source risk |
| `validation.txt` | 617 KB | — | Derived from `full_corpus.txt` | Unknown | **HIGH** |
| `test.txt` | 614 KB | — | Derived from `full_corpus.txt` | Unknown | **HIGH** |
| `jarvis_dialogue.txt` | 15 KB | — | Internal | Unknown | **LOW** — likely synthetic |

### 4.2 Provenance Tracking
| Artifact | Tracking | Evidence |
|---|---|---|
| Dataset SHA256 | YES | `provenance_manifest.json` |
| Generation seed | YES | `provenance_manifest.json`: `generation_seed: 42` |
| Shuffling seed | YES | `provenance_manifest.json`: `shuffling_seed: 42` |
| Original source | NO | Not tracked in manifest |
| Data licenses | NO | Not documented |
| Author attribution | NO | Not documented |

### 4.3 Dataset Risk Summary
The dataset provenance is **incomplete**. While generation parameters are tracked, the original sources of `full_corpus.txt` are not documented. If the corpus contains copyrighted material, this could affect the IP status of the trained model.

---

## 5. Model Provenance

### 5.1 Model Architecture
| Aspect | Status | Evidence |
|---|---|---|
| Architecture name | YES | `MicroTransformer` |
| Architecture version | YES | `jarvis2-microtransformer-v1` |
| Parameter count | YES | 2,737,806 |
| Config | YES | vocab_size=8590, d_model=128, num_heads=4, d_ff=256, num_layers=4, max_len=256 |
| Training code | YES | `trainer.py`, `train.py` |
| Training data | PARTIAL | `full_corpus.txt` (source unclear) |
| Training seed | YES | Included in checkpoint |
| Training config | YES | Included in checkpoint |

### 5.2 Model Artifacts
| File | Size | Format | Content |
|---|---|---|---|
| `model.pkl` | 21.9 MB | pickle | Trained weights (69 arrays) |
| `best_model.json` | 53 MB | JSON | Alternative config (num_layers=2) |
| `baseline_model.pkl` | 3.4 MB | pickle | Baseline model (730K params) |
| `tokenizer.pkl` | — | pickle | Tokenizer vocabulary |

---

## 6. Code Provenance

### 6.1 Repository Status
- **Not a git repository** — `git status` returns "not a git repository"
- **No commit history** — cannot trace individual changes
- **File modification times** — available via filesystem but not systematically tracked

### 6.2 Imported Code
| Module | Source | Evidence |
|---|---|---|
| `numpy` | PyPI | `requirements.txt` |
| `requests` | PyPI | `requirements.txt` |
| `beautifulsoup4` | PyPI | `requirements.txt` |
| `pyyaml` | PyPI | `requirements.txt` |
| `pytest` | PyPI | `requirements.txt` |
| `torch` | PyPI (optional) | `torch_model.py`, not in requirements.txt |

### 6.3 Code Quality Indicators
| Indicator | Status | Evidence |
|---|---|---|
| TODO/FIXME/HACK/XXX in source | NONE | `grep` returned 0 results |
| `except: pass` in source | NONE | `grep` returned 0 results (one match in docstring) |
| Mock implementations | MINIMAL | `_web_search` in `e2e_scenario.py` is documented mock |
| Test-only implementations | MINIMAL | Some test helpers exist but do not affect production |
| Dead code | MINIMAL | No unreachable code identified |

---

## 7. License Summary

| Component | License | Obligations |
|---|---|---|
| numpy | BSD | Attribution required |
| requests | Apache 2.0 | Attribution required |
| beautifulsoup4 | MIT | Attribution required |
| pyyaml | MIT | Attribution required |
| pytest | MIT | Attribution required |
| JARVIS source code | UNKNOWN | **No LICENSE file found in repository** |

**CRITICAL FINDING:** No `LICENSE` file exists in the repository. The proprietary status of the JARVIS code is not explicitly declared. This must be resolved before acquisition.

---

*This document is part of the Technical Acquisition Package.*
