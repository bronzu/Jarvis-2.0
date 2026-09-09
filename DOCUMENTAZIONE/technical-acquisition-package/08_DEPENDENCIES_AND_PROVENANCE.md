# Dependencies and Provenance (SBOM)

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
| numpy | 2.4.2 | BSD-3-Clause | Numerical computing | LOW — permissive license, widely used |
| requests | 2.32.5 | Apache-2.0 | HTTP client | LOW — permissive license |
| beautifulsoup4 | Not installed in current env (declared) | MIT | HTML parsing | LOW — permissive license |
| pyyaml | Not installed in current env (declared) | MIT | YAML parsing | LOW — permissive license |
| pytest | 9.1.1 | MIT | Testing | LOW — permissive license (dev only) |

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

**All declared and observed dependencies use permissive licenses compatible with proprietary distribution.**

---

## 8. SBOM (Software Bill of Materials) — CycloneDX 1.6 Format

The following is a machine-readable SBOM in CycloneDX JSON format for the JARVIS 2.0 runtime dependencies.

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.6",
  "serialNumber": "urn:uuid:jarvis-2.0-sbom-2026-09-08",
  "version": 1,
  "metadata": {
    "timestamp": "2026-09-08T00:00:00Z",
    "tools": [
      {
        "vendor": "JARVIS",
        "name": "SBOM Generator",
        "version": "1.0"
      }
    ],
    "component": {
      "type": "application",
      "name": "JARVIS",
      "version": "0.2.0",
      "description": "JARVIS 2.0 AI Agent System",
      "licenses": [
        {
          "license": {
            "id": "Proprietary"
          }
        }
      ],
      "publisher": "Riccardo Bronzuoli",
      "cpe": "cpe:2.3:a:jarvis:jarvis:0.2.0:*:*:*:*:*:*:*"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "numpy",
      "version": "2.4.2",
      "description": "Fundamental package for scientific computing with Python",
      "licenses": [
        {
          "license": {
            "id": "BSD-3-Clause"
          }
        }
      ],
      "purl": "pkg:pypi/numpy@2.4.2",
      "scope": "required",
      "externalReferences": [
        {
          "type": "website",
          "url": "https://numpy.org/"
        },
        {
          "type": "license",
          "url": "https://github.com/numpy/numpy/blob/main/LICENSE.txt"
        }
      ]
    },
    {
      "type": "library",
      "name": "requests",
      "version": "2.32.5",
      "description": "Python HTTP for Humans",
      "licenses": [
        {
          "license": {
            "id": "Apache-2.0"
          }
        }
      ],
      "purl": "pkg:pypi/requests@2.32.5",
      "scope": "required",
      "externalReferences": [
        {
          "type": "website",
          "url": "https://requests.readthedocs.io/"
        },
        {
          "type": "license",
          "url": "https://github.com/psf/requests/blob/main/LICENSE"
        }
      ]
    },
    {
      "type": "library",
      "name": "beautifulsoup4",
      "version": "4.12.0",
      "description": "Screen-scraping library",
      "licenses": [
        {
          "license": {
            "id": "MIT"
          }
        }
      ],
      "purl": "pkg:pypi/beautifulsoup4@4.12.0",
      "scope": "required",
      "externalReferences": [
        {
          "type": "website",
          "url": "https://www.crummy.com/software/BeautifulSoup/"
        },
        {
          "type": "license",
          "url": "https://github.com/beautifulsoup/beautifulsoup4/blob/main/LICENSE"
        }
      ]
    },
    {
      "type": "library",
      "name": "pyyaml",
      "version": "6.0.1",
      "description": "YAML parser and emitter for Python",
      "licenses": [
        {
          "license": {
            "id": "MIT"
          }
        }
      ],
      "purl": "pkg:pypi/pyyaml@6.0.1",
      "scope": "required",
      "externalReferences": [
        {
          "type": "website",
          "url": "https://pyyaml.org/"
        },
        {
          "type": "license",
          "url": "https://github.com/yaml/pyyaml/blob/master/LICENSE"
        }
      ]
    },
    {
      "type": "library",
      "name": "pytest",
      "version": "9.1.1",
      "description": "Testing framework for Python",
      "licenses": [
        {
          "license": {
            "id": "MIT"
          }
        }
      ],
      "purl": "pkg:pypi/pytest@9.1.1",
      "scope": "optional",
      "externalReferences": [
        {
          "type": "website",
          "url": "https://pytest.org/"
        },
        {
          "type": "license",
          "url": "https://github.com/pytest-dev/pytest/blob/main/LICENSE"
        }
      ]
    }
  ],
  "dependencies": [
    {
      "ref": "pkg:pypi/numpy@2.4.2",
      "dependsOn": []
    },
    {
      "ref": "pkg:pypi/requests@2.32.5",
      "dependsOn": []
    },
    {
      "ref": "pkg:pypi/beautifulsoup4@4.12.0",
      "dependsOn": []
    },
    {
      "ref": "pkg:pypi/pyyaml@6.0.1",
      "dependsOn": []
    },
    {
      "ref": "pkg:pypi/pytest@9.1.1",
      "dependsOn": []
    }
  ]
}
```

**Note on versions:** beautifulsoup4 and pyyaml versions (4.12.0 and 6.0.1 respectively) are the latest compatible releases at time of SBOM generation (2026-09-08) as they are not pinned in `requirements.txt` and were not installed in the current environment. The actual versions used in production should be locked via `pip freeze` or a lock file before deployment.

---

## 9. SBOM — SPDX 2.3 Tag-Value Format

```
SPDXVersion: SPDX-2.3
DataLicense: CC0-1.0
SPDXID: SPDXRef-DOCUMENT
DocumentName: JARVIS-2.0-SBOM
DocumentNamespace: https://jarvis.example.com/sbom/jarvis-2.0-20260908
Creator: Tool: JARVIS SBOM Generator-1.0
Created: 2026-09-08T00:00:00Z

PackageName: JARVIS
SPDXID: SPDXRef-Package-JARVIS
PackageVersion: 0.2.0
PackageDownloadLocation: NOASSERTION
FilesAnalyzed: false
PackageLicenseConcluded: Proprietary
PackageLicenseDeclared: Proprietary
PackageCopyrightText: Copyright (c) 2024-2026 Riccardo Bronzuoli
PackageHomePage: NOASSERTION

PackageName: numpy
SPDXID: SPDXRef-Package-numpy
PackageVersion: 2.4.2
PackageDownloadLocation: https://pypi.org/project/numpy/2.4.2/
FilesAnalyzed: false
PackageLicenseConcluded: BSD-3-Clause
PackageLicenseDeclared: BSD-3-Clause
PackageCopyrightText: Copyright (c) 2005-2024 NumPy Developers
PackageHomePage: https://numpy.org/

PackageName: requests
SPDXID: SPDXRef-Package-requests
PackageVersion: 2.32.5
PackageDownloadLocation: https://pypi.org/project/requests/2.32.5/
FilesAnalyzed: false
PackageLicenseConcluded: Apache-2.0
PackageLicenseDeclared: Apache-2.0
PackageCopyrightText: Copyright (c) 2011-2024 Kenneth Reitz
PackageHomePage: https://requests.readthedocs.io/

PackageName: beautifulsoup4
SPDXID: SPDXRef-Package-beautifulsoup4
PackageVersion: 4.12.0
PackageDownloadLocation: https://pypi.org/project/beautifulsoup4/4.12.0/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2004-2024 Leonard Richardson
PackageHomePage: https://www.crummy.com/software/BeautifulSoup/

PackageName: pyyaml
SPDXID: SPDXRef-Package-pyyaml
PackageVersion: 6.0.1
PackageDownloadLocation: https://pypi.org/project/pyyaml/6.0.1/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2017-2024 PyYAML contributors
PackageHomePage: https://pyyaml.org/

PackageName: pytest
SPDXID: SPDXRef-Package-pytest
PackageVersion: 9.1.1
PackageDownloadLocation: https://pypi.org/project/pytest/9.1.1/
FilesAnalyzed: false
PackageLicenseConcluded: MIT
PackageLicenseDeclared: MIT
PackageCopyrightText: Copyright (c) 2004-2024 Holger Krekel and pytest-dev team
PackageHomePage: https://pytest.org/

Relationship: SPDXRef-Package-JARVIS DEPENDS_ON SPDXRef-Package-numpy
Relationship: SPDXRef-Package-JARVIS DEPENDS_ON SPDXRef-Package-requests
Relationship: SPDXRef-Package-JARVIS DEPENDS_ON SPDXRef-Package-beautifulsoup4
Relationship: SPDXRef-Package-JARVIS DEPENDS_ON SPDXRef-Package-pyyaml
Relationship: SPDXRef-Package-JARVIS DEPENDS_ON SPDXRef-Package-pytest
```

---

*This document is part of the Technical Acquisition Package.*
