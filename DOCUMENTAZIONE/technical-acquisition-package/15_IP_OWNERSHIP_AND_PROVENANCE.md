# IP Ownership & Provenance

**Date:** 2026-09-08  
**Classification:** Due Diligence Only  
**Version:** 1.0

---

## 1. Executive Summary

This document consolidates the intellectual property ownership claims and provenance evidence for the JARVIS 2.0 repository. Based on repository evidence and author declaration, the project appears to be developed by a single author with internally developed components, though several areas require legal verification before acquisition.

---

## 2. Author Declaration

### 2.1 Stated Position

| Aspect | Declaration | Evidence in Repository |
|---|---|---|
| **Author** | Single author: Riccardo Bronzuoli | `config.yaml:127-129` (creator field) |
| **Code origin** | All code written personally, no external LLM/AI generation | No AI-generated code patterns detected; consistent style across 236 files |
| **Development period** | Files contain author name and temporal information | File modification times span Aug 2024 - Sep 2026 |
| **Components** | AgentRuntime, sandbox, security layer, memory, planning, world model, tokenizer, training pipeline, MicroTransformer, tests, documentation developed internally | All components present in repository with original implementations |

### 2.2 Evidence Supporting Declaration

| Evidence Type | Description | Verification Status |
|---|---|---|
| **Author attribution in config** | `config.yaml` explicitly names "Riccardo Bronzuoli" as creator | VERIFIED |
| **Consistent coding style** | Uniform patterns, naming conventions, architecture across all modules | OBSERVED |
| **No AI-generation markers** | No characteristic LLM patterns (excessive comments, generic variable names, boilerplate) | OBSERVED |
| **Original implementations** | MicroTransformer in pure NumPy, custom tokenizer, proprietary sandbox, AgentRuntime lifecycle | VERIFIED |
| **Comprehensive test suite** | 1,031 tests written alongside implementation code | VERIFIED |
| **Documentation depth** | Technical reports, architecture docs, audit reports showing iterative development | VERIFIED |

---

## 3. Code Provenance Analysis

### 3.1 Repository Composition

| Category | Files | Lines (est.) | Provenance |
|---|---|---|---|
| **Neural Core** | 47 | ~15,000 | Original implementation |
| **Cognitive Core** | 32 | ~12,000 | Original implementation |
| **Executive System** | 19 | ~8,000 | Original implementation |
| **Expansion System** | 14 | ~5,000 | Original implementation (disabled) |
| **Infrastructure** | 9 | ~3,000 | Original implementation |
| **Interface Layer** | 2 | ~500 | Original implementation |
| **CORE System** | 4 | ~500 | Original implementation |
| **Entry Points** | 2 | ~500 | Original implementation |
| **Tests** | 178 | ~35,000 | Original implementation |
| **Total** | **307** | **~79,500** | **All original** |

### 3.2 Third-Party Dependencies

| Dependency | License | Purpose | Provenance Risk |
|---|---|---|---|
| **numpy** | BSD-3-Clause | Numerical computing | LOW — permissive |
| **requests** | Apache-2.0 | HTTP client | LOW — permissive |
| **beautifulsoup4** | MIT | HTML parsing | LOW — permissive |
| **pyyaml** | MIT | YAML parsing | LOW — permissive |
| **pytest** | MIT | Testing framework | LOW — permissive |
| **PyTorch** (optional) | BSD-3-Clause | GPU parity testing only | LOW — not in production |

**No GPL or copyleft dependencies detected.**

### 3.3 Open Source Components

| Component | Source | License | Status |
|---|---|---|---|
| **Python standard library** | PSF | PSF-2.0 | Runtime dependency |
| **macOS sandbox-exec** | Apple | APSL-2.0 | System binary, not distributed |
| **Linux chroot/rlimits** | Linux kernel | GPL-2.0 (kernel) | System calls, not distributed |

---

## 4. Component Ownership Matrix

| Component | Development | Third-Party Code | License Status | Notes |
|---|---|---|---|---|
| **MicroTransformer (network.py)** | Internal | None | Proprietary | Pure NumPy transformer |
| **Tokenizer (tokenizer.py)** | Internal | None | Proprietary | Whitespace + technical modes |
| **Trainer (trainer.py, train.py)** | Internal | None | Proprietary | Adam, schedulers, checkpoints |
| **AgentRuntime** | Internal | None | Proprietary | 11-state lifecycle, recovery |
| **OS Sandbox (os_sandbox.py)** | Internal | System calls | Proprietary | macOS sandbox-exec, Linux chroot |
| **Security Layer** | Internal | None | Proprietary | 5-layer defense |
| **Memory Systems** | Internal | None | Proprietary | 7 subsystems |
| **Reasoning Engine** | Internal | None | Proprietary | Deductive, inductive, classification |
| **World Model** | Internal | None | Proprietary | Entity-relation-state |
| **Planner** | Internal | None | Proprietary | Goal decomposition |
| **Tools (5 tools)** | Internal | None | Proprietary | Filesystem, terminal, python, web, internet |
| **Permission Guard** | Internal | None | Proprietary | Regex blacklist |
| **Audit System** | Internal | None | Proprietary | Structured logging, redaction |
| **Network Policy** | Internal | None | Proprietary | Host allowlist |
| **Self-Improvement** | Internal | None | Proprietary | Disabled by config |
| **Discovery Engine** | Internal | None | Proprietary | Disabled by config |

---

## 5. Dataset Provenance

### 5.1 Dataset Inventory

| Dataset | Size | Source | Copyright Status | License | Evidence | Training Use |
|---|---|---|---|---|---|---|
| `full_corpus.txt` | 33.4 MB | Unknown / online books | **UNVERIFIED** | **UNVERIFIED** | `provenance_manifest.json` tracks SHA256, seed | Primary training corpus |
| `cleaned_literature.txt` | 5.7 MB | Unknown | **UNVERIFIED** | **UNVERIFIED** | Derived from full_corpus | Training |
| `train.txt` | 4.9 MB | Derived | **INHERITS RISK** | **INHERITS RISK** | Split from full_corpus | Training |
| `validation.txt` | 617 KB | Derived | **INHERITS RISK** | **INHERITS RISK** | Split from full_corpus | Validation |
| `test.txt` | 614 KB | Derived | **INHERITS RISK** | **INHERITS RISK** | Split from full_corpus | Testing |
| `jarvis_dialogue.txt` | 15 KB | Likely synthetic/internal | **LOW RISK** | **UNVERIFIED** | Internal generation | Dialogue fine-tuning |
| `corpus_addestramento.txt` | 1.5 KB | Example/placeholder | **UNVERIFIED** | **UNVERIFIED** | Root directory | Example only |

### 5.2 Critical Finding

> **REQUIRES LEGAL/PROVENANCE VERIFICATION**: The primary training corpus (`full_corpus.txt`, 33.4 MB, 30,361 records) has **undocumented original sources**. The author declares it comprises "opere scaricabili senza copyright / public domain" but **no source URLs, author attributions, or public domain verifications are recorded in the repository**. The `provenance_manifest.json` tracks generation parameters (seed=42, deduplication) but not original data provenance.

### 5.3 Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Copyright infringement in training data | **HIGH** | Audit full_corpus.txt sources; obtain licenses |
| Model weights tainted by unlicensed data | **HIGH** | Retrain on verified corpus if needed |
| Public domain claims unverified | **HIGH** | Legal review of each source |

---

## 6. Model Provenance

### 6.1 Training Pipeline

```
DATASET (full_corpus.txt)
    ↓ preprocessing (dataset.py, build_dataset.py)
TOKENIZER (tokenizer.py) → tokenizer.pkl / tokenizer.json
    ↓ training (trainer.py, train.py)
CHECKPOINT (checkpoint.py, checkpoint_lineage.py)
    ↓ model artifact
MODEL (model.pkl, 21.9 MB, 2.7M params)
```

### 6.2 Model Artifact Lineage

| Artifact | Verification |
|---|---|
| **Architecture** | MicroTransformer, d_model=128, 4 heads, 4 layers, d_ff=256 |
| **Parameter count** | 2,737,806 (verified from code) |
| **Training config** | Embedded in checkpoint (epochs, lr, batch_size, seed) |
| **Tokenizer** | Serialized with model (vocab_size=8,590) |
| **Dataset identity** | SHA256 per split in `provenance_manifest.json` |
| **Code version** | 0.2.0 (from `VERSIONE.md`) |
| **Git commit** | N/A — repository not a git repo |
| **Environment** | Python version, NumPy version, platform, timestamp in checkpoint |
| **Reproducibility** | `test_reproducibility.py` passes; fixed seed |

---

## 7. Areas Requiring Legal Verification

| Area | Status | Required Action |
|---|---|---|
| **Dataset copyright status** | **UNVERIFIED** | Legal audit of full_corpus.txt sources |
| **Model weight ownership** | **DEPENDS ON DATASET** | Resolve dataset first |
| **Patentability of Class A components** | **NOT ASSESSED** | Engage patent counsel |
| **Freedom-to-operate (transformer)** | **NOT ASSESSED** | Patent landscape analysis |
| **Freedom-to-operate (sandbox)** | **NOT ASSESSED** | Patent landscape analysis |
| **Trademark "JARVIS"** | **UNKNOWN** | Trademark search |
| **Contributor agreements** | **NOT COLLECTED** | Single author declared; confirm no other contributors |
| **Third-party license compliance** | **VERIFIED** | All dependencies permissive |

---

## 8. Evidence Limitations

| Limitation | Impact |
|---|---|
| **No Git history** | Cannot prove chronological development or individual authorship of specific commits |
| **File timestamps only** | Modification times can be altered; not cryptographic proof |
| **No signed attestations** | Author declaration is self-reported |
| **Dataset sources unknown** | Cannot verify public domain claims without external investigation |
| **No independent audit** | All verification performed by development team |

---

## 9. Recommended Legal Actions (Pre-Acquisition)

1. **Dataset Provenance Audit** — Engage legal counsel to trace `full_corpus.txt` sources and verify public domain / licensing status
2. **IP Assignment Confirmation** — Execute written IP assignment from author to acquiring entity
3. **Patent Filing** — File provisional patents for Class A components (AgentRuntime lifecycle, OS sandbox integration, structured action/verification/recovery loop) before any public disclosure
4. **Freedom-to-Operate Analysis** — Engage patent counsel for transformer architecture and OS sandbox approaches
5. **Trademark Clearance** — Search and register "JARVIS" trademark if not already protected
6. **License Decision** — Determine proprietary vs. open-source strategy; add appropriate LICENSE file
7. **Third-Party Notices** — Create THIRD_PARTY_NOTICES.md with dependency attributions

---

## 10. Summary Statement

> **Based on the repository evidence and author declaration:**
> - The codebase (236 Python files, ~79K lines) appears to be original work by a single author
> - No evidence of external AI-generated code or copied implementations
> - All neural core, cognitive core, executive system, and infrastructure components are internally developed
> - Third-party dependencies are limited to 5 permissive-license Python packages
> - **Critical gap**: Training dataset (`full_corpus.txt`) provenance is undocumented and requires legal verification
> - **Critical gap**: No LICENSE file exists; proprietary status not formally declared
> - **Critical gap**: No Git history exists to cryptographically prove development timeline

This document should be reviewed by legal counsel before any acquisition transaction.

---

*This document is part of the Technical Acquisition Package.*