# Dataset / Corpus Provenance

**Date:** 2026-09-08  
**Classification:** Due Diligence Only  
**Version:** 1.0

---

## 1. Executive Summary

This document provides a structured provenance record for all datasets and corpora used in the training of the JARVIS 2.0 MicroTransformer model. The primary training corpus (`full_corpus.txt`) consists of **30,360 lines** of Italian-language text covering general knowledge topics. **Original sources are not documented in the repository**, and copyright/public domain status is **UNVERIFIED** and **REQUIRES LEGAL/PROVENANCE VERIFICATION**.

---

## 2. Dataset Inventory

| Dataset/Opera | Autore | Source | Copyright/Public Domain Status | License | Evidence | Training Use |
|---|---|---|---|---|---|---|
| `full_corpus.txt` | Unknown / Generated | Repository file (33.4 MB, 30,360 lines) | **UNVERIFIED** | **UNVERIFIED** | `provenance_manifest.json` tracks SHA256, seed=42 | Primary training corpus |
| `cleaned_literature.txt` | Niccolò Machiavelli (public domain works) | Repository file (5.7 MB, ~14 very long lines) | **LIKELY PUBLIC DOMAIN** (Machiavelli died 1527) | Public Domain (Italy/US) | Content matches "Il Principe", "Discorsi", "Arte della Guerra" | Secondary corpus (literature) |
| `train.txt` | Derived | Split from `full_corpus.txt` (4.95 MB) | **INHERITS RISK** from full_corpus | **INHERITS RISK** | `provenance_manifest.json` | Training split |
| `validation.txt` | Derived | Split from `full_corpus.txt` (617 KB) | **INHERITS RISK** from full_corpus | **INHERITS RISK** | `provenance_manifest.json` | Validation split |
| `test.txt` | Derived | Split from `full_corpus.txt` (614 KB) | **INHERITS RISK** from full_corpus | **INHERITS RISK** | `provenance_manifest.json` | Test split |
| `jarvis_dialogue.txt` | Internal / Synthetic | Repository file (15 KB, 290 lines, 194 Q&A pairs) | **LOW RISK** — internally generated | **UNVERIFIED** | Internal generation | Dialogue fine-tuning |
| `corpus_addestramento.txt` | Example/Placeholder | Root directory (1.5 KB, 22 lines) | **UNVERIFIED** | **UNVERIFIED** | Example corpus in root | Example only (not used in production training) |

---

## 3. Detailed Provenance Analysis

### 3.1 full_corpus.txt (Primary Training Corpus)

**File:** `neural_core/data/full_corpus.txt`  
**Size:** 33,428,159 bytes (33.4 MB)  
**Lines:** 30,360  
**Records (per manifest):** 30,361  
**SHA256 (from manifest):** `b4a55d911a5fa9a42fab5da9e0e043a9`  
**Generation seed:** 42  
**Template family:** mixed

**Content Analysis:**
- Italian language text covering general knowledge topics: Italian culture, coffee traditions, technology, science, geography, mathematics, renewable energy, databases, SQL, version control, etc.
- Text appears to be **generated or curated synthetic content**, not extracted from published books
- Repetitive patterns observed (e.g., "Il caffè espresso è una tradizione italiana..." appears multiple times)
- No clear attribution to specific authors, publishers, or public domain sources
- No metadata headers indicating source URLs, ISBNs, or repository origins

**Provenance Evidence:**
- `provenance_manifest.json` tracks: generation seed (42), shuffling seed (42), deduplication method, SHA256 checksums
- **Original sources NOT documented** — no source URLs, author attributions, publication dates, or public domain verifications
- Author declaration: "opere scaricabili senza copyright / public domain" — **NOT VERIFIED** by repository evidence

**Risk Assessment:** **HIGH** — Copyright status unknown. If corpus contains non-public-domain material, model weights may be tainted.

---

### 3.2 cleaned_literature.txt (Literature Corpus)

**File:** `neural_core/data/cleaned_literature.txt`  
**Size:** 5,681,732 bytes (5.7 MB)  
**Lines:** ~14 (very long lines containing full texts)  
**Records (per manifest):** 8  
**SHA256 (from manifest):** `e1960fef1d883e05d930549260099320`  
**Content Analysis:**
- Contains complete texts of Niccolò Machiavelli's works:
  - "Il Principe" (The Prince) — published 1532, author died 1527
  - "Discorsi sopra la prima deca di Tito Livio" (Discourses on Livy)
  - "Dell'arte della guerra" (Art of War)
  - "Provvisioni" (Provisions for Florentine militia)
  - "Consulti" and "Relazioni" (consultations and reports)
- Machiavelli died in 1527 — works are **public domain** in Italy (70 years post-mortem) and US (published before 1929)
- Text appears to be from Project Gutenberg or similar public domain source

**Provenance Evidence:**
- Content directly matches known public domain works by Machiavelli
- **LIKELY PUBLIC DOMAIN** — but specific source edition/URL not documented in repository
- No explicit license file accompanying the text

**Risk Assessment:** **LOW** — Author died 1527, works unequivocally public domain. However, specific digital edition source should be documented for complete provenance.

---

### 3.3 jarvis_dialogue.txt (Dialogue Fine-tuning Corpus)

**File:** `neural_core/data/jarvis_dialogue.txt`  
**Size:** 15,016 bytes (15 KB)  
**Lines:** 290 (194 Q&A pairs)  
**Records (per manifest):** 194  
**SHA256 (from manifest):** `89aa0f144e5d06ee5bf79c87b3eba00d`  
**Content Analysis:**
- 194 synthetic dialogue pairs in Italian
- Covers: greetings, identity questions, capability questions, technical explanations (ML concepts), math problems, tool usage scenarios, safety refusals, personal questions
- Clearly **internally generated** for dialogue fine-tuning
- Format: "Utente: ...\nAssistente: ..." pairs

**Provenance Evidence:**
- Internal synthetic generation — no external source
- Author: JARVIS development team (Riccardo Bronzuoli)

**Risk Assessment:** **LOW** — Internally generated, no third-party copyright concerns.

---

### 3.4 Derived Splits (train.txt, validation.txt, test.txt)

| File | Size | Source | Provenance |
|---|---|---|---|
| `train.txt` | 4.95 MB | Split from `full_corpus.txt` | Inherits full_corpus risk |
| `validation.txt` | 617 KB | Split from `full_corpus.txt` | Inherits full_corpus risk |
| `test.txt` | 614 KB | Split from `full_corpus.txt` | Inherits full_corpus risk |

**Split Methodology (from `provenance_manifest.json`):**
- Generation seed: 42
- Shuffling seed: 42
- Deduplication: MinHash/LSH (threshold 0.85), sentence-level
- Exact duplicates: 0 (verified)
- Normalized duplicates: 0 (verified)
- Sentence duplicates: 0 (verified)
- Near duplicates: 0 (verified)

---

## 4. Provenance Tracking Artifacts

| Artifact | Path | Content |
|---|---|---|
| Provenance Manifest | `neural_core/data/provenance_manifest.json` | Generation seed, shuffling seed, split ratios, deduplication method, SHA256 per file, record counts |
| Provenance Sentences | `neural_core/data/provenance_sentences.jsonl` | Sentence-level provenance tracking (12.9 MB) |
| Baseline Manifest | `neural_core/data/baseline_manifest.json` | Model artifact SHA256, parameter count, architecture config |

---

## 5. Critical Findings

### 5.1 UNVERIFIED: Primary Corpus Copyright Status
> **REQUIRES LEGAL/PROVENANCE VERIFICATION**
> 
> The primary training corpus (`full_corpus.txt`, 33.4 MB) has **no documented original sources**. The author declares it comprises "opere scaricabili senza copyright / public domain" but the repository contains:
> - No source URLs
> - No author attributions
> - No publication dates
> - No public domain verification records
> - No license documentation
> 
> The content appears to be generated/curated synthetic Italian text, not recognizable public domain works.

### 5.2 VERIFIED: Literature Corpus Public Domain
> The `cleaned_literature.txt` contains works by Niccolò Machiavelli (d. 1527), which are unequivocally public domain. However, the specific digital edition source is not documented.

### 5.3 VERIFIED: Dialogue Corpus Internal
> The `jarvis_dialogue.txt` is internally generated synthetic dialogue data — no third-party copyright concerns.

### 5.4 Training Nature
> The MicroTransformer training is **experimental / proof-of-architecture training** on a small corpus (33.4 MB / ~5M tokens). This is not production-scale language model training.

---

## 6. Recommendations for Due Diligence

| Action | Priority | Description |
|---|---|---|
| **Source audit of full_corpus.txt** | CRITICAL | Trace each text segment to verifiable public domain source or obtain licenses |
| **Document digital edition of Machiavelli works** | HIGH | Record Project Gutenberg / LiberLiber / other source URL for cleaned_literature.txt |
| **Legal review of dataset composition** | CRITICAL | Engage IP counsel to assess copyright risk of training on unverified corpus |
| **Retrain on verified corpus if needed** | CONTINGENT | If full_corpus.txt contains copyrighted material, retrain on fully licensed data |
| **Add source metadata to provenance_manifest.json** | HIGH | Extend manifest to include source_url, author, publication_date, license fields |

---

## 7. Summary Table for Acquisition Review

| Dataset | Size | Source Verified | Copyright Status | Risk Level | Action Required |
|---|---|---|---|---|---|
| full_corpus.txt | 33.4 MB | **NO** | **UNVERIFIED** | **CRITICAL** | Legal audit required |
| cleaned_literature.txt | 5.7 MB | **PARTIAL** (content matches PD works) | **PUBLIC DOMAIN** (likely) | LOW | Document edition source |
| train/val/test splits | ~6 MB | Derived | **INHERITS RISK** | **CRITICAL** | Depends on full_corpus audit |
| jarvis_dialogue.txt | 15 KB | **YES** (internal) | **INTERNAL** | LOW | None |
| corpus_addestramento.txt | 1.5 KB | **NO** | **UNVERIFIED** | MEDIUM | Document or remove |

---

*This document is part of the Technical Acquisition Package.*