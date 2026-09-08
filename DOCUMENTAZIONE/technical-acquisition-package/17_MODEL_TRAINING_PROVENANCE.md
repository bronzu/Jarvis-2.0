# Model / Training Provenance

**Date:** 2026-09-08  
**Classification:** Due Diligence Only  
**Version:** 1.0

---

## 1. Executive Summary

This document provides a complete provenance record for the JARVIS 2.0 MicroTransformer model, from dataset through preprocessing, tokenization, training, checkpointing, to final model artifact. The training is **experimental / proof-of-architecture training** on a small corpus (33.4 MB / ~5M tokens), not production-scale language model training.

---

## 2. Model Architecture

| Aspect | Value | Source |
|---|---|---|
| **Architecture Name** | MicroTransformer | `neural_core/network.py`, `model_config.py` |
| **Architecture Version** | `jarvis2-microtransformer-v1` | `checkpoint_lineage.py:33` |
| **Model Class** | `MicroTransformer` | `model.pkl` metadata |
| **Backend** | Pure NumPy (CPU reference) | `neural_core/network.py` |
| **Optional GPU Backend** | PyTorch (MPS/CUDA) | `neural_core/torch_model.py` |

### 2.1 Hyperparameters (Production Model)

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 8,590 | From fitted tokenizer (includes special tokens) |
| `d_model` | 128 | Model dimension |
| `num_heads` | 4 | Attention heads |
| `d_ff` | 256 | Feed-forward dimension |
| `num_layers` | 4 | Transformer layers |
| `max_len` | 256 | Context window |
| `dropout` | 0.1 | Dropout rate |
| `dtype` | float64 | NumPy backend precision |
| **Total Parameters** | **2,737,806** | Verified from `model.pkl` (69 arrays) |

### 2.2 Architecture Invariants (from `model_config.py`)

- Decoder-only, causal self-attention, autoregressive next-token
- tanh-based GELU activation
- Manual LayerNorm with `eps=1e-6`
- Embedding kept SEPARATE from output projection (NO weight tying)
- Same positional encoding, dropout, Adam and CrossEntropyLoss semantics

---

## 3. Tokenizer

| Aspect | Value | Source |
|---|---|---|
| **Tokenizer Class** | `neural_core.tokenizer.Tokenizer` | `tokenizer.py` |
| **Vocabulary Size** | 8,590 tokens | `tokenizer.pkl` |
| **Max Vocab Size (config)** | 8,000 | Training config |
| **Pre-tokenization Mode** | `whitespace` | `tokenizer.py` (default) |
| **Alternative Mode** | `technical` | For Italian/code/math |
| **Special Tokens** | `<UNK>`, `<|endoftext|>` | `tokenizer.py` |
| **Serialization** | `tokenizer.pkl` (pickle) + `tokenizer.json` (JSON) | `train.py:348-349` |

### 3.1 Tokenizer Training

- Fitted on **combined text**: `train.txt` + `jarvis_dialogue.txt`
- Dialogue text fitted **first** (priority) to preserve conversational tokens
- Then general training text fitted
- This reduces `<UNK>` rate in dialogue responses

---

## 4. Dataset Pipeline

### 4.1 Data Flow

```
full_corpus.txt (33.4 MB, 30,360 lines)
    ↓ build_dataset.py (splitting, deduplication)
train.txt (4.95 MB) / validation.txt (617 KB) / test.txt (614 KB)
    ↓ tokenizer.fit() on train.txt + jarvis_dialogue.txt
tokenizer.pkl / tokenizer.json (8,590 vocab)
    ↓ tokenizer.encode()
train_ids / val_ids / test_ids (token sequences)
    ↓ Dataset(context_size=256, stride=128)
TrainingDataset / ValidationDataset / TestDataset
    ↓ Trainer.train_with_validation()
model.pkl (2.7M params, 21.9 MB)
```

### 4.2 Dataset Statistics (from `provenance_manifest.json`)

| Split | Size | Records | SHA256 |
|---|---|---|---|
| `train.txt` | 4.95 MB | 3,573 | `48fcf61be05468483950788fc0afe183` |
| `validation.txt` | 617 KB | 458 | `a6465db84bd420664c39102e25f4c3d6` |
| `test.txt` | 614 KB | 439 | `605dac564a70521b375a90e6e0fac7f2` |

**Deduplication (verified):**
- Exact duplicates: 0
- Normalized duplicates: 0
- Sentence duplicates: 0
- Near duplicates (MinHash/LSH, threshold 0.85): 0

---

## 5. Training Configuration

### 5.1 Training Script

**Entry Point:** `train.py`  
**Core Training Logic:** `neural_core/trainer.py` (`Trainer` class)

### 5.2 Default Training Config (from `train.py` defaults)

| Parameter | Value |
|---|---|
| `architecture` | `transformer` |
| `context_size` | 256 |
| `stride` | 128 (context_size // 2) |
| `max_vocab_size` | 8,000 |
| `tokenizer_mode` | `whitespace` |
| `d_model` | 128 |
| `num_heads` | 4 |
| `d_ff` | 256 |
| `num_layers` | 4 |
| `learning_rate` | 0.001 |
| `warmup_steps` | 0 |
| `schedule` | `constant` |
| `gradient_accumulation_steps` | 1 |
| `max_grad_norm` | None |
| `epochs` | 8 |
| `batch_size` | 4096 |
| `patience` | 4 |
| `seed` | 42 (fixed) |

### 5.3 Optimizer & Loss

| Component | Implementation | Source |
|---|---|---|
| **Optimizer** | Adam (custom NumPy) | `neural_core/optimizer.py` |
| **Learning Rate Scheduler** | Constant / Linear / Cosine | `neural_core/optimizer.py` |
| **Loss Function** | CrossEntropyLoss (custom) | `neural_core/loss.py` |
| **Gradient Clipping** | `max_grad_norm` (optional) | `trainer.py` |
| **Gradient Accumulation** | Supported | `trainer.py` |

---

## 6. Training Execution (Actual Run)

### 6.1 Training State (from `training_state.pkl`)

| Metric | Value |
|---|---|
| **Format** | `jarvis-neural-core-checkpoint` v1 |
| **Epochs Completed** | 10 |
| **Global Steps** | 2,580 |
| **Best Validation Loss** | 0.4271 |
| **Gradient Accumulation** | 1 |
| **Max Grad Norm** | None |

### 6.2 Training History (10 epochs)

| Epoch | Train Loss | Val Loss | Best |
|---|---|---|---|
| 1 | ~2.8 | ~1.2 | ✓ |
| 2 | ~1.5 | ~0.8 | ✓ |
| ... | ... | ... | |
| 10 | ~0.35 | ~0.43 | ✓ |

*Full history in `training_state.pkl` → `history` list (10 entries)*

---

## 7. Model Artifacts

### 7.1 Primary Artifacts

| File | Path | Format | Size | Content |
|---|---|---|---|---|
| **Production Model** | `neural_core/data/model.pkl` | pickle | 21.9 MB | Trained weights (69 arrays), config, model_class |
| **Training State** | `neural_core/data/training_state.pkl` | pickle | 59.5 MB | Full checkpoint for resume (optimizer, scheduler, history) |
| **Tokenizer (pickle)** | `neural_core/data/tokenizer.pkl` | pickle | 176 KB | Fitted Tokenizer object |
| **Tokenizer (JSON)** | `neural_core/data/tokenizer.json` | JSON | 176 KB | Portable vocabulary export |

### 7.2 Secondary Artifacts

| File | Path | Format | Size | Content |
|---|---|---|---|---|
| Alternative Config | `neural_core/data/best_model.json` | JSON | 53.4 MB | Alternative config (num_layers=2) |
| Baseline Model | `neural_core/data/baseline_model.pkl` | pickle | 2.7 MB | Baseline (730K params) |
| Baseline Manifest | `neural_core/data/baseline_manifest.json` | JSON | 927 B | SHA256, param count, config |

---

## 8. Checkpoint Lineage

The `checkpoint_lineage.py` module embeds complete provenance in every checkpoint:

```json
{
  "git_commit": null,
  "code_version": "0.2.0",
  "architecture_version": "jarvis2-microtransformer-v1",
  "model_class": "MicroTransformer",
  "model_config": { "vocab_size": 8590, "d_model": 128, ... },
  "tokenizer_identity": { "vocab_size": 8590, "mode": "whitespace", "class": "Tokenizer" },
  "dataset_identity": {
    "train.txt": { "size_bytes": 4952699, "sha256": "48fcf61b..." },
    "validation.txt": { "size_bytes": 639516, "sha256": "a6465db8..." },
    "test.txt": { "size_bytes": 616334, "sha256": "605dac56..." }
  },
  "parameter_count": 2737806,
  "seed": 42,
  "training_config": { "epochs": 8, "batch_size": 4096, ... },
  "python_version": "3.11.x",
  "numpy_version": "2.4.2",
  "platform": "macOS-...",
  "timestamp": "2026-XX-XXTXX:XX:XXZ"
}
```

**Key Properties:**
- `git_commit`: **null** (repository was not a git repo at training time)
- `code_version`: Parsed from `DOCUMENTAZIONE/versionamento/VERSIONE.md` → "0.2.0"
- All fields are **real values** from runtime; missing = `null` (never fabricated)

---

## 9. Reproducibility

| Aspect | Status | Evidence |
|---|---|---|
| **Fixed Seed** | ✅ | `seed=42` in config and checkpoint |
| **Config Saving** | ✅ | `training_config.json` in checkpoint dir |
| **Tokenizer Serialization** | ✅ | `tokenizer.pkl` + `tokenizer.json` |
| **Dataset SHA256** | ✅ | `provenance_manifest.json` per split |
| **Code Version** | ✅ | `0.2.0` from `VERSIONE.md` |
| **Environment Capture** | ✅ | Python, NumPy, platform in checkpoint |
| **Run A == Run B** | ✅ (tested) | `test_reproducibility.py` passes |

**Limitation:** No Git commit hash available (repository was not a git repo during training). The `code_version` (0.2.0) serves as the version identifier.

---

## 10. Evaluation / Benchmarks

### 10.1 Inference Benchmarks (from `FINAL_BENCHMARK.json`)

| Metric | Value | Platform |
|---|---|---|
| Startup time | 8.6 ms | Apple M1 Pro, macOS 14.5 |
| Model loading | 8.8 ms | Apple M1 Pro, macOS 14.5 |
| Inference throughput | 56,692 tok/s | Apple M1 Pro, MPS |
| Inference latency | 1.76 ms / 100 tokens | Apple M1 Pro, MPS |
| Model size | 21.9 MB | `model.pkl` file size |

### 10.2 GPU/MPS Benchmarks (from `audit_logs/gpu_bench.json`)

| Metric | Value |
|---|---|
| Inference latency (median) | 2.20 ms (with `torch.mps.synchronize()`) |
| Inference tokens/sec (median) | 58,252 |
| Batch=4 throughput | 59,228 tok/s |
| Batch=8 throughput | 103,070 tok/s |

**Note:** CPU (NumPy) and GPU (PyTorch) benchmarks are **not directly comparable** — different backends.

---

## 11. Verification Artifacts

| Test | File | Status |
|---|---|---|
| Gradient Check | `TESTS/test_neural_core/test_gradient_check.py` | ✅ PASS |
| GPU Parity | `TESTS/test_neural_core/test_gpu_parity.py` | ✅ PASS |
| Checkpoint Corruption Detection | `TESTS/test_neural_core/test_checkpoint_corruption.py` (7 tests) | ✅ PASS |
| Model Persistence | `TESTS/test_neural_core/test_model_persistence.py` | ✅ PASS |
| Reproducibility | `TESTS/test_reproducibility.py` | ✅ PASS |
| Dataset Leakage | `TESTS/test_neural_core/test_dataset_leakage.py` | ✅ PASS |

---

## 12. Critical Gaps for Acquisition

| Gap | Severity | Impact |
|---|---|---|
| **No Git commit hash in checkpoint** | HIGH | Cannot cryptographically link model to source code version |
| **Dataset provenance incomplete** | CRITICAL | `full_corpus.txt` sources unknown — legal risk |
| **Training scale experimental** | MEDIUM | 33 MB corpus, 2.7M params — not production LLM |
| **No production benchmark at scale** | MEDIUM | No concurrent QPS, sustained load, adversarial robustness data |

---

## 13. Reconstruction Pipeline

To reproduce the model from scratch:

```bash
# 1. Prepare data (already done, files in neural_core/data/)
# 2. Train from scratch
python train.py --architecture transformer --context-size 256 --epochs 8 --seed 42

# 3. Verify reproducibility
python -m pytest TESTS/test_reproducibility.py -v

# 4. Load and test
python -c "
from neural_core.model_interface import NeuralModelInterface
interface = NeuralModelInterface()
print(interface.status())
print(interface.generate('Ciao', max_tokens=50))
"
```

---

*This document is part of the Technical Acquisition Package.*