# JARVIS 2.0 — PHASE 2B MODEL SCALING PREPARATION REPORT

*Generated: 2026-08-26 23:19:44*
*Machine: Python 3.14.3 / Darwin 23.5.0 / arm64*

> PHASE 2B = PREPARE THE SCALING. No full training. No final model selection. No baseline change. Architecture preserved.

## 1. Current Architecture

Decoder-only micro-transformer (autoregressive next-token prediction):

- causal self-attention (multi-head)
- pre-norm Transformer blocks with residual connections
- manual LayerNorm (population variance, `eps=1e-6`)
- tanh-approximated GELU (same formula as NumPy reference)
- sinusoidal positional encoding
- FFN with GELU
- separate embedding and output projection (NO weight tying)
- CrossEntropyLoss (next-token), Adam optimizer

Backends (math-identical, not architecturally different):

- Reference: NumPy CPU, float64 (authoritative, immutable)
- Backend:   PyTorch, float32 (MPS locally; CUDA target on Kaggle)

## 2. Preserved Architectural Constraints

- decoder-only structure (no encoder/decoder, no MoE, no RNN/CNN)
- causal masking and autoregressive objective
- tanh-based GELU (NOT replaced by erf GELU or other variant)
- manual LayerNorm with eps=1e-6 (NO RMSNorm, NO alternative)
- embedding separate from output projection (NO weight tying)
- same dropout, Adam, CrossEntropyLoss, gradient-accumulation semantics
- tokenizer unchanged; vocabulary kept at the real value 8590
- NO RoPE / ALiBi / FlashAttention / mixed precision introduced

## 3. Model Factory

`neural_core.model_factory` builds any model from a validated `ModelConfig` without touching the reference backends:

```python
from neural_core.model_config import ModelConfig
from neural_core import model_factory
cfg = ModelConfig(vocab_size=8590, d_model=128, num_heads=4,
                  d_ff=256, num_layers=2, max_len=256, dropout=0.1)
model = model_factory.build_model(cfg, backend='torch', device='mps')
```

`ModelConfig` validates `d_model % num_heads == 0`, positive dims, dropout in `[0,1)`, and a supported backend/device. It is fully serializable (`to_dict` / `from_dict` / `to_json` / `from_json`).

## 4. Scaling Ladder

| Tier | Target | d_model | heads | d_ff | layers | context |
|------|--------|---------|-------|------|---------|---------|
| CURRENT | ~2.5M | 128 | 4 | 256 | 2 | 256 |
| SMALL | ~10M | 256 | 8 | 512 | 11 | 256 |
| MEDIUM | ~25M | 384 | 12 | 768 | 16 | 256 |
| LARGE | ~50M | 512 | 16 | 1024 | 20 | 256 |
| OPTIONAL | ~100M | 768 | 16 | 1536 | 18 | 256 |

Targets are approximate. Width/depth/FFN/context vary; the fundamental math is unchanged. `num_heads` never changes the parameter count.

## 5. Candidate Configurations & Parameter Counts

Theoretical vs material parameter counts. A mismatch is a HARD STOP.

| Tier | Theoretical | Material (torch) | Match | Embedding | Attention | FFN | LayerNorm | Output |
|------|-------------|------------------|-------|-----------|-----------|-----|-----------|--------|
| CURRENT | 2,472,846 | 2,472,846 | YES | 1,099,520 | 132,096 | 131,840 | 1,280 | 1,108,110 |
| SMALL | 10,205,326 | 10,205,326 | YES | 2,199,040 | 2,894,848 | 2,892,032 | 11,776 | 2,207,630 |
| MEDIUM | 25,548,430 | 25,548,430 | YES | 3,298,560 | 9,461,760 | 9,455,616 | 25,344 | 3,307,150 |
| LARGE | 50,861,454 | 50,861,454 | YES | 4,398,080 | 21,012,480 | 21,002,240 | 41,984 | 4,406,670 |
| OPTIONAL | 98,291,086 | 98,291,086 | YES | 6,597,120 | 42,522,624 | 42,508,800 | 56,832 | 6,605,710 |


## 6. Memory Estimates (THEORETICAL)

Theoretical estimate at batch=8, seq=256 (float32, 4 bytes/param). This is an analytical estimate, NOT a measurement.

| Tier | Params | Grad | Adam m1 | Adam m2 | Activations | Attn WS | Total Est. |
|------|--------|-------|---------|---------|-------------|---------|------------|
| CURRENT | 18.9 MB | 18.9 MB | 18.9 MB | 18.9 MB | 356.4 MB | 32.0 MB | 463.9 MB |
| SMALL | 77.9 MB | 77.9 MB | 77.9 MB | 77.9 MB | 1,200.4 MB | 352.0 MB | 1,863.9 MB |
| MEDIUM | 194.9 MB | 194.9 MB | 194.9 MB | 194.9 MB | 2,296.4 MB | 768.0 MB | 3,844.1 MB |
| LARGE | 388.0 MB | 388.0 MB | 388.0 MB | 388.0 MB | 3,644.4 MB | 1,280.0 MB | 6,476.6 MB |
| OPTIONAL | 749.9 MB | 749.9 MB | 749.9 MB | 749.9 MB | 4,252.4 MB | 1,152.0 MB | 8,404.0 MB |

`Params+Grad+Adam(m1+m2)` dominate for large models; activations and the attention workspace scale with batch/seq. Overhead (framework reserve) is environment-specific and excluded here.

## 7. Context Length Analysis (256 / 512 / 1024)

Parameter count is identical across context lengths; only activation and attention memory grow. Base = CURRENT width/depth (batch=8).

| Context | Params | Activation Mem | Attention WS | Total Est. |
|---------|--------|----------------|-------------|------------|
| 256 | 2,472,846 | 356.4 MB | 32.0 MB | 463.9 MB |
| 512 | 2,472,846 | 776.9 MB | 128.0 MB | 980.3 MB |
| 1024 | 2,472,846 | 1,809.8 MB | 512.0 MB | 2,397.2 MB |

Context length is NOT changed from the 256 default without authorization; 1024 is analyzed here only.

## 8. Measured GPU Memory

Measured on device = `mps` (allocated-memory probe, smoke shapes b=2/s=32).

NOTE: measured (smoke shapes b=2/s=32) and theoretical (b=8/s=256) use DIFFERENT batch/sequence shapes, so they are NOT directly comparable. Theoretical is an analytical upper bound; measured reflects the actual allocator on this device.

| Tier | Base | After Forward | After Step | Step Δ |
|------|------|---------------|------------|--------|
| CURRENT | 9.6 MB | 11.7 MB | 42.1 MB | 32.5 MB |
| SMALL | 39.2 MB | 41.3 MB | 176.8 MB | 137.6 MB |
| MEDIUM | 101.6 MB | 103.7 MB | 530.6 MB | 429.1 MB |
| LARGE | 203.9 MB | 206.1 MB | 1,022.3 MB | 818.4 MB |
| OPTIONAL | 384.7 MB | 386.9 MB | 2,062.5 MB | 1,677.8 MB |

## 9. Benchmark (micro, no full training)

Forward / training-step wall time and throughput. CPU = NumPy float64 reference (CURRENT only, as an anchor). MPS = PyTorch float32. CUDA = NOT MEASURED LOCALLY.

### CPU (NumPy float64 reference, CURRENT config)

- baseline param count == 2,472,846 : YES
- forward : 13.159 ms
- train step : 74.381 ms (3442 tok/s)

### MPS (PyTorch float32)

| Tier | Forward (ms) | Train Step (ms) | Tok/s (step) |
|------|-------------|-----------------|--------------|
| CURRENT | 1.15 | 6.80 | 37627 |
| SMALL | 5.44 | 54.12 | 4730 |
| MEDIUM | 29.99 | 120.52 | 2124 |
| LARGE | 50.62 | 215.00 | 1191 |
| OPTIONAL | 75.49 | 367.73 | 696 |

> CUDA benchmark: NOT MEASURED LOCALLY (no CUDA device). Will be measured on Kaggle before full training.

## 10. Device Support

- cpu  : supported (NumPy float64 reference)
- mps  : available = True (PyTorch float32)
- cuda : available = False (target for Kaggle)

`neural_core.device.select_device` resolves a preferred device with safe fallback: cuda -> mps -> cpu. CUDA absence is never treated as failure.

## 11. CUDA Readiness

- Local CUDA available: False
- PyTorch CUDA backend supported via the same `TorchMicroTransformer` (device='cuda'); no CUDA-specific dependency blocks CPU/MPS use.
- On Kaggle (NVIDIA CUDA) the same factory path applies: `build_model(cfg, backend='torch', device='cuda')`.

## 12. Kaggle Readiness

Preparation delivered (no remote run executed):

- serializable `ModelConfig` + `Experiment` reproducible structure
- `neural_core.dataset_verification` for checksum / token-count / tokenizer / vocabulary verification on the remote machine
- device selection with explicit fallback (cpu/mps/cuda)
- theoretical GPU memory estimates to size the Kaggle GPU
- backward-compatible checkpoint metadata (model_class, config, tokenizer/dataset identity, param count, seed)
- requirements: numpy, torch, pytest (torch already present locally)

Startup sequence on Kaggle (planned, not executed):
```
python -c "import torch; assert torch.cuda.is_available()"
python phase2b_report.py   # re-run smoke tests + estimates on CUDA
# then, in a SEPARATE Phase 3 run, full training of the chosen config
```

## 13. Checkpoint Compatibility

`neural_core.checkpoint_metadata.build_checkpoint_record` extends the historical format ADDITIVELY (`checkpoint_metadata` block). Legacy loaders still read `model_name` + `parameters`. Baseline format in `neural_core/data/backup_20260826_103727/` is UNCHANGED.

## 14. Tokenizer Status

- mode: whitespace
- real vocabulary size: 8590 (kept for scaling)
- declared max_vocab_size: 8000
- documented discrepancy (declared 8000 vs fitted 8590): YES — NOT auto-corrected

## 15. Dataset Status

- train.txt: sha256=49416a394642d9b4… tokens=4,224,043 status=OK
- full_corpus.txt: sha256=a03fd7166e766c4d… tokens=5,064,882 status=OK
- Dataset NOT regenerated, NOT cleaned, NOT modified. Verification mechanism only (checksum + token count + vocab).

## 16. Test Results

- Pre-existing suite: 307+ tests (unchanged target; must remain PASS).
- Phase 2B new tests added (see TESTS/test_neural_core/test_phase2b_*.py): model factory, invalid configurations, parameter count, scaling configs, device selection, checkpoint config, experiment config, memory estimation.
- Smoke tests executed for all 5 ladder tiers: construction, param count, forward, loss, backward, single step, checkpoint round-trip.

## 17. Baseline Integrity

- backup_dir: /Users/riccardobronzuoli/Documents/Jarvis 2.0 completo/neural_core/data/backup_20260826_103727
- all artifacts byte-identical (SHA256 == manifest): YES
  - model.pkl: match=True size_ok=True
  - training_state.pkl: match=True size_ok=True
  - best_model.json: match=True size_ok=True
  - tokenizer.json: match=True size_ok=True
  - tokenizer.pkl: match=True size_ok=True

## 18. Open Issues

- Tokenizer declares max_vocab_size=8000 but fitted vocab=8590; kept as-is (documented, not corrected).
- CUDA memory/benchmark NOT measured locally (no CUDA device); must be measured on Kaggle before full training.
- 1024 context length NOT adopted as default (requires authorization); analyzed separately in §18 context table.
- Final target model (SMALL/MEDIUM/LARGE/OPTIONAL) NOT selected here (Phase 3 decision).

## 19. Recommendation for Next Target

Phase 2B preparation is COMPLETE. Recommended next step (Phase 3 decision, NOT executed here):

1. Re-run this report on the Kaggle CUDA node to obtain MEASURED memory and benchmark for the candidate tiers.
2. Pick the largest tier that fits the target GPU with margin (start from **SMALL (~10M)** for a first successful CUDA run, then **MEDIUM (~25M)**).
3. Keep context length = 256 unless §7 shows headroom and authorization is granted for 512/1024.
4. Train on Kaggle via the reproducible `Experiment` config; preserve the CPU float64 baseline as reference.

---
END OF PHASE 2B REPORT. No full training performed. No architecture modified. Baseline protected.
