# JARVIS 2.0 — PHASE 2A GPU BACKEND REPORT

**Scope:** GPU backend implementation + CPU↔GPU numerical parity + benchmark.
**Status:** PRE-SCALING. No model scaling, no full training, no re-baseline.
**Date:** 2026-08-26

---

## 1. Environment
| Item | Value |
|------|-------|
| OS | macOS 14.5 (Darwin 23.5.0) |
| Arch | arm64 — Apple Silicon (M-series) |
| Python | 3.14.3 |
| NumPy | 2.4.2 (pre-existing, untouched) |
| PyTorch | 2.13.0 (installed this phase) |
| MPS backend | built = **True**, available = **True** |
| MPS float64 | **NOT supported** (TypeError) → confirms authorized dtype policy |

## 2. Dependencies installed
- `torch==2.13.0` (CPU/MPS macOS arm64 wheel, cp314).
- Transitive: filelock, fsspec, jinja2, networkx, sympy, mpmath, MarkupSafe.
- No pre-existing package was removed or downgraded.
- Reversible: `python3 -m pip uninstall torch` (plus its transitive deps).
- Documented: Python 3.14.3 / NumPy 2.4.2 / macOS 14.5 / arm64 / Apple Silicon.

## 3. Backend implemented
- **`neural_core/torch_model.py`** — `TorchMicroTransformer`: PyTorch/MPS port of the
  CPU reference, **float32**, same architecture & math.
  - GELU uses the **same tanh approximation** as the NumPy reference (not PyTorch's
    default erf GELU) to preserve model math.
  - LayerNorm computed manually with `eps=1e-6` (population variance) to match
    `neural_core.network.LayerNorm` exactly.
  - Parameter ordering identical to `MicroTransformer.get_parameters()` so CPU
    float64 weights transfer losslessly (cast to float32).
- **`neural_core/backend.py`** — backend abstraction: `NeuralBackend` interface,
  `NumpyBackend` (CPU reference), `TorchBackend` (GPU), `get_backend()` factory.

## 4. Files modified / added
**Added (new, isolated):**
- `neural_core/torch_model.py` — GPU model.
- `neural_core/backend.py` — backend abstraction + factory.
- `neural_core/gpu_benchmark.py` — reproducible CPU vs GPU micro-benchmark.
- `TESTS/test_neural_core/test_gpu_parity.py` — 7 parity/step tests.

**NOT modified (baseline protected):**
`network.py`, `model.py`, `optimizer.py`, `loss.py`, `embeddings.py`,
`tokenizer.py`, `checkpoint.py`, `trainer.py`, `model_interface.py`, and all
baseline/working checkpoint artifacts.

## 5. Backend architecture
```
        INPUT
          ↓
   NeuralModelInterface
          ↓
   Backend abstraction (neural_core/backend.py)
     ├── NumpyBackend  → MicroTransformer   (CPU, float64, REFERENCE)
     └── TorchBackend  → TorchMicroTransformer (GPU/MPS, float32)
```
CPU NumPy reference remains the authoritative baseline and is never invoked by the GPU path.

## 6. CPU↔GPU parity results (same weights, same inputs)
Measured on the real trained weights (`model.pkl`), batch 4 × seq 64, dropout=0.

| Component | Metric | Value | Tolerance | Verdict |
|-----------|--------|-------|-----------|---------|
| embedding / attn / ffn / layernorm / final_norm | max abs diff | **~1e-6** | <1e-3 | PASS |
| logits | max abs diff | **9.1e-6** | <1e-3 | PASS |
| loss (cross-entropy) | abs diff | **5.4e-5** | <1e-2 | PASS |
| gradients (37 tensors) | max abs diff | **1.9e-7** | <1e-4 | PASS |
| gradients | sign mismatch among |g|>1e-3 | **0 / 288,795** | — | PASS |

## 7. Loss comparison
CPU float64 = **13.439567**, GPU float32 = **13.439621**, Δ = 5.4e-5.

## 8. Gradient comparison
Max absolute gradient difference = **1.87e-7**. Among the 288,795 elements with
meaningful gradient magnitude (|g|>1e-3, 11.7% of params), the CPU/GPU sign
mismatch is **zero**. Gradients are numerically equivalent.

## 9. Optimizer comparison (single Adam step)
- CPU Adam (`neural_core.optimizer.Adam`) vs GPU `torch.optim.Adam`, both
  lr=1e-3, β=(0.9,0.999), ε=1e-8, wd=0.
- Adam's first-step update is `lr·sign(g)`, independent of magnitude. Elements
  with **near-zero gradients** (~float-rounding) can flip sign between float64
  (CPU) and float32 (GPU). This is a **bounded artifact ≤ 2·lr = 2e-3** and is
  numerically irrelevant (those gradients are effectively zero).
- Result: max update divergence = **2.0e-3** (exactly the 2·lr bound), mean
  update diff = **1.0e-4**, only ~4.98% of elements exceed lr (all near-zero
  gradient), **0 unbounded divergence**.
- Interpretation: optimizer parity PASS with a documented, bounded,
  float32/float64-boundary artifact.

## 10. Benchmark (micro; no full training)
batch 8 × seq 128 (1,024 tokens/call), 10 repeats + 3 warmup.

| Operation | CPU (NumPy f64) | GPU (MPS f32) | Speedup | Throughput |
|-----------|-----------------|---------------|---------|------------|
| forward | 49.55 ms | 2.00 ms | **24.8×** | — |
| train step | 214.51 ms | 32.04 ms | **6.7×** | CPU 4,774 tok/s → GPU 31,961 tok/s |

## 11. GPU memory usage
- Allocated after training step: **128.9 MB** (forward +0.1 MB, step +118.6 MB).
- Reported via `torch.mps.current_allocated_memory()`. MPS does not expose a
  simple peak-API; "allocated" is the resident GPU memory for the model+activations.

## 12. Checkpoint compatibility
- **Baseline backup** `backup_20260826_103727/` SHA256 matches
  `baseline_manifest.json` **exactly** (all 5 artifacts) → baseline untouched.
- **GPU checkpoint round-trip**: parameters serialized via pickle and reloaded
  reproduce forward with **0.0** difference; parameter count matches (2,472,846).
- Working checkpoints (model/training_state/best) remain distinct from baseline
  per the Phase-2 authorization; they are NOT modified by this phase.
- GPU checkpoints are distinguishable (float32, explicit parameter ordering).

## 13. Baseline integrity
`baseline_id = baseline_20260826_103727` — all hashes verified against manifest:
`d169a678…ba0d`, `797d8e18…d0a92`, `c1339fd9…eb1a`, `0de73c1d…fa840`,
`a60f9a9a…472e6`. **Immutable and recoverable.**

## 14. Test results
- Parity suite `TESTS/test_neural_core/test_gpu_parity.py`: **7 passed**
  (param count, components, logits, loss, gradients, optimizer step, training-step
  trajectory).
- Full `TESTS/test_neural_core`: **307 passed** (300 pre-existing + 7 new) — no
  regression.
- Benchmark executes cleanly; no full training performed.

## 15. Open issues / documented limitations
1. **float64 vs float32**: CPU baseline is float64; GPU is float32. Parity is
   numerical (within documented tolerances), never bitwise — by design
   (authorization Option A). Not a baseline regression.
2. **Dropout RNG**: parity tests use dropout=0. Dropout>0 training parity would
   require matching NumPy vs PyTorch RNG streams — not in scope; documented.
3. **Adam near-zero-gradient sign artifact**: bounded ≤2·lr, numerically
   irrelevant; documented in §9.
4. **Tokenizer**: vocab 8590 vs configured `max_vocab_size=8000` persists
   (unchanged, per authorization — no automatic tokenizer modification).
5. **MPS determinism**: exact GPU bit-reproducibility across runs is not
   guaranteed by MPS (float32 reduction order); CPU↔GPU parity holds within
   tolerance.

## 16. GPU PARITY GATE — VERDICT: ✅ PASS
- [x] GPU forward PASS
- [x] GPU backward PASS
- [x] CPU↔GPU numerical parity PASS (components, logits, loss, gradients)
- [x] GPU training-step PASS (loss tracks CPU, both decrease)
- [x] checkpoint compatibility PASS (baseline intact, GPU round-trips)
- [x] no baseline corruption (hashes unchanged)
- [x] benchmark completed

All gate conditions satisfied.

---

## STOP — NEXT PHASE BLOCKED BY AUTHORIZATION
Per the mandated sequence, Phase 2A ends here. **Not performed:** model scaling
(BASELINE→SMALL→MEDIUM→LARGE), full training, re-baseline, tokenizer change,
dataset regeneration.

**Recommendation for the first real training (Phase 2B, pending authorization):**
proceed with the scaling matrix (BASELINE / SMALL / MEDIUM / LARGE) using the
existing `TorchMicroTransformer` (float32) and the backend factory; keep the
float64 CPU reference as the parity oracle; validate each scaling step with the
parity suite before advancing.
