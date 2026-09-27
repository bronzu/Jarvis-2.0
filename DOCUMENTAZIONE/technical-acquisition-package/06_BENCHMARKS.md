# Benchmarks

## 1. Available Benchmarks

### 1.1 Neural Core Performance (`FINAL_BENCHMARK.json`)

| Metric | Value | Platform | Notes |
|---|---|---|---|
| Startup time | 8.6 ms | Apple M1 Pro, macOS 14.5 | Median, 5 iterations |
| Model loading | 8.8 ms | Apple M1 Pro, macOS 14.5 | Median, 5 iterations |
| Inference throughput | 56,692 tokens/sec | Apple M1 Pro, MPS | Median, 100 tokens generated |
| Inference latency | 1.76 ms | Apple M1 Pro, MPS | For 100 tokens |
| Model size | 21.9 MB | — | `model.pkl` file size |
| Parameter count | 2,737,806 | — | Verified from model architecture |
| Array count | 69 | — | Number of parameter arrays |

### 1.2 GPU/MPS Benchmark (`audit_logs/gpu_bench.json`)

| Metric | Value | Notes |
|---|---|---|
| Inference latency median | 2.20 ms | With `torch.mps.synchronize()` |
| Inference tokens/sec median | 58,252 | With MPS sync |
| Batch=4 tokens/sec mean | 59,228 | Real GPU throughput |
| Batch=8 tokens/sec mean | 103,070 | Real GPU throughput |
| Synchronization overhead | 0.00027 ms | p50, warmup applied |
| MPS current_allocated | 60,160 bytes | Active tensor footprint |
| MPS driver_allocated | 19,562,496 bytes | Full MPS pool |

**Important distinction:** `final_benchmark.py` (CPU) reports ~155K tok/s @ batch=4 because it uses the NumPy MicroTransformer. `gpu_benchmark_extended.py` (GPU) reports ~57K tok/s @ batch=4 because the PyTorch model includes MPS dispatch + sync overhead. These are **not comparable** — different models and different paths.

### 1.3 CPU/RAM Benchmark (`audit_logs/final_bench.json`)
| Metric | Value | Notes |
|---|---|---|
| RSS | 244.69 MB | Process physical memory |
| Python tracemalloc peak | 66 MB | Python allocator peak |
| Virtual memory | None on macOS | `ru_maxrss` is bytes on Darwin |

### 1.4 Dataset Statistics
| Metric | Value | Notes |
|---|---|---|
| Train size | 7,144 lines | After filtering empty lines |
| Validation size | 914 lines | — |
| Test size | 876 lines | — |
| Exact duplicates | 0 | Within and across splits |
| Normalized duplicates | 0 | Lowercase + whitespace normalized |
| Sentence duplicates | 0 | Sentence-level deduplication |
| Near duplicates | 0 | MinHash/LSH, threshold 0.85 |
| Token count (corpus) | ~5.0M tokens | After deduplication |
| Vocabulary size | 8,590 | Fitted tokenizer |
| UNK rate (train) | 3.50% | Whitespace mode, max_vocab=8000 |

### 1.5 INT8 Quantization Benchmarks (`neural_core/quantization.py`)

| Layer | Precision | Original Size | INT8 Size | Compression Ratio | MSE | Max Abs Error |
|---|---|---|---|---|---|---|
| 64→128 | FP64→INT8 (per-channel) | 65,536 B | 8,704 B | 7.53x | 6.7e-7 | 1.9e-3 |
| 1024×1024 | FP64→INT8 (per-channel) | 8,396,800 B (8.2 MB) | 1,056,768 B (1.0 MB) | 7.95x | 6.0e-8 | 5.9e-4 |
| 1024×1024 | FP64→INT8 (per-tensor) | 8,396,800 B (8.2 MB) | 1,048,584 B (1.0 MB) | 8.01x | 1.2e-7 | 9.5e-4 |
| 1024×1024 | FP32→INT8 (per-channel) | 4,194,304 B (4.0 MB) | 1,056,768 B (1.0 MB) | 3.97x | — | — |

- Symmetric absmax quantization: S = max(|W|) / 127
- Per-channel (axis=1) achieves lower MSE than per-tensor for weights with varying column magnitudes
- STE gradient check passes (numeric vs analytic gradient, 1e-3 relative tolerance)
- All 35 quantization tests pass

### 1.6 Causal Linear Attention Benchmarks (`neural_core/network.py`)

| Seq Length | Linear Attention (ms) | Standard Attention (ms) | Speedup | Memory Complexity |
|---|---|---|---|---|
| 64 | 0.59 | 0.13 | 0.23x | O(N) |
| 128 | 1.19 | 0.35 | 0.30x | O(N) |
| 256 | 2.43 | 1.22 | 0.50x | O(N) |
| 512 | 6.66 | 5.38 | 0.81x | O(N) |
| 1,024 | 11.29 | 23.52 | 2.08x | O(N) |
| 2,048 | 24.66 | 97.70 | 3.96x | O(N) |
| 4,096 | 52.61 | 1607.30 | 30.55x | O(N) |

- **Model config**: d_model=64, 4 heads, batch=1, Apple M1 Pro, Python 3.14, NumPy
- **Crossover point**: ~512 tokens — below this, constant overhead dominates; above, O(N²) standard attention grows quadratically
- **Method**: ElU(x)+1 feature map with prefix-sum (cumsum) for causal accumulation — true O(N) complexity
- **Gradient check**: Analytic backward matches numeric gradient (1e-3 relative tolerance) on all 8 parameters (W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o)
- **All 20 linear attention tests pass**

### 1.8 Data Pipeline Benchmarks (`neural_core/data_pipeline.py`)

| Stage | Operation | Complexity | Test Count | Status |
|---|---|---|---|---|
| Semantic Dedup | MinHash (64 perms, 3-shingles) + LSH banding | O(N) signatures, O(candidates) verification | 6 deduplication tests | PASS |
| Quality Filter | Char repetition ratio, Shannon entropy, punctuation ratio | O(chars) per document | 10 quality filter tests | PASS |
| CoT Formatting | Code block regex, logical label detection, numbered list extraction | O(chars) per document | 8 CoT format tests | PASS |
| Full Pipeline | Dedup → Quality → CoT | O(N) end-to-end | 9 pipeline tests | PASS |

- Deduplication: reuses existing `minhash_lsh.find_near_duplicates` (threshold 0.85, 3-shingles, 64 permutations)
- Quality filters: accepts documents with entropy > 2.0 bits, char repetition < 30%, punctuation ratio > 2%, length > 50 chars
- CoT formatting: detects `thinking` blocks from code blocks, labeled sections, and numbered lists
- All 38 data pipeline tests pass; 449 total neural_core tests pass (no regressions)
- Platform: Apple M1 Pro, Python 3.14, NumPy-only

### 1.9 Checkpoint Lineage
| Field | Value | Notes |
|---|---|---|
| Code version | 0.2.0 | Parsed from `VERSIONE.md` |
| Git commit | null | Repository is not a git repo |
| Dataset identity | SHA256 per split | `provenance_manifest.json` |
| Tokenizer identity | class, vocab_size, mode | Serialized in checkpoint |
| Architecture version | jarvis2-microtransformer-v1 | — |
| Training config | Included | epochs, lr, batch_size, etc. |
| Seed | Included | Deterministic reproduction |
| Environment | python_version, numpy_version, platform, timestamp | — |

---

## 2. Missing Benchmarks

### 2.1 Recommended Pre-Acquisition Benchmarks

| Benchmark | Why Needed | How to Measure |
|---|---|---|
| **Inference latency (production path)** | Current numbers are for isolated model forward pass, not full pipeline | Measure `ExecutiveRouter.generate()` end-to-end latency |
| **Throughput under concurrent load** | Single-process only; no data on parallel request handling | Run multiple concurrent requests, measure QPS |
| **Training throughput (full corpus)** | Current benchmarks use synthetic/small data | Time full training run on production corpus |
| **Memory usage under load** | Only MPS memory measured; CPU/RAM not profiled | Use `memory_profiler` or `tracemalloc` during sustained operation |
| **Tool execution latency** | No timing for filesystem, terminal, python runner tools | Measure each tool's execution time distribution |
| **Security benchmark (throughput)** | No data on permission check overhead | Measure `PermissionGuard.validate_command()` latency at scale |
| **Recovery time** | No measurement of recovery strategy execution time | Time retry → reparam → different_tool → different_plan → escalation |
| **Long-horizon task completion** | No benchmark for multi-step autonomous tasks | Design task requiring 10+ iterations, measure completion rate |
| **Conversational quality** | No benchmark for response relevance/coherence | Human evaluation or automatic metrics (BLEU, ROUGE) |
| **Robustness to adversarial input** | Limited negative testing | Fuzz tokenizer, parser, security gates |

### 2.2 Benchmark Quality Assessment

| Benchmark | Quality | Representativeness | Reproducibility |
|---|---|---|---|
| Startup time | HIGH | Representative | YES — deterministic |
| Model loading | HIGH | Representative | YES — deterministic |
| Inference (MPS) | HIGH | Representative for GPU | YES — with sync fix |
| Inference (CPU) | HIGH | Representative for CPU | YES — deterministic |
| Training throughput | LOW | Uses synthetic data | PARTIAL |
| Memory usage | LOW | MPS-only, not comprehensive | PARTIAL |
| Dataset leakage | HIGH | Critical for ML | YES — MinHash/LSH |
| Gradient check | HIGH | Critical for training | YES — numerical |
| GPU parity | HIGH | Critical for correctness | YES — tolerance-based |
| Checkpoint corruption | HIGH | Critical for reliability | YES — deterministic |

---

## 3. Benchmark Artifacts

| Artifact | Location | Description |
|---|---|---|
| `FINAL_BENCHMARK.json` | Root | Main benchmark results |
| `audit_logs/gpu_bench.json` | `audit_logs/` | GPU-specific metrics |
| `audit_logs/final_bench.json` | `audit_logs/` | Final benchmark results |
| `audit_logs/e2e_scenario.json` | `audit_logs/` | E2E scenario traces |
| `neural_core/data/` | `neural_core/data/` | Model artifacts, datasets |

---

## 4. Reproducibility Status

| Aspect | Status | Evidence |
|---|---|---|
| Training seed | VERIFIED | Fixed seed in config |
| Config saving | VERIFIED | `model_config` in checkpoint |
| Tokenizer serialization | VERIFIED | `tokenizer.pkl` + `tokenizer.json` |
| Dataset SHA256 | VERIFIED | `provenance_manifest.json` |
| Code version | VERIFIED | `code_version=0.2.0` from `VERSIONE.md` |
| Run A == Run B | VERIFIED | `test_reproducibility.py` passes |
| Environment capture | VERIFIED | Python version, NumPy version, platform in checkpoint |

**Limitation:** No Run A vs Run B experiment was executed in the documentation. The `test_reproducibility.py` test exists and passes, confirming bitwise reproducibility under controlled conditions.

---

*This document is part of the Technical Acquisition Package.*
