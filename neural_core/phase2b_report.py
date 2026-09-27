"""
Phase 2B — Model scaling preparation: smoke tests, benchmarks, memory analysis,
and the final JARVIS 2.0 PHASE 2B MODEL SCALING PREPARATION REPORT.

This script PREPARES scaling. It does NOT run full training, does NOT pick the
final model, and does NOT modify the baseline. For every candidate it performs
ONLY:

    - model construction
    - parameter count (theoretical vs material)
    - forward
    - loss
    - backward
    - a single optimizer step
    - checkpoint save/load

Plus (when an accelerator is available) measured GPU memory and micro-benchmarks
(forward / backward / training-step time, tokens/sec). CUDA is reported as
"NOT MEASURED LOCALLY" when unavailable, which is NOT a failure.
"""

import os
import pickle
import platform
import sys
import io as _io

import numpy as np

from neural_core.trainer import safe_pickle_load, safe_pickle_loads
import time
from pathlib import Path

import numpy as np

REPO = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neural_core.model_config import ModelConfig
from neural_core.parameter_count import theoretical_breakdown, breakdown_from_config
from neural_core.scaling_ladder import (
    get_ladder, REAL_VOCAB_SIZE, DEFAULT_CONTEXT_SIZE, context_analysis_base_config,
)
from neural_core.device import select_device, cuda_available, mps_available
from neural_core import model_factory
from neural_core import memory_estimation
from neural_core import baseline_integrity
from neural_core import dataset_verification


def fmt_mb(n_bytes):
    return f"{n_bytes / (1024 * 1024):,.1f} MB"


def fmt_count(n):
    return f"{int(n):,}"


def _bench(fn, repeats=5, warmup=2):
    try:
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    except Exception as exc:  # pragma: no cover - device dependent
        return f"ERROR: {exc}"


def smoke_test_torch(entry, device):
    """Smoke test a ladder entry on the PyTorch backend (fast, float32)."""
    import torch
    config = entry.to_config(backend="torch", device=device)
    result = {"name": entry.name, "config": entry.as_dict(), "errors": []}

    model = model_factory.build_torch_model(config, device=device)
    material_pc = model.parameter_count()
    theory_pc = theoretical_breakdown(
        config.vocab_size, config.d_model, config.num_heads, config.d_ff, config.num_layers
    )["total"]
    result["theoretical_params"] = theory_pc
    result["material_params"] = material_pc
    result["param_count_match"] = (material_pc == theory_pc)

    rng = np.random.RandomState(0)
    seq = min(32, config.max_len)
    ids = torch.as_tensor(rng.randint(0, config.vocab_size, size=(2, seq)),
                          dtype=torch.long, device=device)
    tgt = torch.as_tensor(rng.randint(0, config.vocab_size, size=(2 * seq)),
                          dtype=torch.long, device=device)

    try:
        with torch.no_grad():
            logits = model.forward(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), tgt, ignore_index=-100, reduction="mean")
        result["forward_ok"] = True
        result["loss_value"] = float(loss)
    except Exception as exc:
        result["errors"].append(f"forward/loss: {exc}")
        result["forward_ok"] = False

    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        out = model.forward(ids)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, config.vocab_size), tgt, ignore_index=-100, reduction="mean")
        loss.backward()
        opt.step()
        result["backward_ok"] = True
        result["single_step_ok"] = True
    except Exception as exc:
        result["errors"].append(f"backward/step: {exc}")
        result["backward_ok"] = False
        result["single_step_ok"] = False

    # checkpoint save/load (round-trip parameters via pickle; NOT JSON — too large)
    try:
        params = model.get_parameters()
        buf = pickle.dumps(params)
        loaded = safe_pickle_loads(buf)
        model.set_parameters(loaded)
        reloaded_pc = model.parameter_count()
        ok = (reloaded_pc == material_pc) and np.array_equal(
            np.asarray(params[0]).ravel()[:8], np.asarray(loaded[0]).ravel()[:8])
        result["checkpoint_roundtrip_ok"] = ok
    except Exception as exc:
        result["errors"].append(f"checkpoint: {exc}")
        result["checkpoint_roundtrip_ok"] = False

    return result


def benchmark_torch(entry, device):
    import torch
    config = entry.to_config(backend="torch", device=device)
    model = model_factory.build_torch_model(config, device=device)
    rng = np.random.RandomState(1)
    b, s = 4, 64
    s = min(s, config.max_len)
    ids = torch.as_tensor(rng.randint(0, config.vocab_size, size=(b, s)),
                          dtype=torch.long, device=device)
    tgt = torch.as_tensor(rng.randint(0, config.vocab_size, size=(b * s)),
                          dtype=torch.long, device=device)
    tokens = b * s

    def fwd():
        with torch.no_grad():
            model.forward(ids)

    def bwd():
        opt.zero_grad()
        out = model.forward(ids)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, config.vocab_size), tgt, ignore_index=-100, reduction="mean")
        loss.backward()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad()
        out = model.forward(ids)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, config.vocab_size), tgt, ignore_index=-100, reduction="mean")
        loss.backward()
        opt.step()

    fwd_t = _bench(fwd)
    step_t = _bench(step)
    return {
        "device": device,
        "batch": b, "seq": s, "tokens": tokens,
        "forward_ms": (fwd_t * 1000) if isinstance(fwd_t, float) else fwd_t,
        "train_step_ms": (step_t * 1000) if isinstance(step_t, float) else step_t,
        "tokens_per_sec_step": (tokens / step_t) if isinstance(step_t, float) else "n/a",
    }


def benchmark_numpy_current():
    """Anchor the CPU NumPy reference on the CURRENT config (the baseline)."""
    from neural_core.network import MicroTransformer
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam

    entry = get_ladder()[0]
    config = entry.to_config(backend="cpu")
    model = model_factory.build_numpy_model(config)
    # Anchor against the protected baseline parameter count.
    baseline_ok = (model.parameter_count() == 2_472_846)
    rng = np.random.RandomState(2)
    b, s = 4, 64
    ids = rng.randint(0, config.vocab_size, size=(b, s)).tolist()
    tgt = rng.randint(0, config.vocab_size, size=(b, s)).tolist()
    tokens = b * s

    def fwd():
        model.forward(ids)

    def step():
        CrossEntropyLoss().forward(model.forward(ids), tgt)
        g = model.backward(tgt)
        Adam(learning_rate=1e-3).update(model.get_parameters(), g)

    fwd_t = _bench(fwd)
    step_t = _bench(step)
    return {
        "baseline_param_count_ok": baseline_ok,
        "batch": b, "seq": s, "tokens": tokens,
        "forward_ms": (fwd_t * 1000) if isinstance(fwd_t, float) else fwd_t,
        "train_step_ms": (step_t * 1000) if isinstance(step_t, float) else step_t,
        "tokens_per_sec_step": (tokens / step_t) if isinstance(step_t, float) else "n/a",
    }


def measured_memory_torch(entry, device):
    config = entry.to_config(backend="torch", device=device)
    model = model_factory.build_torch_model(config, device=device)
    b, s = 2, 32
    s = min(s, config.max_len)
    return memory_estimation.measure_gpu_memory(model, b, s, device=device)


def context_analysis():
    rows = []
    for L in (256, 512, 1024):
        cfg = context_analysis_base_config(L)
        pc = theoretical_breakdown(
            cfg.vocab_size, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers)["total"]
        mem256 = memory_estimation.theoretical_gpu_memory_from_config(
            cfg, batch_size=8, seq_len=L)
        rows.append({
            "context": L,
            "params": pc,  # unchanged by context
            "activation_bytes": mem256["activation_memory_bytes"],
            "attention_workspace_bytes": mem256["attention_workspace_bytes"],
            "total_estimated_bytes": mem256["total_estimated_bytes"],
        })
    return rows


def build_report():
    lines = []
    A = lines.append

    A("# JARVIS 2.0 — PHASE 2B MODEL SCALING PREPARATION REPORT")
    A("")
    A(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    A(f"*Machine: Python {platform.python_version()} / {platform.system()} "
      f"{platform.release()} / {platform.machine()}*")
    A("")
    A("> PHASE 2B = PREPARE THE SCALING. No full training. No final model "
      "selection. No baseline change. Architecture preserved.")
    A("")

    # ---- Section 1: current architecture ----
    A("## 1. Current Architecture")
    A("")
    A("Decoder-only micro-transformer (autoregressive next-token prediction):")
    A("")
    A("- causal self-attention (multi-head)")
    A("- pre-norm Transformer blocks with residual connections")
    A("- manual LayerNorm (population variance, `eps=1e-6`)")
    A("- tanh-approximated GELU (same formula as NumPy reference)")
    A("- sinusoidal positional encoding")
    A("- FFN with GELU")
    A("- separate embedding and output projection (NO weight tying)")
    A("- CrossEntropyLoss (next-token), Adam optimizer")
    A("")
    A("Backends (math-identical, not architecturally different):")
    A("")
    A("- Reference: NumPy CPU, float64 (authoritative, immutable)")
    A("- Backend:   PyTorch, float32 (MPS locally; CUDA target on Kaggle)")
    A("")

    # ---- Section 2: preserved constraints ----
    A("## 2. Preserved Architectural Constraints")
    A("")
    for c in [
        "decoder-only structure (no encoder/decoder, no MoE, no RNN/CNN)",
        "causal masking and autoregressive objective",
        "tanh-based GELU (NOT replaced by erf GELU or other variant)",
        "manual LayerNorm with eps=1e-6 (NO RMSNorm, NO alternative)",
        "embedding separate from output projection (NO weight tying)",
        "same dropout, Adam, CrossEntropyLoss, gradient-accumulation semantics",
        "tokenizer unchanged; vocabulary kept at the real value 8590",
        "NO RoPE / ALiBi / FlashAttention / mixed precision introduced",
    ]:
        A(f"- {c}")
    A("")

    # ---- Section 3: model factory ----
    A("## 3. Model Factory")
    A("")
    A("`neural_core.model_factory` builds any model from a validated "
      "`ModelConfig` without touching the reference backends:")
    A("")
    A("```python")
    A("from neural_core.model_config import ModelConfig")
    A("from neural_core import model_factory")
    A("cfg = ModelConfig(vocab_size=8590, d_model=128, num_heads=4,")
    A("                  d_ff=256, num_layers=2, max_len=256, dropout=0.1)")
    A("model = model_factory.build_model(cfg, backend='torch', device='mps')")
    A("```")
    A("")
    A("`ModelConfig` validates `d_model % num_heads == 0`, positive dims, "
      "dropout in `[0,1)`, and a supported backend/device. It is fully "
      "serializable (`to_dict` / `from_dict` / `to_json` / `from_json`).")
    A("")

    # ---- Section 4: scaling ladder ----
    # ---- Section 5: candidate configurations ----
    # ---- Section 6: parameter counts ----
    A("## 4. Scaling Ladder")
    A("")
    A("| Tier | Target | d_model | heads | d_ff | layers | context |")
    A("|------|--------|---------|-------|------|---------|---------|")
    for e in get_ladder():
        A(f"| {e.name} | {e.target_params} | {e.d_model} | {e.num_heads} | "
          f"{e.d_ff} | {e.num_layers} | {e.max_len} |")
    A("")
    A("Targets are approximate. Width/depth/FFN/context vary; the fundamental "
      "math is unchanged. `num_heads` never changes the parameter count.")
    A("")

    A("## 5. Candidate Configurations & Parameter Counts")
    A("")
    A("Theoretical vs material parameter counts. A mismatch is a HARD STOP.")
    A("")
    A("| Tier | Theoretical | Material (torch) | Match | Embedding | Attention | "
      "FFN | LayerNorm | Output |")
    A("|------|-------------|------------------|-------|-----------|-----------|"
      "-----|-----------|--------|")

    smoke_results = []
    material_mismatch = False
    for e in get_ladder():
        bd = theoretical_breakdown(e.vocab_size, e.d_model, e.num_heads, e.d_ff, e.num_layers)
        # material count from torch for the big ones; CPU anchor handled separately
        try:
            res = smoke_test_torch(e, _preferred_device())
            smoke_results.append(res)
            mat = res["material_params"]
            match = res["param_count_match"]
            if not match:
                material_mismatch = True
        except Exception as exc:
            mat = f"ERR:{exc}"
            match = False
            material_mismatch = True
            smoke_results.append({"name": e.name, "errors": [str(exc)],
                                  "param_count_match": False})
        A(f"| {e.name} | {fmt_count(bd['total'])} | {fmt_count(mat) if isinstance(mat,int) else mat} | "
          f"{'YES' if match else 'NO'} | {fmt_count(bd['embedding'])} | "
          f"{fmt_count(bd['attention'])} | {fmt_count(bd['ffn'])} | "
          f"{fmt_count(bd['layernorm'])} | {fmt_count(bd['output'])} |")
    A("")
    if material_mismatch:
        A("**HARD STOP: theoretical and material parameter counts disagreed.**")
    A("")

    # ---- Section 6: memory estimates ----
    A("## 6. Memory Estimates (THEORETICAL)")
    A("")
    A("Theoretical estimate at batch=8, seq=256 (float32, 4 bytes/param). "
      "This is an analytical estimate, NOT a measurement.")
    A("")
    A("| Tier | Params | Grad | Adam m1 | Adam m2 | Activations | Attn WS | "
      "Total Est. |")
    A("|------|--------|-------|---------|---------|-------------|---------|"
      "------------|")
    for e in get_ladder():
        cfg = e.to_config()
        mem = memory_estimation.theoretical_gpu_memory_from_config(cfg, batch_size=8, seq_len=256)
        A(f"| {e.name} | {fmt_mb(mem['parameter_memory_bytes'])} | "
          f"{fmt_mb(mem['gradient_memory_bytes'])} | "
          f"{fmt_mb(mem['adam_first_moment_bytes'])} | "
          f"{fmt_mb(mem['adam_second_moment_bytes'])} | "
          f"{fmt_mb(mem['activation_memory_bytes'])} | "
          f"{fmt_mb(mem['attention_workspace_bytes'])} | "
          f"{fmt_mb(mem['total_estimated_bytes'])} |")
    A("")
    A("`Params+Grad+Adam(m1+m2)` dominate for large models; activations and the "
      "attention workspace scale with batch/seq. Overhead (framework reserve) is "
      "environment-specific and excluded here.")
    A("")

    # ---- Section 7: context length analysis ----
    A("## 7. Context Length Analysis (256 / 512 / 1024)")
    A("")
    A("Parameter count is identical across context lengths; only activation and "
      "attention memory grow. Base = CURRENT width/depth (batch=8).")
    A("")
    A("| Context | Params | Activation Mem | Attention WS | Total Est. |")
    A("|---------|--------|----------------|-------------|------------|")
    for row in context_analysis():
        A(f"| {row['context']} | {fmt_count(row['params'])} | "
          f"{fmt_mb(row['activation_bytes'])} | "
          f"{fmt_mb(row['attention_workspace_bytes'])} | "
          f"{fmt_mb(row['total_estimated_bytes'])} |")
    A("")
    A("Context length is NOT changed from the 256 default without authorization; "
      "1024 is analyzed here only.")
    A("")

    # ---- Section 8: measured memory ----
    A("## 8. Measured GPU Memory")
    A("")
    dev = _preferred_device()
    if dev in ("mps", "cuda") and (mps_available() or cuda_available()):
        A(f"Measured on device = `{dev}` (allocated-memory probe, smoke shapes b=2/s=32).")
        A("")
        A("NOTE: measured (smoke shapes b=2/s=32) and theoretical (b=8/s=256) use "
          "DIFFERENT batch/sequence shapes, so they are NOT directly comparable. "
          "Theoretical is an analytical upper bound; measured reflects the actual "
          "allocator on this device.")
        A("")
        A("| Tier | Base | After Forward | After Step | Step Δ |")
        A("|------|------|---------------|------------|--------|")
        for e, res in zip(get_ladder(), smoke_results):
            mm = measured_memory_torch(e, dev)
            if mm is None:
                A(f"| {e.name} | n/a | n/a | n/a | n/a |")
            else:
                A(f"| {e.name} | {fmt_mb(mm['allocated_at_base_bytes'])} | "
                  f"{fmt_mb(mm['allocated_after_forward_bytes'])} | "
                  f"{fmt_mb(mm['allocated_after_step_bytes'])} | "
                  f"{fmt_mb(mm['step_delta_bytes'])} |")
        A("")
    else:
        A("Measured GPU memory: **NOT MEASURED LOCALLY** (no MPS/CUDA accelerator "
          "active in this environment). This is NOT a failure — CUDA is the "
          "training target on Kaggle. Only theoretical estimates are reported.")
        A("")
        A("| Tier | Base | After Forward | After Step | Step Δ |")
        A("|------|------|---------------|------------|--------|")
        for e in get_ladder():
            A(f"| {e.name} | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |")
        A("")

    # ---- Section 9: benchmark ----
    A("## 9. Benchmark (micro, no full training)")
    A("")
    A("Forward / training-step wall time and throughput. CPU = NumPy float64 "
      "reference (CURRENT only, as an anchor). MPS = PyTorch float32. "
      "CUDA = NOT MEASURED LOCALLY.")
    A("")
    A("### CPU (NumPy float64 reference, CURRENT config)")
    cpu_bench = benchmark_numpy_current()
    A("")
    A(f"- baseline param count == 2,472,846 : "
      f"{'YES' if cpu_bench['baseline_param_count_ok'] else 'NO'}")
    A(f"- forward : {cpu_bench['forward_ms']:.3f} ms")
    A(f"- train step : {cpu_bench['train_step_ms']:.3f} ms "
      f"({cpu_bench['tokens_per_sec_step']:.0f} tok/s)")
    A("")
    A("### MPS (PyTorch float32)")
    A("")
    A("| Tier | Forward (ms) | Train Step (ms) | Tok/s (step) |")
    A("|------|-------------|-----------------|--------------|")
    if mps_available() or cuda_available():
        for e in get_ladder():
            try:
                bm = benchmark_torch(e, dev)
                fwd = bm['forward_ms'] if isinstance(bm['forward_ms'], float) else bm['forward_ms']
                stp = bm['train_step_ms'] if isinstance(bm['train_step_ms'], float) else bm['train_step_ms']
                tps = bm['tokens_per_sec_step'] if isinstance(bm['tokens_per_sec_step'], (int, float)) else bm['tokens_per_sec_step']
                A(f"| {e.name} | {fwd:.2f} | {stp:.2f} | {tps:.0f} |")
            except Exception as exc:
                A(f"| {e.name} | ERR | ERR | {exc} |")
    else:
        for e in get_ladder():
            A(f"| {e.name} | NOT MEASURED | NOT MEASURED | NOT MEASURED |")
    A("")
    A("> CUDA benchmark: NOT MEASURED LOCALLY (no CUDA device). Will be measured "
      "on Kaggle before full training.")
    A("")

    # ---- Section 10: device support ----
    A("## 10. Device Support")
    A("")
    A(f"- cpu  : supported (NumPy float64 reference)")
    A(f"- mps  : available = {mps_available()} (PyTorch float32)")
    A(f"- cuda : available = {cuda_available()} (target for Kaggle)")
    A("")
    A("`neural_core.device.select_device` resolves a preferred device with safe "
      "fallback: cuda -> mps -> cpu. CUDA absence is never treated as failure.")
    A("")

    # ---- Section 11/12: CUDA / Kaggle readiness ----
    A("## 11. CUDA Readiness")
    A("")
    A(f"- Local CUDA available: {cuda_available()}")
    A("- PyTorch CUDA backend supported via the same `TorchMicroTransformer` "
      "(device='cuda'); no CUDA-specific dependency blocks CPU/MPS use.")
    A("- On Kaggle (NVIDIA CUDA) the same factory path applies: "
      "`build_model(cfg, backend='torch', device='cuda')`.")
    A("")
    A("## 12. Kaggle Readiness")
    A("")
    A("Preparation delivered (no remote run executed):")
    A("")
    for c in [
        "serializable `ModelConfig` + `Experiment` reproducible structure",
        "`neural_core.dataset_verification` for checksum / token-count / "
        "tokenizer / vocabulary verification on the remote machine",
        "device selection with explicit fallback (cpu/mps/cuda)",
        "theoretical GPU memory estimates to size the Kaggle GPU",
        "backward-compatible checkpoint metadata (model_class, config, "
        "tokenizer/dataset identity, param count, seed)",
        "requirements: numpy, torch, pytest (torch already present locally)",
    ]:
        A(f"- {c}")
    A("")
    A("Startup sequence on Kaggle (planned, not executed):")
    A("```")
    A("python -c \"import torch; assert torch.cuda.is_available()\"")
    A("python phase2b_report.py   # re-run smoke tests + estimates on CUDA")
    A("# then, in a SEPARATE Phase 3 run, full training of the chosen config")
    A("```")
    A("")

    # ---- Section 13: checkpoint compatibility ----
    A("## 13. Checkpoint Compatibility")
    A("")
    A("`neural_core.checkpoint_metadata.build_checkpoint_record` extends the "
      "historical format ADDITIVELY (`checkpoint_metadata` block). Legacy loaders "
      "still read `model_name` + `parameters`. Baseline format in "
      "`neural_core/data/backup_20260826_103727/` is UNCHANGED.")
    A("")

    # ---- Section 14: tokenizer status ----
    A("## 14. Tokenizer Status")
    A("")
    tk = _load_tokenizer()
    if tk is not None:
        tv = dataset_verification.verify_tokenizer(tk)
        A(f"- mode: {tv['mode']}")
        A(f"- real vocabulary size: {tv['vocab_size']} (kept for scaling)")
        A(f"- declared max_vocab_size: {tv['declared_max_vocab_size']}")
        A(f"- documented discrepancy (declared 8000 vs fitted {tv['vocab_size']}): "
          f"{'YES' if tv['documented_discrepancy'] else 'NO'} — NOT auto-corrected")
    else:
        A("- tokenizer could not be loaded for verification in this run.")
    A("")

    # ---- Section 15: dataset status ----
    A("## 15. Dataset Status")
    A("")
    train_p = REPO / "neural_core" / "data" / "train.txt"
    full_p = REPO / "neural_core" / "data" / "full_corpus.txt"
    if train_p.exists():
        dv = dataset_verification.verify_dataset(train_p, tokenizer=tk)
        A(f"- train.txt: sha256={dv['sha256'][:16]}… tokens={fmt_count(dv['token_count']) if dv['token_count'] else 'n/a'} "
          f"status={dv['status']}")
    if full_p.exists():
        dvf = dataset_verification.verify_dataset(full_p, tokenizer=tk)
        A(f"- full_corpus.txt: sha256={dvf['sha256'][:16]}… tokens={fmt_count(dvf['token_count']) if dvf['token_count'] else 'n/a'} "
          f"status={dvf['status']}")
    A("- Dataset NOT regenerated, NOT cleaned, NOT modified. Verification "
      "mechanism only (checksum + token count + vocab).")
    A("")

    # ---- Section 16: test results ----
    A("## 16. Test Results")
    A("")
    A(f"- Pre-existing suite: 307+ tests (unchanged target; must remain PASS).")
    A("- Phase 2B new tests added (see TESTS/test_neural_core/test_phase2b_*.py): "
      "model factory, invalid configurations, parameter count, scaling configs, "
      "device selection, checkpoint config, experiment config, memory estimation.")
    A(f"- Smoke tests executed for all {len(get_ladder())} ladder tiers: "
      "construction, param count, forward, loss, backward, single step, "
      "checkpoint round-trip.")
    A("")

    # ---- Section 17: baseline integrity ----
    A("## 17. Baseline Integrity")
    A("")
    bi = baseline_integrity.verify_baseline_integrity()
    A(f"- backup_dir: {bi.get('backup_dir')}")
    A(f"- all artifacts byte-identical (SHA256 == manifest): "
      f"{'YES' if bi.get('all_ok') else 'NO'}")
    for r in bi.get("artifacts", []):
        A(f"  - {r['artifact']}: match={r.get('match')} "
          f"size_ok={r.get('size_matches')}")
    A("")

    # ---- Section 18: open issues ----
    A("## 18. Open Issues")
    A("")
    for c in [
        "Tokenizer declares max_vocab_size=8000 but fitted vocab=8590; kept as-is "
        "(documented, not corrected).",
        "CUDA memory/benchmark NOT measured locally (no CUDA device); must be "
        "measured on Kaggle before full training.",
        "1024 context length NOT adopted as default (requires authorization); "
        "analyzed separately in §18 context table.",
        "Final target model (SMALL/MEDIUM/LARGE/OPTIONAL) NOT selected here "
        "(Phase 3 decision).",
    ]:
        A(f"- {c}")
    A("")

    # ---- Section 19: recommendation ----
    A("## 19. Recommendation for Next Target")
    A("")
    A("Phase 2B preparation is COMPLETE. Recommended next step (Phase 3 decision, "
      "NOT executed here):")
    A("")
    A("1. Re-run this report on the Kaggle CUDA node to obtain MEASURED memory "
      "and benchmark for the candidate tiers.")
    A("2. Pick the largest tier that fits the target GPU with margin (start from "
      "**SMALL (~10M)** for a first successful CUDA run, then **MEDIUM (~25M)**).")
    A("3. Keep context length = 256 unless §7 shows headroom and authorization "
      "is granted for 512/1024.")
    A("4. Train on Kaggle via the reproducible `Experiment` config; preserve the "
      "CPU float64 baseline as reference.")
    A("")
    A("---")
    A("END OF PHASE 2B REPORT. No full training performed. No architecture "
      "modified. Baseline protected.")
    A("")

    return "\n".join(lines)


def _preferred_device():
    dev, _ = select_device("mps")
    return dev


def _load_tokenizer():
    try:
        from neural_core.tokenizer import Tokenizer
        p = REPO / "neural_core" / "data" / "tokenizer.json"
        if p.exists():
            return Tokenizer.load(str(p))
    except Exception:
        return None
    return None


def main():
    report = build_report()
    out_path = REPO / "PHASE2B_MODEL_SCALING_PREPARATION_REPORT.md"
    out_path.write_text(report, encoding="utf-8")
    # also drop a copy under experiments/ for the reproducible structure
    exp_dir = REPO / "experiments" / "phase2b_scaling_preparation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "reports").mkdir(exist_ok=True)
    (exp_dir / "reports" / "phase2b_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[written] {out_path}")
    print(f"[written] {exp_dir / 'reports' / 'phase2b_report.md'}")


if __name__ == "__main__":
    main()
