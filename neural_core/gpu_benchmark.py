"""
Phase 2A — CPU vs GPU benchmark (reproducible micro-benchmark).

Measures wall-time and throughput for:
    - forward
    - backward (gradient computation)
    - full training step (forward + loss + backward + Adam update)

on the SAME model (current MicroTransformer arch) and SAME inputs, comparing:
    CPU  : NumPy / float64  (reference)
    GPU  : PyTorch / MPS / float32

Also reports GPU memory usage and environment/software versions.
This is a MICRO benchmark only; it does NOT run full training.
"""

import os
import pickle
import platform
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
MODEL_PKL = os.path.join(REPO, "neural_core", "data", "model.pkl")
CONFIG = {
    "vocab_size": 8590, "d_model": 128, "num_heads": 4, "d_ff": 256,
    "num_layers": 2, "max_len": 256, "dropout": 0.0, "seed": None,
}


def _load_weights():
    with open(MODEL_PKL, "rb") as fh:
        return pickle.load(fh)["parameters"]


def _bench(fn, repeats=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.mean(ts))


def main():
    import torch
    from neural_core.network import MicroTransformer
    from neural_core.torch_model import TorchMicroTransformer
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam

    params = _load_weights()
    rng = np.random.RandomState(42)
    batch, seq = 8, 128
    ids = rng.randint(0, CONFIG["vocab_size"], size=(batch, seq)).tolist()
    tgt = rng.randint(0, CONFIG["vocab_size"], size=(batch, seq)).tolist()
    tokens = batch * seq

    # ---- CPU (NumPy float64) ----
    cpu = MicroTransformer(**CONFIG)
    cpu.set_parameters(params)
    ce = CrossEntropyLoss()

    def cpu_forward():
        return cpu.forward(ids)

    def cpu_step():
        ce.forward(cpu.forward(ids), tgt)
        g = cpu.backward(tgt)
        Adam(learning_rate=1e-3).update(cpu.get_parameters(), g)

    cpu_fwd_med, _ = _bench(cpu_forward)
    cpu_step_med, _ = _bench(cpu_step)

    # ---- GPU (PyTorch MPS float32) ----
    gpu = TorchMicroTransformer(**CONFIG, device="mps")
    gpu.set_parameters(params)
    idst = torch.as_tensor(np.asarray(ids), dtype=torch.long, device="mps")
    tgtt = torch.as_tensor(np.asarray(tgt).reshape(-1), dtype=torch.long, device="mps")

    def gpu_forward():
        with torch.no_grad():
            return gpu.forward(idst)

    gopt = torch.optim.Adam(gpu.parameters(), lr=1e-3)

    def gpu_step():
        gpu.zero_grad()
        logits = gpu.forward(idst)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, gpu.vocab_size), tgtt, ignore_index=-100, reduction="mean")
        loss.backward()
        gopt.step()

    # GPU memory probe
    mem_before = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
    gpu_fwd_med, _ = _bench(gpu_forward)
    mem_after_fwd = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
    gpu_step_med, _ = _bench(gpu_step)
    mem_after_step = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0

    def r(x):
        return f"{x*1000:.3f} ms"

    print("=" * 70)
    print("JARVIS 2.0 — PHASE 2A MICRO BENCHMARK (CPU float64 vs GPU float32)")
    print("=" * 70)
    print(f"Environment : Python {sys.version.split()[0]} / {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"NumPy       : {np.__version__}")
    print(f"PyTorch     : {torch.__version__}  (MPS available: {torch.backends.mps.is_available()})")
    print(f"Model       : d_model={CONFIG['d_model']} heads={CONFIG['num_heads']} d_ff={CONFIG['d_ff']} "
          f"layers={CONFIG['num_layers']} vocab={CONFIG['vocab_size']} max_len={CONFIG['max_len']}")
    print(f"Params      : {gpu.parameter_count():,}")
    print(f"Batch/Seq   : {batch}/{seq}  ({tokens:,} tokens per call)")
    print("-" * 70)
    print(f"{'operation':<14}{'CPU (NumPy)':>16}{'GPU (MPS)':>16}{'speedup':>12}")
    print(f"{'forward':<14}{r(cpu_fwd_med):>16}{r(gpu_fwd_med):>16}"
          f"{cpu_fwd_med/gpu_fwd_med:>11.2f}x")
    print(f"{'train step':<14}{r(cpu_step_med):>16}{r(gpu_step_med):>16}"
          f"{cpu_step_med/gpu_step_med:>11.2f}x")
    print("-" * 70)
    print(f"Throughput  : CPU {tokens/cpu_step_med:,.0f} tok/s (step) | "
          f"GPU {tokens/gpu_step_med:,.0f} tok/s (step)")
    print(f"GPU memory  : allocated {mem_after_step/1e6:.1f} MB "
          f"(fwd +{max(0,mem_after_fwd-mem_before)/1e6:.1f} MB, step +{max(0,mem_after_step-mem_before)/1e6:.1f} MB)")
    print("=" * 70)
    print("NOTE: micro-benchmark only; no full training performed.")


if __name__ == "__main__":
    main()
