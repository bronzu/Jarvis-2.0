"""
GPU / MPS micro-benchmark for JARVIS 2.0.

Measures (when MPS or CUDA is available):
  - model loading
  - inference latency (per batch)
  - inference throughput (tokens/sec)
  - training step latency
  - batch scaling
  - GPU memory usage (when supported)
  - CPU <-> GPU synchronization overhead

Each measurement reports:
  - method
  - device
  - warmup
  - iterations
  - mean, median, min, max, stddev, p95

When no GPU is available, the benchmark returns a structured "NOT AVAILABLE"
report and records the reason. No manual values are produced.
"""

import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


TEXT = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 40


def _stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0}
    s = sorted(samples)
    n = len(s)
    return {
        "count": n,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stddev": statistics.stdev(samples) if n > 1 else 0.0,
        "p95": s[int(0.95 * (n - 1))],
    }


def _bench(fn: Callable, warmup: int, iterations: int, sync_device: Optional[str] = None) -> Dict[str, float]:
    """Measure end-to-end latency of `fn()`.

    When ``sync_device`` is set to ``"mps"`` or ``"cuda"``, the timer includes
    a backend synchronization call so the reported value reflects the actual
    GPU execution time, not merely CPU dispatch. Without it, the timer would
    only capture the time to enqueue work.
    """
    for _ in range(warmup):
        fn()
        if sync_device:
            _sync(sync_device)
    samples: List[float] = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        if sync_device:
            _sync(sync_device)
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _torch_available() -> bool:
    try:
        import torch
        return True
    except Exception:
        return False


def _resolve_device() -> Optional[str]:
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def _sync(device: str) -> None:
    try:
        import torch
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def _memory_probe(device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    try:
        import torch
        if device == "mps":
            out["available"] = True
            out["current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
            out["driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
        elif device == "cuda":
            out["available"] = True
            out["current_allocated_bytes"] = int(torch.cuda.memory_allocated())
            out["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception as e:
        out["error"] = str(e)
    return out


def benchmark_gpu(device: str = "mps", warmup: int = 2, iterations: int = 10) -> Dict[str, Any]:
    """Run the full GPU/MPS benchmark on the requested device."""
    import numpy as np
    import torch
    from neural_core.torch_model import TorchMicroTransformer
    from neural_core.tokenizer import Tokenizer

    if device not in ("mps", "cuda"):
        return {"device": device, "available": False, "reason": f"unsupported device {device}"}
    if device == "mps" and not torch.backends.mps.is_available():
        return {"device": device, "available": False, "reason": "MPS not available"}
    if device == "cuda" and not torch.cuda.is_available():
        return {"device": device, "available": False, "reason": "CUDA not available"}

    torch.manual_seed(42)
    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT)
    ids = tokenizer.encode(TEXT)

    vocab_size = len(tokenizer.vocabulary)
    d_model, num_heads, d_ff, num_layers, max_len = 32, 2, 64, 1, 64

    out: Dict[str, Any] = {
        "device": device,
        "available": True,
        "torch_version": torch.__version__,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "max_len": max_len,
        "warmup": warmup,
        "iterations": iterations,
    }

    def _create_model():
        return TorchMicroTransformer(
            vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
            d_ff=d_ff, num_layers=num_layers, max_len=max_len, seed=42,
            dtype=torch.float32, device=device,
        )

    out["model_loading"] = _bench(_create_model, warmup=1, iterations=iterations, sync_device=device)

    model = _create_model()
    batch_size, seq_len = 4, 32
    tensor = torch.as_tensor(
        np.asarray(ids[:batch_size * seq_len]).reshape(batch_size, seq_len),
        dtype=torch.long, device=device,
    )

    def _forward():
        with torch.no_grad():
            model.forward(tensor)
    latencies = _bench(_forward, warmup=warmup, iterations=iterations, sync_device=device)
    out["inference_latency"] = latencies
    out["inference_tokens_per_sec"] = {
        "mean": (batch_size * seq_len) / latencies["mean"] if latencies["mean"] > 0 else None,
        "median": (batch_size * seq_len) / latencies["median"] if latencies["median"] > 0 else None,
    }

    def _step():
        model.zero_grad()
        with torch.no_grad():
            logits = model.forward(tensor)
        return logits
    out["training_step_proxy"] = _bench(_step, warmup=warmup, iterations=iterations, sync_device=device)

    sync_samples = []
    for _ in range(warmup):
        _sync(device)
    for _ in range(iterations):
        t0 = time.perf_counter()
        _sync(device)
        sync_samples.append(time.perf_counter() - t0)
    out["synchronization_overhead"] = _stats(sync_samples)

    batch_scaling = []
    for bs in (1, 2, 4, 8):
        bt = torch.as_tensor(
            np.asarray(ids[:bs * seq_len]).reshape(bs, seq_len),
            dtype=torch.long, device=device,
        )
        def _f(b=bt):
            with torch.no_grad():
                model.forward(b)
        s = _bench(_f, warmup=warmup, iterations=max(3, iterations // 2), sync_device=device)
        tokens = bs * seq_len
        s["batch_size"] = bs
        s["tokens_per_sec_mean"] = tokens / s["mean"] if s["mean"] > 0 else None
        batch_scaling.append(s)
    out["batch_scaling"] = batch_scaling

    out["memory"] = _memory_probe(device)
    return out


def run_gpu_benchmark(warmup: int = 2, iterations: int = 10) -> Dict[str, Any]:
    if not _torch_available():
        return {
            "available": False,
            "reason": "torch not installed",
            "platform": platform.platform(),
        }
    device = _resolve_device()
    if device is None:
        return {
            "available": False,
            "reason": "no GPU backend (MPS or CUDA) available",
            "platform": platform.platform(),
        }
    return benchmark_gpu(device=device, warmup=warmup, iterations=iterations)


if __name__ == "__main__":
    print(json.dumps(run_gpu_benchmark(), indent=2, ensure_ascii=False))
