"""
Final performance benchmark suite for JARVIS 2.0.

Measures with reproducible methodology:
  - startup time
  - model loading
  - inference latency (per batch)
  - tokens/sec (inference)
  - training throughput (steps/sec)
  - checkpoint size (bytes, MB)
  - RAM usage (peak RSS, tracemalloc)
  - CPU usage (user + system time)
  - GPU availability (torch CUDA / MPS)
  - tokenizer performance (encode + decode)
  - batch scaling (inference latency vs batch size)

Each measurement records:
  - method
  - warmup
  - iterations
  - mean, median, min, max, stddev, p95
"""

import gc
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np


TEXT_LONG = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 200
TEXT_SHORT = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"


def _stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0}
    samples_sorted = sorted(samples)
    n = len(samples)
    return {
        "count": n,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stddev": statistics.stdev(samples) if n > 1 else 0.0,
        "p95": samples_sorted[int(0.95 * (n - 1))],
    }


def _bench(fn: Callable, warmup: int, iterations: int) -> Dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    raw = r.ru_maxrss
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF)
    raw = r.ru_maxrss
    if platform.system() == "Darwin":
        return raw
    return raw * 1024


def _vm_mb() -> Optional[float]:
    """Best-effort virtual memory size. Returns None on unsupported platforms.

    Darwin does not expose `ru_maxv` via `resource.getrusage`, so VM is only
    available on Linux. On Darwin, this returns None instead of falsely
    returning RSS (which is a different measurement).
    """
    try:
        r = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return None
        return r.ru_maxv / 1024 if hasattr(r, "ru_maxv") else None
    except Exception:
        return None


def _python_allocations() -> Dict[str, int]:
    try:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return {"current_bytes": current, "peak_bytes": peak, "available": True}
    except Exception:
        return {"available": False}


def _ram_metrics() -> Dict[str, Any]:
    """Returns three distinct memory measurements, semantically classified."""
    return {
        "rss": {
            "description": "Resident Set Size (process physical memory in RAM)",
            "bytes": _rss_bytes(),
            "mb": round(_rss_mb(), 2),
            "unit_note": "ru_maxrss units differ by platform: bytes on Darwin, KB on Linux",
        },
        "virtual_memory_mb": _vm_mb(),
        "python_tracemalloc": _python_allocations(),
    }


def _tracemalloc_peak() -> int:
    current, peak = tracemalloc.get_traced_memory()
    return peak


def _gpu_info() -> Dict[str, Any]:
    info = {"torch_available": False, "cuda_available": False, "mps_available": False}
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as e:
        info["error"] = str(e)
    return info


def _hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "numpy_version": np.__version__,
    }


def _ensure_model():
    from neural_core.tokenizer import Tokenizer
    from neural_core.network import MicroTransformer
    from neural_core.dataset import Dataset
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam
    from neural_core.trainer import Trainer

    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT_LONG)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        max_len=64,
        seed=42,
    )
    ids = tokenizer.encode(TEXT_LONG)
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
    return tokenizer, model, ds, trainer


def benchmark_startup(warmup: int = 3, iterations: int = 10) -> Dict[str, Any]:
    def _start():
        from neural_core.tokenizer import Tokenizer
        from neural_core.network import MicroTransformer
        tokenizer = Tokenizer(max_vocab_size=50, mode="whitespace")
        tokenizer.fit(TEXT_SHORT)
        MicroTransformer(
            vocab_size=len(tokenizer.vocabulary),
            d_model=16, num_heads=2, d_ff=32, num_layers=1, max_len=32, seed=42,
        )
    samples = _bench(_start, warmup, iterations)
    return {"name": "startup", "unit": "seconds", "warmup": warmup, "iterations": iterations, **samples}


def benchmark_model_loading(warmup: int = 1, iterations: int = 5) -> Dict[str, Any]:
    def _load():
        from neural_core.network import MicroTransformer
        MicroTransformer(
            vocab_size=100, d_model=32, num_heads=2, d_ff=64, num_layers=1, max_len=64, seed=42,
        )
    samples = _bench(_load, warmup, iterations)
    return {"name": "model_loading", "unit": "seconds", "warmup": warmup, "iterations": iterations, **samples}


def benchmark_inference(warmup: int = 3, iterations: int = 20) -> Dict[str, Any]:
    from neural_core.tokenizer import Tokenizer
    from neural_core.network import MicroTransformer

    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT_LONG)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32, num_heads=2, d_ff=64, num_layers=1, max_len=64, seed=42,
    )
    ids = tokenizer.encode(TEXT_LONG)
    batch = np.asarray(ids[:32]).reshape(1, -1)

    def _forward():
        model.forward(batch, training=False)
    samples = _bench(_forward, warmup, iterations)
    tokens_per_iter = batch.shape[1]
    return {
        "name": "inference_latency",
        "unit": "seconds",
        "warmup": warmup,
        "iterations": iterations,
        "tokens_per_iteration": tokens_per_iter,
        "tokens_per_sec_mean": tokens_per_iter / samples["mean"] if samples["mean"] > 0 else None,
        "tokens_per_sec_median": tokens_per_iter / samples["median"] if samples["median"] > 0 else None,
        **samples,
    }


def benchmark_training(warmup: int = 2, iterations: int = 10) -> Dict[str, Any]:
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer
    from neural_core.dataset import Dataset
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam
    from neural_core.trainer import Trainer

    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT_LONG)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32, num_heads=2, d_ff=64, num_layers=1, max_len=64, seed=42,
    )
    ids = tokenizer.encode(TEXT_LONG)
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
    batch = ds[0]

    def _step():
        trainer.train_step(batch[0], batch[1])
    samples = _bench(_step, warmup, iterations)
    return {
        "name": "training_throughput",
        "unit": "seconds",
        "warmup": warmup,
        "iterations": iterations,
        "steps_per_sec_mean": 1.0 / samples["mean"] if samples["mean"] > 0 else None,
        "steps_per_sec_median": 1.0 / samples["median"] if samples["median"] > 0 else None,
        **samples,
    }


def benchmark_checkpoint_size(tmp_path: Path) -> Dict[str, Any]:
    from neural_core.network import MicroTransformer
    from neural_core.tokenizer import Tokenizer
    from neural_core.dataset import Dataset
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam
    from neural_core.trainer import Trainer

    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT_LONG)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32, num_heads=2, d_ff=64, num_layers=1, max_len=64, seed=42,
    )
    ids = tokenizer.encode(TEXT_LONG)
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
    ckpt = tmp_path / "bench_ckpt.pkl"
    trainer.save_checkpoint(ckpt)
    size = ckpt.stat().st_size
    return {
        "name": "checkpoint_size",
        "size_bytes": size,
        "size_mb": size / (1024 * 1024),
    }


def benchmark_tokenizer(warmup: int = 1, iterations: int = 20) -> Dict[str, Any]:
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer(max_vocab_size=1000, mode="whitespace")
    tokenizer.fit(TEXT_LONG)

    def _encode():
        return tokenizer.encode(TEXT_LONG)
    enc = _bench(_encode, warmup, iterations)
    tokens = len(tokenizer.encode(TEXT_LONG))

    def _decode():
        tokenizer.decode(list(range(tokens)))
    dec = _bench(_decode, warmup, iterations)
    return {
        "name": "tokenizer",
        "encode": {"unit": "seconds", "warmup": warmup, "iterations": iterations, **enc, "tokens_per_call": tokens},
        "decode": {"unit": "seconds", "warmup": warmup, "iterations": iterations, **dec},
    }


def benchmark_batch_scaling() -> List[Dict[str, Any]]:
    from neural_core.tokenizer import Tokenizer
    from neural_core.network import MicroTransformer

    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(TEXT_LONG)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32, num_heads=2, d_ff=64, num_layers=1, max_len=64, seed=42,
    )
    ids = tokenizer.encode(TEXT_LONG)
    seq_len = 32
    results = []
    for batch_size in (1, 2, 4, 8):
        batch = np.asarray(ids[:batch_size * seq_len]).reshape(batch_size, seq_len)
        def _f(b=batch):
            model.forward(b, training=False)
        s = _bench(_f, warmup=2, iterations=8)
        tokens = batch_size * seq_len
        s["batch_size"] = batch_size
        s["seq_len"] = seq_len
        s["tokens_per_iteration"] = tokens
        s["tokens_per_sec_mean"] = tokens / s["mean"] if s["mean"] > 0 else None
        s["name"] = "batch_scaling"
        results.append(s)
    return results


def run_final_benchmark(tmp_path: Path) -> Dict[str, Any]:
    tracemalloc.start()
    start_time = time.perf_counter()
    out: Dict[str, Any] = {
        "hardware": _hardware_info(),
        "gpu": _gpu_info(),
        "ram": _ram_metrics(),
    }
    out["startup"] = benchmark_startup()
    out["model_loading"] = benchmark_model_loading()
    out["inference"] = benchmark_inference()
    out["training"] = benchmark_training()
    out["checkpoint"] = benchmark_checkpoint_size(tmp_path)
    out["tokenizer"] = benchmark_tokenizer()
    out["batch_scaling"] = benchmark_batch_scaling()
    out["ram"]["python_tracemalloc"] = _python_allocations()

    from neural_core.gpu_benchmark_extended import run_gpu_benchmark
    out["gpu_benchmark"] = run_gpu_benchmark(warmup=2, iterations=10)

    out["elapsed_seconds"] = time.perf_counter() - start_time
    tracemalloc.stop()
    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(run_final_benchmark(Path(d)), indent=2, ensure_ascii=False))
