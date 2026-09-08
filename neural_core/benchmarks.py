"""
Real performance benchmarks for JARVIS 2.0.

Measures:
  - startup time
  - inference latency
  - training throughput
  - checkpoint size

All values are REAL measurements, not simulated.
"""

import os
import statistics
import time
from pathlib import Path

import numpy as np

from neural_core.network import MicroTransformer
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer
from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam


def _benchmark_startup(iterations: int = 5):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
        tokenizer = Tokenizer(max_vocab_size=50, mode="whitespace")
        tokenizer.fit(text)
        model = MicroTransformer(
            vocab_size=len(tokenizer.vocabulary),
            d_model=16,
            num_heads=2,
            d_ff=32,
            num_layers=1,
            max_len=32,
            seed=42,
        )
        ids = tokenizer.encode(text)
        ds = Dataset(ids, context_size=8, stride=8)
        loss = CrossEntropyLoss()
        optimizer = Adam(learning_rate=0.01)
        trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
        times.append(time.perf_counter() - start)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "unit": "seconds",
        "iterations": iterations,
    }


def _benchmark_inference(iterations: int = 20):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    tokenizer = Tokenizer(max_vocab_size=50, mode="whitespace")
    tokenizer.fit(text)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=32,
        seed=42,
    )
    ids = tokenizer.encode(text)
    ds = Dataset(ids, context_size=8, stride=8)

    latencies = []
    for _ in range(iterations):
        batch = np.asarray(ds[0][0]).reshape(1, -1)
        start = time.perf_counter()
        model.forward(batch, training=False)
        latencies.append(time.perf_counter() - start)

    tokens_per_sec = []
    for lat in latencies:
        tokens = batch.shape[1]
        tokens_per_sec.append(tokens / lat if lat > 0 else float("inf"))

    return {
        "latency_mean": statistics.mean(latencies),
        "latency_median": statistics.median(latencies),
        "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "tokens_per_sec_mean": statistics.mean(tokens_per_sec),
        "tokens_per_sec_median": statistics.median(tokens_per_sec),
        "unit": "seconds",
        "iterations": iterations,
    }


def _benchmark_training(iterations: int = 10):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 20
    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(text)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        max_len=64,
        seed=42,
    )
    ids = tokenizer.encode(text)
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)

    step_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        trainer.train_step(ds[0][0], ds[0][1])
        step_times.append(time.perf_counter() - start)

    return {
        "step_time_mean": statistics.mean(step_times),
        "step_time_median": statistics.median(step_times),
        "step_time_std": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "steps_per_sec_mean": 1.0 / statistics.mean(step_times) if statistics.mean(step_times) > 0 else float("inf"),
        "unit": "seconds",
        "iterations": iterations,
    }


def _benchmark_checkpoint_size(tmp_path: Path):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    tokenizer = Tokenizer(max_vocab_size=20, mode="whitespace")
    tokenizer.fit(text)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=32,
        seed=42,
    )
    ids = tokenizer.encode(text)
    ds = Dataset(ids, context_size=8, stride=8)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)

    ckpt_path = tmp_path / "benchmark_ckpt.pkl"
    trainer.save_checkpoint(ckpt_path)
    size_bytes = ckpt_path.stat().st_size
    return {
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_size_mb": size_bytes / (1024 * 1024),
    }


def run_all(tmp_path: Path) -> dict:
    return {
        "startup": _benchmark_startup(),
        "inference": _benchmark_inference(),
        "training": _benchmark_training(),
        "checkpoint": _benchmark_checkpoint_size(tmp_path),
        "hardware": {
            "cpu": os.cpu_count(),
            "platform": os.name,
        },
    }
