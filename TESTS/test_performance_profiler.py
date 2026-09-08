"""
Tests for neural_core.performance_profiler.
"""

import numpy as np

from neural_core.network import MicroTransformer
from neural_core.performance_profiler import (
    _hardware_info,
    _get_cpu_times,
    _get_python_memory,
    profile_inference,
    profile_section,
    profile_training_step,
)


def test_hardware_info_returns_positive_cpu_count():
    info = _hardware_info()
    assert info["cpu_count"] > 0
    assert "python_version" in info


def test_get_cpu_times_returns_non_negative():
    cpu = _get_cpu_times()
    assert cpu["user_seconds"] >= 0
    assert cpu["system_seconds"] >= 0
    assert cpu["max_rss_bytes"] >= 0


def test_get_python_memory_returns_dict():
    mem = _get_python_memory()
    assert "tracemalloc_current_bytes" in mem
    assert "tracemalloc_peak_bytes" in mem


def test_profile_section_returns_result_and_snapshot():
    def add(a, b):
        return a + b

    result, snap = profile_section("add", add, 2, 3)
    assert result == 5
    assert snap.label == "add"
    assert "cpu" in snap.to_dict()
    assert "memory" in snap.to_dict()


def test_profile_training_step_returns_metrics(tmp_path):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 20
    from neural_core.tokenizer import Tokenizer
    from neural_core.trainer import Trainer
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam

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
    from neural_core.dataset import Dataset
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)
    batch = ds[0]

    metrics = profile_training_step(trainer, batch[0], batch[1])
    assert "label" in metrics
    assert metrics["label"] == "training_step"
    assert metrics["elapsed_seconds"] > 0


def test_profile_inference_returns_metrics():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    from neural_core.tokenizer import Tokenizer

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
    batch = np.asarray(ids).reshape(1, -1)

    metrics = profile_inference(model, batch)
    assert metrics["label"] == "inference"
    assert metrics["elapsed_seconds"] > 0
