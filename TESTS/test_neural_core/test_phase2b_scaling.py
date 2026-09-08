"""
Phase 2B — Model scaling preparation tests.

Covers (per the Phase 2B test-regression requirement):

    - model factory (build from config, backend selection, device fallback)
    - invalid configurations (d_model % num_heads, ranges, dropout, dtype)
    - parameter count (theoretical vs material) for every ladder tier
    - scaling configurations (ladder structure + targets)
    - device selection (cpu/mps/cuda with safe fallback)
    - checkpoint config (backward-compatible metadata)
    - experiment config (reproducible structure)
    - memory estimation (theoretical components sum correctly)

These tests do NOT modify the baseline. Smoke tests of forward/backward/step
are exercised by ``neural_core.phase2b_report`` and here for the CURRENT tier.
"""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from neural_core.model_config import ModelConfig
from neural_core.parameter_count import (
    theoretical_parameter_count,
    theoretical_breakdown,
    breakdown_from_config,
)
from neural_core.scaling_ladder import (
    get_ladder, REAL_VOCAB_SIZE, get_ladder_entry, DEFAULT_CONTEXT_SIZE,
)
from neural_core import model_factory
from neural_core import device as device_mod
from neural_core import checkpoint_metadata
from neural_core import experiment as experiment_mod
from neural_core import memory_estimation


# ----------------------------------------------------------------------
# Model factory & config validation
# ----------------------------------------------------------------------
def test_model_config_valid():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=128, num_heads=4,
                      d_ff=256, num_layers=2, max_len=256, dropout=0.1)
    assert cfg.d_k == 32
    assert cfg.context_size == 256


def test_model_config_serializable():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=128, num_heads=4,
                      d_ff=256, num_layers=2, max_len=256)
    cfg2 = ModelConfig.from_dict(cfg.to_dict())
    assert cfg2.to_dict() == cfg.to_dict()
    cfg3 = ModelConfig.from_json(cfg.to_json())
    assert cfg3.to_dict() == cfg.to_dict()


@pytest.mark.parametrize("kwargs", [
    {"d_model": 128, "num_heads": 3},        # 128 % 3 != 0
    {"d_model": 0, "num_heads": 1},          # d_model <= 0
    {"num_heads": 0},                        # num_heads <= 0
    {"num_layers": 0},                       # num_layers <= 0
    {"max_len": -1},                         # max_len <= 0
    {"vocab_size": 0},                       # vocab <= 0
    {"dropout": 1.0},                        # dropout out of range
    {"dropout": -0.1},                       # dropout out of range
    {"dtype": "float16"},                    # unsupported dtype in Phase 2B
    {"backend": "weird"},                    # unsupported backend
    {"device": "tpu"},                       # unsupported device
])
def test_invalid_config_rejected(kwargs):
    base = dict(vocab_size=REAL_VOCAB_SIZE, d_model=128, num_heads=4,
                d_ff=256, num_layers=2, max_len=256, dropout=0.1)
    base.update(kwargs)
    with pytest.raises(ValueError):
        ModelConfig(**base)


def test_factory_builds_numpy_model():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=64, num_heads=4,
                      d_ff=128, num_layers=1, max_len=64, backend="cpu", dtype="float64")
    model = model_factory.build_numpy_model(cfg)
    assert model.parameter_count() == theoretical_parameter_count(
        cfg.vocab_size, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers)


def test_factory_builds_torch_model():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=64, num_heads=4,
                      d_ff=128, num_layers=1, max_len=64, backend="torch", device="cpu")
    model = model_factory.build_torch_model(cfg, device="cpu")
    assert model.parameter_count() == theoretical_parameter_count(
        cfg.vocab_size, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers)


# ----------------------------------------------------------------------
# Parameter count: theoretical vs material, all ladder tiers
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", [e.name for e in get_ladder()])
def test_parameter_count_theoretical_vs_material(name):
    entry = get_ladder_entry(name)
    cfg = entry.to_config(backend="torch", device="cpu")
    theory = theoretical_parameter_count(
        cfg.vocab_size, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers)
    model = model_factory.build_torch_model(cfg, device="cpu")
    material = model.parameter_count()
    assert material == theory, f"{name}: material {material} != theory {theory}"


def test_current_baseline_param_count():
    # The protected baseline is exactly 2,472,846 parameters.
    entry = get_ladder_entry("CURRENT")
    cfg = entry.to_config()
    assert theoretical_parameter_count(
        cfg.vocab_size, cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.num_layers) == 2_472_846


def test_parameter_count_breakdown_sums():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=128, num_heads=4,
                      d_ff=256, num_layers=2, max_len=256)
    bd = breakdown_from_config(cfg)
    assert bd["embedding"] + bd["attention"] + bd["ffn"] + bd["layernorm"] + bd["output"] == bd["total"]


# ----------------------------------------------------------------------
# Scaling configurations
# ----------------------------------------------------------------------
def test_scaling_ladder_present():
    names = [e.name for e in get_ladder()]
    assert names == ["CURRENT", "SMALL", "MEDIUM", "LARGE", "OPTIONAL"]


def test_ladder_divisibility():
    for e in get_ladder():
        assert e.d_model % e.num_heads == 0


def test_ladder_progressive_params():
    counts = [
        theoretical_parameter_count(e.vocab_size, e.d_model, e.num_heads, e.d_ff, e.num_layers)
        for e in get_ladder()
    ]
    # strictly increasing across the ladder
    assert all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))


def test_ladder_vocab_is_real_8590():
    for e in get_ladder():
        assert e.vocab_size == REAL_VOCAB_SIZE == 8590


# ----------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------
def test_device_fallback_cpu():
    dev, info = device_mod.select_device("cpu")
    assert dev == "cpu"


def test_device_unknown_falls_back_to_cpu():
    dev, info = device_mod.select_device("does-not-exist")
    assert dev == "cpu"


def test_device_cuda_requested_but_unavailable_falls_back():
    # On a machine without CUDA this must safely fall back (not raise).
    dev, info = device_mod.select_device("cuda")
    assert dev in ("cuda", "mps", "cpu")
    # no crash; resolved is a valid device (cpu/mps/cuda)
    assert dev in ("cpu", "mps", "cuda")


# ----------------------------------------------------------------------
# Checkpoint metadata (backward compatible)
# ----------------------------------------------------------------------
def test_checkpoint_metadata_backward_compatible():
    cfg = ModelConfig(vocab_size=REAL_VOCAB_SIZE, d_model=64, num_heads=4,
                      d_ff=128, num_layers=1, max_len=64)
    model = model_factory.build_numpy_model(cfg)
    record = checkpoint_metadata.build_checkpoint_record(
        model,
        model_class="MicroTransformer",
        parameters=model.get_parameters(),
        model_config=cfg.to_dict(),
        tokenizer_identity={"vocab_size": REAL_VOCAB_SIZE},
        dataset_identity={"sha256": "abc"},
        seed=42,
        training_config={"optimizer": "Adam"},
    )
    # legacy loader contract preserved
    assert checkpoint_metadata.is_backward_compatible(record)
    assert "model_name" in record and "parameters" in record
    meta = record["checkpoint_metadata"]
    assert meta["model_class"] == "MicroTransformer"
    assert meta["vocab_size"] == REAL_VOCAB_SIZE
    assert meta["d_model"] == 64
    assert meta["num_layers"] == 1
    assert meta["parameter_count"] == model.parameter_count()


# ----------------------------------------------------------------------
# Experiment config (reproducible structure)
# ----------------------------------------------------------------------
def test_experiment_structure(tmp_path):
    exp = experiment_mod.Experiment("phase2b_test", root=tmp_path)
    exp.save_config(
        model_config={"d_model": 64},
        optimizer_config={"lr": 1e-3},
        training_config={"epochs": 1},
        seed=7,
        dataset_identity={"sha256": "x"},
        tokenizer_identity={"vocab_size": REAL_VOCAB_SIZE},
        backend="torch",
        dtype="float32",
        device="mps",
    )
    loaded = exp.load_config()
    assert loaded["seed"] == 7
    assert loaded["backend"] == "torch"
    assert (exp.path / "config" / "experiment.json").exists()
    for sub in experiment_mod.EXPERIMENT_SUBDIRS:
        assert (exp.path / sub).exists()


# ----------------------------------------------------------------------
# Memory estimation
# ----------------------------------------------------------------------
def test_memory_estimate_components_sum():
    pc = theoretical_parameter_count(REAL_VOCAB_SIZE, 128, 4, 256, 2)
    mem = memory_estimation.theoretical_gpu_memory(
        pc, REAL_VOCAB_SIZE, 128, 256, 4, 2, batch_size=8, seq_len=256, bytes_per_param=4)
    expected = (
        mem["parameter_memory_bytes"] + mem["gradient_memory_bytes"]
        + mem["adam_first_moment_bytes"] + mem["adam_second_moment_bytes"]
        + mem["activation_memory_bytes"] + mem["attention_workspace_bytes"]
        + mem["overhead_bytes"]
    )
    assert mem["total_estimated_bytes"] == expected


def test_memory_estimate_depends_on_context():
    pc = theoretical_parameter_count(REAL_VOCAB_SIZE, 128, 4, 256, 2)
    m256 = memory_estimation.theoretical_gpu_memory(
        pc, REAL_VOCAB_SIZE, 128, 256, 4, 2, batch_size=8, seq_len=256)
    m512 = memory_estimation.theoretical_gpu_memory(
        pc, REAL_VOCAB_SIZE, 128, 256, 4, 2, batch_size=8, seq_len=512)
    # context length does NOT change params but increases attention workspace
    assert m256["parameter_memory_bytes"] == m512["parameter_memory_bytes"]
    assert m512["attention_workspace_bytes"] > m256["attention_workspace_bytes"]


# ----------------------------------------------------------------------
# Smoke test: current tier forward / loss / backward / single step
# ----------------------------------------------------------------------
def test_smoke_current_torch_forward_backward_step():
    import torch
    entry = get_ladder_entry("CURRENT")
    cfg = entry.to_config(backend="torch", device="cpu")
    model = model_factory.build_torch_model(cfg, device="cpu")
    rng = np.random.RandomState(0)
    ids = torch.as_tensor(rng.randint(0, cfg.vocab_size, size=(2, 16)),
                          dtype=torch.long, device="cpu")
    tgt = torch.as_tensor(rng.randint(0, cfg.vocab_size, size=(32,)),
                          dtype=torch.long, device="cpu")
    out = model.forward(ids)
    loss = torch.nn.functional.cross_entropy(
        out.reshape(-1, cfg.vocab_size), tgt, reduction="mean")
    assert float(loss) > 0
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert model.parameter_count() == 2_472_846


def test_smoke_checkpoint_roundtrip_current():
    import pickle
    entry = get_ladder_entry("CURRENT")
    cfg = entry.to_config(backend="torch", device="cpu")
    model = model_factory.build_torch_model(cfg, device="cpu")
    params = model.get_parameters()
    reloaded = pickle.loads(pickle.dumps(params))
    model.set_parameters(reloaded)
    assert model.parameter_count() == 2_472_846
