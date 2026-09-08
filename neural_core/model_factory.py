"""
Phase 2B — Model factory.

Builds a JARVIS decoder-only transformer from a :class:`ModelConfig` and a
backend, WITHOUT modifying the CPU NumPy reference or the GPU backend. Both
backends are imported unchanged; the factory only validates, selects the device
with safe fallback, and wires the config through.

Supported backends:

    "cpu" / "numpy" -> NumPy CPU reference (float64, authoritative baseline)
    "gpu" / "torch" / "mps" / "cuda" -> PyTorch backend (float32, MPS/CUDA)

Device resolution falls back safely (cuda -> mps -> cpu) and never assumes CUDA
is present. The factory is the single, explicit entry point for constructing
scaling-ladder models.
"""

from typing import Optional, Tuple

from neural_core.model_config import ModelConfig, SUPPORTED_BACKENDS
from neural_core import device as device_mod


def _normalize_backend(backend: str) -> str:
    backend = backend.lower()
    if backend in ("cpu", "numpy", "numpy_cpu"):
        return "cpu"
    if backend in ("gpu", "torch", "torch_gpu", "mps", "cuda"):
        return "torch"
    raise ValueError(f"Unknown backend: {backend}")


def build_numpy_model(config: ModelConfig):
    """Build the CPU NumPy reference model from a config (float64, unchanged)."""
    from neural_core.network import MicroTransformer
    return MicroTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        max_len=config.max_len,
        dropout=config.dropout,
        seed=config.seed,
        dtype=__import__("numpy").float64,
    )


def build_torch_model(config: ModelConfig, device: Optional[str] = None):
    """Build the PyTorch backend model from a config (float32, MPS/CUDA/CPU)."""
    import torch
    from neural_core.torch_model import TorchMicroTransformer

    resolved_device, _ = device_mod.select_device(device or config.device)
    return TorchMicroTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        max_len=config.max_len,
        dropout=config.dropout,
        seed=config.seed,
        dtype=torch.float32,
        device=resolved_device,
    )


def build_model(config: ModelConfig, backend: Optional[str] = None,
                device: Optional[str] = None):
    """
    Build a model from a validated :class:`ModelConfig`.

    Args:
        config:  validated model configuration (ModelConfig).
        backend: optional backend override ("cpu"/"numpy"/"gpu"/"torch"/"mps").
        device:  optional preferred device ("cpu"/"mps"/"cuda").

    Returns:
        A backend model instance (NumPy ``MicroTransformer`` or PyTorch
        ``TorchMicroTransformer``).
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig instance")

    chosen = _normalize_backend(backend or config.backend)

    if chosen == "cpu":
        return build_numpy_model(config)
    return build_torch_model(config, device=device)


def build_backend(config: ModelConfig, backend: Optional[str] = None,
                  device: Optional[str] = None):
    """
    Build the backend wrapper (``NumpyBackend`` / ``TorchBackend``) used by the
    parity tests and benchmarks. Validates the config and resolves the device.
    """
    from neural_core import backend as backend_mod

    chosen = _normalize_backend(backend or config.backend)
    if chosen == "cpu":
        return backend_mod.NumpyBackend(config.to_dict())
    resolved_device, _ = device_mod.select_device(device or config.device)
    return backend_mod.TorchBackend(config.to_dict(), device=resolved_device)
