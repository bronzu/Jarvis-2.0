"""
Phase 2B — Explicit, parametric model configuration for JARVIS 2.0.

This module defines :class:`ModelConfig`, a fully serializable, validated
description of the decoder-only micro-transformer. It is the single source of
truth for *which* model to build and *how* to build it, supporting the model
scaling preparation without changing the underlying architecture.

Architecture invariants preserved (see Phase 2B constraints):

    - decoder-only, causal self-attention, autoregressive next-token;
    - tanh-based GELU (implemented in the backends, unchanged);
    - manual LayerNorm with ``eps=1e-6`` (implemented in the backends);
    - embedding kept SEPARATE from the output projection (NO weight tying);
    - same positional encoding, dropout, Adam and CrossEntropyLoss semantics.

This module ONLY describes and validates configuration. It does not touch the
CPU NumPy reference (``neural_core.network``) or the GPU backend
(``neural_core.torch_model``); the model factory builds those unchanged.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


#: Backend identifiers supported by the factory.
SUPPORTED_BACKENDS = ("cpu", "numpy", "gpu", "torch", "mps", "cuda")

#: Device identifiers the factory can resolve / fall back to.
SUPPORTED_DEVICES = ("cpu", "mps", "cuda")


def _validate_positive_int(value, name):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {value}")
    return value


def _validate_nonneg_int(value, name):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


@dataclass
class ModelConfig:
    """
    Explicit, serializable description of a JARVIS decoder-only transformer.

    Every field is required and validated on construction. The configuration is
    backend-agnostic: the same config can be built by the NumPy CPU reference
    (float64) or by the PyTorch backend (float32, MPS/CUDA).
    """

    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    max_len: int
    dropout: float = 0.1
    dtype: str = "float64"
    seed: Optional[int] = None
    backend: str = "cpu"
    device: str = "cpu"

    def __post_init__(self):
        self.vocab_size = _validate_positive_int(self.vocab_size, "vocab_size")
        self.d_model = _validate_positive_int(self.d_model, "d_model")
        self.num_heads = _validate_positive_int(self.num_heads, "num_heads")
        self.d_ff = _validate_positive_int(self.d_ff, "d_ff")
        self.num_layers = _validate_positive_int(self.num_layers, "num_layers")
        self.max_len = _validate_positive_int(self.max_len, "max_len")

        try:
            self.dropout = float(self.dropout)
        except (TypeError, ValueError):
            raise ValueError(f"dropout must be a float, got {self.dropout!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        if self.dtype not in ("float64", "float32"):
            raise ValueError(
                f"dtype must be 'float64' or 'float32' for Phase 2B, got {self.dtype!r}"
            )

        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}"
            )
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"device must be one of {SUPPORTED_DEVICES}, got {self.device!r}"
            )

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads "
                f"({self.num_heads})"
            )

        if self.seed is not None:
            self.seed = _validate_nonneg_int(self.seed, "seed")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def d_k(self) -> int:
        return self.d_model // self.num_heads

    @property
    def context_size(self) -> int:
        return self.max_len

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        cleaned = {k: v for k, v in dict(data).items() if k in known}
        return cls(**cleaned)

    @classmethod
    def from_json(cls, text: str) -> "ModelConfig":
        import json
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        return (
            f"ModelConfig(vocab={self.vocab_size}, d_model={self.d_model}, "
            f"heads={self.num_heads}, d_ff={self.d_ff}, layers={self.num_layers}, "
            f"max_len={self.max_len}, dropout={self.dropout}, dtype={self.dtype})"
        )
