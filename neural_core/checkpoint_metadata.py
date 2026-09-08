"""
Phase 2B — Checkpoint metadata (backward-compatible extension).

The historical checkpoint format (``neural_core.checkpoint.Checkpoint``) stores
``{"model_name": ..., "parameters": ...}``. This module EXTENDS it additively:
new checkpoints produced for scaling carry a nested ``"checkpoint_metadata"``
block. Because the keys are ADDITIVE, the existing loader continues to read
``model_name`` / ``parameters`` without change (backward compatible).

The metadata block records everything required to reproduce a run and to keep
the baseline format intact:

    model_class, model_config, vocab_size, d_model, num_heads, d_ff,
    num_layers, max_len, dtype, device (when relevant), tokenizer identity,
    dataset identity, parameter count, seed, training configuration.
"""

import time
from typing import Any, Dict, Optional


def build_checkpoint_record(
    model,
    model_class: str,
    parameters,
    model_config: Dict[str, Any],
    tokenizer_identity: Optional[Dict[str, Any]] = None,
    dataset_identity: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    training_config: Optional[Dict[str, Any]] = None,
    model_name: str = "jarvis_model",
    parameter_count: Optional[int] = None,
):
    """
    Build an extended checkpoint record that is readable by the legacy loader.

    The legacy fields ``model_name`` and ``parameters`` are preserved verbatim;
    all Phase 2B metadata lives under ``checkpoint_metadata``.
    """
    try:
        pc = int(parameter_count if parameter_count is not None else model.parameter_count())
    except Exception:
        pc = None

    cfg = dict(model_config)
    metadata = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_class": model_class,
        "model_config": cfg,
        "vocab_size": cfg.get("vocab_size"),
        "d_model": cfg.get("d_model"),
        "num_heads": cfg.get("num_heads"),
        "d_ff": cfg.get("d_ff"),
        "num_layers": cfg.get("num_layers"),
        "max_len": cfg.get("max_len"),
        "dtype": cfg.get("dtype"),
        "device": cfg.get("device"),
        "seed": seed,
        "parameter_count": pc,
        "tokenizer_identity": tokenizer_identity or {},
        "dataset_identity": dataset_identity or {},
        "training_config": training_config or {},
    }
    return {
        "model_name": model_name,
        "parameters": parameters,
        "checkpoint_metadata": metadata,
    }


def extract_metadata(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the Phase 2B metadata block, if present (None for legacy)."""
    return record.get("checkpoint_metadata")


def is_backward_compatible(record: Dict[str, Any]) -> bool:
    """True if a legacy loader can still read model_name + parameters."""
    return ("model_name" in record) and ("parameters" in record)
