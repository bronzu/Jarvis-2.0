"""
Phase 2B — Device selection with safe fallback.

The heavy training target is an NVIDIA/CUDA GPU on a cloud environment (e.g.
Kaggle). This module resolves a *preferred* device into an actually-available
one, with explicit, safe fallback:

    cuda  -> mps  -> cpu
    mps   -> cpu
    cpu   -> cpu

CUDA is NOT assumed available locally. A request for CUDA that is unavailable
is reported (not treated as a fatal error) so callers can decide. The CPU
NumPy reference and the MPS/CPU PyTorch backends must all remain usable.
"""

from typing import Optional, Tuple


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def select_device(preferred: Optional[str] = None) -> Tuple[str, dict]:
    """
    Resolve ``preferred`` into an available device.

    Returns ``(device, info)`` where ``info`` records what was requested and
    what fallback occurred. ``preferred`` may be ``None`` (auto).
    """
    info = {"requested": preferred, "fallbacks": [], "cuda_available": cuda_available(),
            "mps_available": mps_available()}

    preferred = (preferred or "auto").lower()

    if preferred in ("auto", "cuda"):
        if cuda_available():
            info["resolved"] = "cuda"
            return "cuda", info
        info["fallbacks"].append("cuda->unavailable")
        if preferred == "cuda":
            # caller explicitly asked for cuda; report, then fall back safely
            if mps_available():
                info["fallbacks"].append("cuda->mps")
                info["resolved"] = "mps"
                return "mps", info
            info["fallbacks"].append("cuda->cpu")
            info["resolved"] = "cpu"
            return "cpu", info

    if preferred in ("auto", "mps", "gpu"):
        if mps_available():
            info["resolved"] = "mps"
            return "mps", info
        if preferred == "mps":
            info["fallbacks"].append("mps->cpu")
            info["resolved"] = "cpu"
            return "cpu", info

    # cpu or anything else
    info["resolved"] = "cpu"
    return "cpu", info


def backend_for_device(device: str) -> str:
    """Map a resolved device to the torch backend name used by the factory."""
    if device == "cuda":
        return "torch"
    if device == "mps":
        return "torch"
    return "cpu"
