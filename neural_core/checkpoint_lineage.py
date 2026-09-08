"""
Checkpoint lineage (extended metadata) for JARVIS 2.0.

Adds lineage fields to the existing checkpoint record:
  - git_commit (if available)
  - code_version (from VERSION file or hardcoded)
  - architecture_version
  - model_config
  - tokenizer_version / state
  - dataset_identity / hash
  - seed
  - python_version
  - numpy_version
  - timestamp (ISO 8601 UTC)

The schema is additive and backward-compatible with the legacy checkpoint
format. Missing fields are recorded as null, never fabricated.
"""

import datetime
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_VERSION = "jarvis2-microtransformer-v1"
CODE_VERSION_FILE_CANDIDATES = [
    REPO_ROOT / "DOCUMENTAZIONE" / "versionamento" / "VERSIONE.md",
    REPO_ROOT / "VERSIONE.md",
]


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(REPO_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _code_version() -> Optional[str]:
    for path in CODE_VERSION_FILE_CANDIDATES:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        in_current = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## Versione corrente"):
                in_current = True
                continue
            if in_current and stripped.startswith("##"):
                in_current = False
                continue
            if in_current and stripped.startswith("**") and stripped.endswith("**"):
                return stripped.strip("*").strip()
        import re as _re
        m = _re.search(r"\*\*(\d+\.\d+\.\d+(?:[-+][\w.]+)?)\*\*", text)
        if m:
            return m.group(1)
    return None


def _file_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        return None
    return h.hexdigest()


def _dataset_identity(data_dir: Path = Path("neural_core/data")) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in ("train.txt", "validation.txt", "test.txt"):
        path = data_dir / split
        if path.exists():
            out[split] = {
                "size_bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
    provenance = data_dir / "provenance_sentences.jsonl"
    if provenance.exists():
        out["provenance_sentences"] = {
            "size_bytes": provenance.stat().st_size,
            "sha256": _file_hash(provenance),
        }
    return out


def _tokenizer_identity(tokenizer) -> Dict[str, Any]:
    if tokenizer is None:
        return {}
    return {
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "max_vocab_size": getattr(tokenizer, "max_vocab_size", None),
        "mode": getattr(tokenizer, "mode", None),
        "class": type(tokenizer).__name__,
    }


def build_lineage(
    model,
    model_class: str,
    model_config: Dict[str, Any],
    tokenizer=None,
    dataset_dir: Path = Path("neural_core/data"),
    seed: Optional[int] = None,
    training_config: Optional[Dict[str, Any]] = None,
    architecture_version: str = ARCHITECTURE_VERSION,
) -> Dict[str, Any]:
    """Build a complete checkpoint lineage record.

    Each field is the real value read from the runtime/environment. If a value
    is unavailable, the field is set to None (never fabricated).
    """
    try:
        param_count = int(model.parameter_count())
    except Exception:
        param_count = None

    return {
        "git_commit": _git_commit(),
        "code_version": _code_version(),
        "architecture_version": architecture_version,
        "model_class": model_class,
        "model_config": dict(model_config),
        "tokenizer_identity": _tokenizer_identity(tokenizer),
        "dataset_identity": _dataset_identity(dataset_dir),
        "parameter_count": param_count,
        "seed": seed,
        "training_config": training_config or {},
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


def attach_to_checkpoint(checkpoint_record: dict, lineage: dict) -> dict:
    """Attach lineage to a checkpoint record (additive, backward-compatible)."""
    if "checkpoint_metadata" not in checkpoint_record:
        checkpoint_record["checkpoint_metadata"] = {}
    checkpoint_record["checkpoint_metadata"].update(lineage)
    return checkpoint_record
