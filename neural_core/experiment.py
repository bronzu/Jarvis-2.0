"""
Phase 2B — Experiment configuration management.

Prepares a reproducible structure for each scaling experiment:

    experiments/
        <experiment_id>/
            config/
            checkpoints/
            metrics/
            logs/
            reports/

Every configuration is reproducible: the experiment record stores the model
config, optimizer config, training config, seed, dataset identity, tokenizer
identity, backend, dtype and device. No training is performed here.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


EXPERIMENT_SUBDIRS = ("config", "checkpoints", "metrics", "logs", "reports")


def _default_root() -> Path:
    import os
    repo = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return repo / "experiments"


class Experiment:
    """A reproducible, isolated experiment directory under ``experiments/<id>``."""

    def __init__(self, experiment_id: str, root: Optional[Path] = None):
        if not experiment_id or "/" in experiment_id or "\\" in experiment_id:
            raise ValueError(f"Invalid experiment_id: {experiment_id!r}")
        self.experiment_id = experiment_id
        self.root = Path(root) if root is not None else _default_root()
        self.path = self.root / experiment_id

    # ------------------------------------------------------------------
    def ensure_directories(self) -> Path:
        for sub in EXPERIMENT_SUBDIRS:
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        return self.path

    # ------------------------------------------------------------------
    def save_config(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        training_config: Dict[str, Any],
        seed: Optional[int],
        dataset_identity: Dict[str, Any],
        tokenizer_identity: Dict[str, Any],
        backend: str,
        dtype: str,
        device: str,
    ) -> Path:
        self.ensure_directories()
        record = {
            "experiment_id": self.experiment_id,
            "model_config": model_config,
            "optimizer_config": optimizer_config,
            "training_config": training_config,
            "seed": seed,
            "dataset_identity": dataset_identity,
            "tokenizer_identity": tokenizer_identity,
            "backend": backend,
            "dtype": dtype,
            "device": device,
        }
        out = self.path / "config" / "experiment.json"
        out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    def load_config(self) -> Dict[str, Any]:
        out = self.path / "config" / "experiment.json"
        if not out.exists():
            raise FileNotFoundError(f"No experiment config at {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    def save_report(self, name: str, content: str) -> Path:
        self.ensure_directories()
        if not name.endswith((".md", ".txt", ".json")):
            name = name + ".md"
        out = self.path / "reports" / name
        out.write_text(content, encoding="utf-8")
        return out

    def save_metrics(self, name: str, data: Dict[str, Any]) -> Path:
        self.ensure_directories()
        if not name.endswith(".json"):
            name = name + ".json"
        out = self.path / "metrics" / name
        out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def checkpoint_dir(self) -> Path:
        self.ensure_directories()
        return self.path / "checkpoints"

    def log_dir(self) -> Path:
        self.ensure_directories()
        return self.path / "logs"

    # ------------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists()

    def remove(self):
        if self.path.exists():
            shutil.rmtree(self.path)
