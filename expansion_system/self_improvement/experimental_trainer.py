"""
Experimental training runner for JARVIS 2.0 self-improvement.

Runs isolated training experiments without modifying the production model.
Uses the existing Trainer, Dataset, Tokenizer, and checkpoint infrastructure.
"""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from expansion_system.self_improvement.training_strategy import TrainingStrategy
from neural_core.trainer import Trainer, build_model
from neural_core.tokenizer import Tokenizer
from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam
from neural_core.checkpoint_lineage import build_lineage, attach_to_checkpoint


class ExperimentalTrainingResult:
    """Result of an experimental training run."""

    def __init__(self, candidate_id: str, strategy: TrainingStrategy):
        self.candidate_id = candidate_id
        self.strategy = strategy
        self.status = "not_started"
        self.checkpoint_path = None
        self.lineage = None
        self.metrics = {}
        self.error = None
        self.resource_usage = {}
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy.strategy_id,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
            "lineage": self.lineage,
            "metrics": dict(self.metrics),
            "error": self.error,
            "resource_usage": dict(self.resource_usage),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ExperimentalTrainer:
    """Esegue training sperimentali isolati per candidate di self-improvement.

    Non modifica il modello production. Ogni training experiment:
    1. Parte da un checkpoint baseline
    2. Crea un checkpoint candidate isolato
    3. Registra la lineage completa
    4. Valida il checkpoint
    5. Restituisce metriche e percorso checkpoint
    """

    def __init__(self, base_dir: str = "/tmp/jarvis_training_experiments",
                 checkpoint_manager=None, tokenizer=None, dataset=None,
                 model_factory: Optional[Callable] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = checkpoint_manager
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model_factory = model_factory or self._default_model_factory

    def _is_allowed_checkpoint_path(self, checkpoint_path: str) -> bool:
        try:
            real_path = Path(checkpoint_path).resolve()
            return str(real_path).startswith(str(self.base_dir.resolve()))
        except Exception:
            return False

    def _safe_pickle_load(self, checkpoint_path: str) -> Dict[str, Any]:
        if not self._is_allowed_checkpoint_path(checkpoint_path):
            raise ValueError(f"Checkpoint path non autorizzato: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        try:
            from neural_core.trainer import _load_checkpoint_safe
            return _load_checkpoint_safe(checkpoint_path)
        except Exception:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Formato checkpoint non valido")
        return data

    def run_training(self, candidate_id: str, strategy: TrainingStrategy,
                     baseline_checkpoint: Optional[str] = None,
                     training_text: Optional[str] = None,
                     validation_text: Optional[str] = None,
                     benchmark_dataset: Optional[Dataset] = None,
                     tokenizer: Optional[Tokenizer] = None,
                     model_factory: Optional[Callable] = None,
                     max_timeout: float = 300.0,
                     max_memory_mb: Optional[float] = None) -> ExperimentalTrainingResult:
        result = ExperimentalTrainingResult(candidate_id, strategy)
        result.started_at = time.time()
        result.status = "TRAINING_STARTED"

        memory_before = self._get_memory_usage()
        start_time = time.time()

        try:
            if training_text is None and self.dataset is None:
                raise ValueError("training_text o dataset è obbligatorio per il training")

            if tokenizer is None:
                tokenizer = self.tokenizer or Tokenizer(max_vocab_size=200, mode="whitespace")

            if benchmark_dataset is not None:
                training_dataset = benchmark_dataset
                validation_dataset = benchmark_dataset
            elif training_text is not None:
                training_text_clean = " ".join(str(training_text).split())
                validation_text_clean = " ".join(str(validation_text).split()) if validation_text else training_text_clean
                tokenizer.fit(training_text_clean)
                training_ids = tokenizer.encode(training_text_clean)
                validation_ids = tokenizer.encode(validation_text_clean)
                context_size = strategy.parameters.get("context_size", 128)
                stride = strategy.parameters.get("stride", context_size)
                training_dataset = Dataset(training_ids, context_size=context_size, stride=stride)
                validation_dataset = Dataset(validation_ids, context_size=context_size, stride=stride)
            else:
                training_dataset = self.dataset
                validation_dataset = self.dataset

            params = strategy.parameters
            learning_rate = float(params.get("learning_rate", 0.001))
            batch_size = int(params.get("batch_size", 32))
            epochs = int(params.get("epochs", 1))
            context_size = params.get("context_size", 128)
            architecture = params.get("architecture", "transformer")
            model_config = {
                "architecture": architecture,
                "vocab_size": len(tokenizer.vocabulary),
                "d_model": 32,
                "num_heads": 2,
                "d_ff": 64,
                "num_layers": 1,
                "max_len": context_size,
                "seed": 42,
            }
            if architecture == "next_token":
                model_config["embedding_dim"] = 32
                model_config["hidden_dim"] = 64
                model_config.pop("d_model", None)
                model_config.pop("num_heads", None)
                model_config.pop("d_ff", None)
                model_config.pop("num_layers", None)
            factory = model_factory or self.model_factory
            model = factory(model_config)

            loss = CrossEntropyLoss()
            optimizer = Adam(learning_rate=learning_rate)
            trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)

            if baseline_checkpoint and os.path.exists(baseline_checkpoint):
                try:
                    trainer.load_checkpoint(baseline_checkpoint)
                    result.status = "BASELINE_LOADED"
                except Exception:
                    pass

            checkpoint_dir = self.base_dir / candidate_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"candidate-{candidate_id[:8]}.pkl"

            history = trainer.train_with_validation(
                training_inputs=training_dataset.get_inputs(),
                training_targets=training_dataset.get_targets(),
                validation_inputs=validation_dataset.get_inputs(),
                validation_targets=validation_dataset.get_targets(),
                epochs=epochs,
                batch_size=batch_size,
                patience=2,
                checkpoint_path=str(checkpoint_path),
                last_checkpoint_path=str(checkpoint_path),
            )

            if not checkpoint_path.exists():
                raise RuntimeError("Checkpoint non creato dopo il training")

            result.status = "CHECKPOINT_CREATED"
            lineage = build_lineage(
                model=model,
                model_class=type(model).__name__,
                model_config=model_config,
                tokenizer=tokenizer,
                seed=42,
                training_config={
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "context_size": context_size,
                    "architecture": "next_token",
                    "training_strategy": strategy.to_dict(),
                },
            )
            lineage["parent_checkpoint"] = baseline_checkpoint
            lineage["candidate_id"] = candidate_id
            lineage["training_status"] = "completed"

            with open(checkpoint_path, "rb") as f:
                checkpoint_data = self._safe_pickle_load(str(checkpoint_path))
            attach_to_checkpoint(checkpoint_data, lineage)

            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)

            result.status = "TRAINING_COMPLETED"
            result.checkpoint_path = str(checkpoint_path)
            result.lineage = lineage
            result.metrics = {
                "final_training_loss": history[-1]["training_loss"] if history else None,
                "final_validation_loss": history[-1]["validation_loss"] if history else None,
                "epochs_completed": len(history),
                "parameter_count": model.parameter_count() if hasattr(model, "parameter_count") else None,
            }

            elapsed = time.time() - start_time
            if elapsed > max_timeout:
                result.status = "TRAINING_TIMEOUT"
                result.error = f"Training exceeded timeout of {max_timeout}s"

            memory_after = self._get_memory_usage()
            result.resource_usage = {
                "elapsed_seconds": elapsed,
                "memory_delta_mb": max(0.0, memory_after - memory_before),
            }
            result.completed_at = time.time()

        except Exception as e:
            result.status = "TRAINING_ERROR"
            result.error = str(e)
            result.completed_at = time.time()

        return result

    def validate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        if not self._is_allowed_checkpoint_path(checkpoint_path):
            return {"valid": False, "error": "Checkpoint path non autorizzato (fuori dalla directory sperimentale)"}
        if not os.path.exists(checkpoint_path):
            return {"valid": False, "error": "Checkpoint non trovato"}
        try:
            data = self._safe_pickle_load(checkpoint_path)
            if not isinstance(data, dict):
                return {"valid": False, "error": "Formato checkpoint non valido"}
            if "parameters" not in data and "model_name" not in data:
                return {"valid": False, "error": "Checkpoint incompleto"}
            return {"valid": True, "checkpoint_size_bytes": os.path.getsize(checkpoint_path)}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _default_model_factory(self, config: Dict[str, Any]):
        from neural_core.trainer import build_model
        architecture = config.pop("architecture", "transformer")
        if architecture == "next_token":
            allowed = {"vocab_size", "embedding_dim", "hidden_dim", "dropout", "seed", "dtype"}
            filtered = {k: v for k, v in config.items() if k in allowed}
            return build_model(architecture, **filtered)
        return build_model(architecture, **config)

    def _get_memory_usage(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0
