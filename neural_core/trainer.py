"""
Trainer del Neural Core di JARVIS.

Supporta indifferentemente i due modelli proprietari:

- `NextTokenModel` (questo modulo): modello a contesto singolo, leggero.
- `neural_core.network.MicroTransformer`: Transformer autoregressivo con
  context window configurabile.

Entrambi rispettano lo stesso protocollo::

    forward(token_ids, training=False) -> logits
    backward(targets)                  -> list[np.ndarray]
    get_parameters()                   -> list[np.ndarray]
    set_parameters(list[np.ndarray])   -> None
    config()                           -> dict

Funzionalità di training:
    - mini batch
    - gradient accumulation
    - learning rate scheduler con warmup
    - gradient clipping
    - checkpoint completo (modello + ottimizzatore + scheduler + stato)
    - resume training
    - best model tracking
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from neural_core.dataset import Dataset
from neural_core.tokenizer import Tokenizer
from neural_core.loss import CrossEntropyLoss
from neural_core.model import gelu, gelu_derivative
from neural_core.optimizer import Adam, LearningRateScheduler


def _derive_seed(master_seed, salt):
    """Deriva un seed deterministico da un master seed e un salt."""
    if master_seed is None:
        return None
    digest = hashlib.md5(f"{master_seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)


class NextTokenModel:
    """
    Modello a contesto singolo: embedding -> GELU -> logits.

    Accetta sia input `(N,)` che sequenze `(batch, seq_len)`; in quest'ultimo
    caso produce logits `(batch, seq_len, vocab_size)` come il Transformer,
    così il `Trainer` può usare gli stessi dati per entrambe le architetture.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        seed=None,
        dtype=np.float64,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size deve essere maggiore di 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim deve essere maggiore di 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim deve essere maggiore di 0")
        if not 0 <= dropout < 1:
            raise ValueError("dropout deve essere compreso tra 0 e 1")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.seed = seed
        self.dtype = np.dtype(dtype)

        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._initial_rng_state = self.rng.get_state() if seed is not None else None

        self.embeddings = self.rng.randn(vocab_size, embedding_dim).astype(self.dtype) * 0.02
        self.W1 = self.rng.randn(embedding_dim, hidden_dim).astype(self.dtype) * 0.02
        self.b1 = np.zeros(hidden_dim, dtype=self.dtype)
        self.W2 = self.rng.randn(hidden_dim, vocab_size).astype(self.dtype) * 0.02
        self.b2 = np.zeros(vocab_size, dtype=self.dtype)

        self._cached = {}

    #: Il modello a contesto singolo non ha limiti di sequenza.
    max_len = None

    @property
    def context_size(self):
        return None

    def config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "seed": self.seed,
            "dtype": str(self.dtype),
        }

    def forward(self, token_ids, training: bool = False) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=int)
        if token_ids.ndim == 0:
            token_ids = token_ids.reshape(1)

        embedded = self.embeddings[token_ids]
        flat_embedded = embedded.reshape(-1, self.embedding_dim)

        z1 = flat_embedded @ self.W1 + self.b1
        h1 = gelu(z1)
        if training and self.dropout > 0:
            if self._initial_rng_state is not None:
                self.rng.set_state(self._initial_rng_state)
            mask = self.rng.binomial(1, 1 - self.dropout, size=h1.shape) / (1 - self.dropout)
            h1 = h1 * mask
        flat_logits = h1 @ self.W2 + self.b2
        logits = flat_logits.reshape(token_ids.shape + (self.vocab_size,))

        self._cached = {
            "token_ids": token_ids,
            "flat_embedded": flat_embedded,
            "z1": z1,
            "h1": h1,
            "logits": logits,
        }
        return logits

    def backward(self, targets, ignore_index: int = -100, probs=None) -> List[np.ndarray]:
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        token_ids = self._cached["token_ids"]
        logits = self._cached["logits"]
        h1 = self._cached["h1"]
        flat_embedded = self._cached["flat_embedded"]

        targets = np.asarray(targets, dtype=int)
        if targets.shape != token_ids.shape:
            targets = targets.reshape(token_ids.shape)

        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)

        if probs is None:
            probs = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
        else:
            probs = probs.copy()

        valid = flat_targets != ignore_index
        valid_count = int(np.sum(valid))
        if valid_count == 0:
            raise ValueError("Nessun target valido nella batch.")

        safe_targets = np.where(valid, flat_targets, 0)
        probs[np.arange(flat_logits.shape[0]), safe_targets] -= 1
        probs[~valid] = 0.0
        grad_logits = probs / valid_count

        grad_W2 = h1.T @ grad_logits
        grad_b2 = np.sum(grad_logits, axis=0)
        grad_h1 = grad_logits @ self.W2.T

        grad_z1 = grad_h1 * gelu_derivative(self._cached["z1"])
        grad_W1 = flat_embedded.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)
        grad_embedded = grad_z1 @ self.W1.T

        full_grad_embeddings = np.zeros_like(self.embeddings, dtype=float)
        np.add.at(full_grad_embeddings, token_ids.reshape(-1), grad_embedded)

        return [full_grad_embeddings, grad_W1, grad_b1, grad_W2, grad_b2]

    def get_parameters(self) -> List[np.ndarray]:
        return [self.embeddings, self.W1, self.b1, self.W2, self.b2]

    def set_parameters(self, parameters) -> None:
        parameters = list(parameters)
        if len(parameters) != 5:
            raise ValueError("Expected 5 parameters")
        self.embeddings = parameters[0]
        self.W1 = parameters[1]
        self.b1 = parameters[2]
        self.W2 = parameters[3]
        self.b2 = parameters[4]

    def parameter_count(self) -> int:
        return int(sum(np.asarray(p).size for p in self.get_parameters()))

    def parameter_count_breakdown(self) -> dict:
        total = self.parameter_count()
        return {
            "total": total,
            "trainable": total,
            "non_trainable": 0,
            "embedding": int(self.embeddings.size),
            "attention": 0,
            "ffn": int(self.W1.size + self.b1.size),
            "layernorm": 0,
            "output": int(self.W2.size + self.b2.size),
        }


# ----------------------------------------------------------------------
# Registro delle architetture proprietarie
# ----------------------------------------------------------------------

def build_model(architecture: str = "next_token", **config):
    """
    Costruisce un modello proprietario del Neural Core.

    Args:
        architecture: ``"next_token"`` oppure ``"transformer"``.
        **config: iperparametri dell'architettura scelta.
    """
    normalized = str(architecture).lower()

    if normalized in {"next_token", "nexttokenmodel"}:
        return NextTokenModel(**config)

    if normalized in {"transformer", "microtransformer"}:
        from neural_core.network import MicroTransformer
        return MicroTransformer(**config)

    raise ValueError(
        "architecture non supportata: usa 'next_token' o 'transformer'."
    )


def load_model_from_weights(path, vocab_size: Optional[int] = None):
    """
    Ricostruisce un modello dai pesi salvati da `Trainer.save_weights()`.

    Supporta sia il formato corrente (con `model_class` e `parameters`) sia il
    formato storico di `NextTokenModel` (`embeddings`, `W1`, `b1`, `W2`, `b2`).

    Returns:
        Il modello con i pesi caricati.
    """
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("File dei pesi non valido.")

    model_class = payload.get("model_class")
    config = dict(payload.get("config") or {})

    if model_class == "MicroTransformer":
        from neural_core.network import MicroTransformer
        if not config.get("vocab_size") and vocab_size is not None:
            config["vocab_size"] = vocab_size
        model = MicroTransformer(**config)
        model.set_parameters([np.asarray(p) for p in payload["parameters"]])
        return model

    stored_vocab_size = payload.get("vocab_size") or config.get("vocab_size")
    legacy_config = {
        "vocab_size": stored_vocab_size or vocab_size,
        "embedding_dim": payload.get("embedding_dim", config.get("embedding_dim", 64)),
        "hidden_dim": payload.get("hidden_dim", config.get("hidden_dim", 128)),
        "dropout": payload.get("dropout", config.get("dropout", 0.1)),
    }
    if not legacy_config["vocab_size"]:
        raise ValueError("vocab_size non determinabile dal file dei pesi.")

    model = NextTokenModel(**legacy_config)

    if isinstance(payload.get("parameters"), list):
        model.set_parameters([np.asarray(p) for p in payload["parameters"]])
    elif "embeddings" in payload:
        model.embeddings = np.asarray(payload["embeddings"])
        model.W1 = np.asarray(payload["W1"])
        model.b1 = np.asarray(payload["b1"])
        model.W2 = np.asarray(payload["W2"])
        model.b2 = np.asarray(payload["b2"])
    else:
        raise ValueError("Il file dei pesi non contiene parametri riconoscibili.")

    return model


class Trainer:
    """
    Trainer del Neural Core, compatibile con tutte le architetture proprietarie.

    Args:
        model: modello proprietario (`NextTokenModel` o `MicroTransformer`).
        tokenizer: tokenizer interno.
        loss: funzione di loss interna.
        optimizer: ottimizzatore interno.
        scheduler: `LearningRateScheduler` opzionale (warmup + decay).
        gradient_accumulation_steps: numero di micro-batch accumulati prima di
            un update dei pesi.
        max_grad_norm: soglia di gradient clipping globale (`None` = disattivo).
    """

    CHECKPOINT_FORMAT = "jarvis-neural-core-checkpoint"
    CHECKPOINT_VERSION = 1

    def __init__(self, model, tokenizer, loss, optimizer, scheduler=None,
                 gradient_accumulation_steps: int = 1, max_grad_norm=None, seed=None):
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps deve essere almeno 1.")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm deve essere maggiore di 0.")

        self.model = model
        self.tokenizer = tokenizer
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        self.global_step = 0
        self.epoch = 0
        self.best_validation_loss = float("inf")
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    # Costruzione da corpus
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str,
        tokenizer=None,
        validation_split: float = 0.2,
        test_split: float = 0.0,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        architecture: str = "next_token",
        context_size: Optional[int] = None,
        stride: Optional[int] = None,
        learning_rate: float = 0.001,
        warmup_steps: int = 0,
        total_steps: Optional[int] = None,
        schedule: str = "constant",
        gradient_accumulation_steps: int = 1,
        max_grad_norm=None,
        seed=None,
        **model_config,
    ):
        if tokenizer is None:
            tokenizer = Tokenizer()

        training_dataset, validation_dataset, *test_dataset = Dataset.from_text(
            Path(corpus_path).read_text(encoding="utf-8"),
            tokenizer,
            validation_split,
            test_split,
            shuffle_blocks=True,
            context_size=context_size,
            stride=stride,
            seed=seed,
        )

        vocab_size = len(tokenizer.vocabulary)

        if str(architecture).lower() in {"transformer", "microtransformer"}:
            config = {"vocab_size": vocab_size}
            if context_size is not None:
                config["max_len"] = max(context_size, model_config.pop("max_len", context_size))
            config["dropout"] = dropout
            config["seed"] = seed
            config.update(model_config)
            model = build_model("transformer", **config)
        else:
            model = build_model(
                "next_token",
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                seed=seed,
            )

        loss = CrossEntropyLoss()
        optimizer = Adam(learning_rate=learning_rate)
        scheduler = None
        if warmup_steps > 0 or total_steps is not None or schedule != "constant":
            scheduler = LearningRateScheduler(
                base_learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                schedule=schedule,
            )

        trainer = cls(
            model,
            tokenizer,
            loss,
            optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            seed=seed,
        )

        if test_dataset:
            return trainer, training_dataset, validation_dataset, test_dataset[0]

        return trainer, training_dataset, validation_dataset

    @classmethod
    def from_dataset(cls, dataset, tokenizer, architecture: str = "transformer", **kwargs):
        """
        Costruisce un Trainer coerente con un `Dataset` già pronto.

        Se il dataset è in modalità sequenza, la context window del modello
        viene allineata alla `context_size` del dataset.
        """
        model_config = kwargs.pop("model_config", {}) or {}
        vocab_size = len(tokenizer.vocabulary)
        model_config.setdefault("vocab_size", vocab_size)

        if dataset.is_windowed():
            model_config.setdefault("max_len", dataset.context_size)

        model = build_model(architecture, **model_config)
        loss = kwargs.pop("loss", None) or CrossEntropyLoss()
        optimizer = kwargs.pop("optimizer", None) or Adam(
            learning_rate=kwargs.pop("learning_rate", 0.001)
        )
        return cls(model, tokenizer, loss, optimizer, **kwargs)

    # ------------------------------------------------------------------
    # Preparazione dei batch
    # ------------------------------------------------------------------

    def _prepare_batch(self, inputs, targets):
        """
        Normalizza input e target.

        - target 1D  -> modalità token singolo, gli input vengono appiattiti.
        - target 2D  -> modalità sequenza, input e target `(N, context_size)`.
        """
        target_ids = np.asarray(targets, dtype=int)

        if target_ids.ndim >= 2:
            input_ids = np.asarray(inputs, dtype=int)
            if input_ids.shape != target_ids.shape:
                raise ValueError(
                    "In modalità sequenza input e target devono avere la stessa forma."
                )
            return input_ids, target_ids

        if isinstance(inputs, np.ndarray):
            input_ids = inputs.astype(int)
            if input_ids.ndim > 1:
                input_ids = input_ids[:, 0]
        else:
            input_ids = np.asarray(
                [
                    ids[0] if isinstance(ids, (list, tuple, np.ndarray)) else ids
                    for ids in inputs
                ],
                dtype=int,
            )
        return input_ids, target_ids

    # ------------------------------------------------------------------
    # Gradienti
    # ------------------------------------------------------------------

    @staticmethod
    def _accumulate_gradients(buffer, gradients, scale):
        scaled = [np.asarray(g, dtype=float) * scale for g in gradients]
        if buffer is None:
            return scaled
        if len(buffer) != len(scaled):
            raise ValueError("Numero di gradienti incoerente tra i micro-batch.")
        return [existing + new for existing, new in zip(buffer, scaled)]

    def _clip_gradients(self, gradients):
        """Gradient clipping sulla norma globale."""
        if self.max_grad_norm is None:
            return gradients, None

        total = float(np.sqrt(sum(float(np.sum(np.square(g))) for g in gradients)))
        if total > self.max_grad_norm and total > 0:
            factor = self.max_grad_norm / total
            gradients = [g * factor for g in gradients]
        return gradients, total

    def _current_learning_rate(self):
        if self.scheduler is None:
            return getattr(self.optimizer, "learning_rate", None)

        learning_rate = self.scheduler.learning_rate(self.global_step)
        self.scheduler.last_step = self.global_step
        if hasattr(self.optimizer, "set_learning_rate"):
            self.optimizer.set_learning_rate(learning_rate)
        else:
            self.optimizer.learning_rate = learning_rate
        return learning_rate

    def _apply_update(self, gradients):
        gradients, _ = self._clip_gradients(gradients)
        self._current_learning_rate()
        parameters = self.model.get_parameters()
        updated = self.optimizer.update(parameters, gradients)
        self.model.set_parameters(updated)
        self.global_step += 1

    # ------------------------------------------------------------------
    # Step di training
    # ------------------------------------------------------------------

    def train_step(self, batch_inputs, batch_targets) -> float:
        """
        Un singolo step di training (forward + backward + update immediato).

        Returns:
            La loss del batch.
        """
        input_ids, target_ids = self._prepare_batch(batch_inputs, batch_targets)
        logits = self.model.forward(input_ids, training=True)
        loss_value = float(self.loss.forward(logits, target_ids))
        probs = getattr(self.loss, "get_probabilities", lambda: None)()
        if probs is not None:
            gradients = self.model.backward(target_ids, probs=probs)
        else:
            gradients = self.model.backward(target_ids)
        self._apply_update([np.asarray(g, dtype=float) for g in gradients])
        return loss_value

    def _run_epoch(self, inputs, targets, batch_size, training=True, shuffle=True):
        input_ids, target_ids = self._prepare_batch(inputs, targets)
        n_samples = len(input_ids)

        if n_samples == 0:
            raise ValueError("Il dataset non può essere vuoto.")

        indices = np.arange(n_samples)
        if training and shuffle:
            if self.seed is not None:
                epoch_seed = (self.seed + self.epoch) % (2**31)
                rng = np.random.RandomState(epoch_seed)
                rng.shuffle(indices)
            else:
                np.random.shuffle(indices)

        epoch_losses = []
        accumulated = None
        micro_batches = 0
        starts = list(range(0, n_samples, batch_size))

        for position, start in enumerate(starts):
            batch_idx = indices[start:start + batch_size]
            batch_inputs = input_ids[batch_idx]
            batch_targets = target_ids[batch_idx]

            logits = self.model.forward(batch_inputs, training=training)
            loss_value = float(self.loss.forward(logits, batch_targets))
            epoch_losses.append(loss_value)

            if training and (position + 1) % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                print(
                    f"  Step {position + 1}/{len(starts)} - batch loss: {loss_value:.4f} - avg(last 10): {avg_loss:.4f}",
                    flush=True,
                )

            if not training:
                continue

            probs = getattr(self.loss, "get_probabilities", lambda: None)()
            if probs is not None:
                gradients = self.model.backward(batch_targets, probs=probs)
            else:
                gradients = self.model.backward(batch_targets)
            accumulated = self._accumulate_gradients(
                accumulated, gradients, 1.0
            )
            micro_batches += 1

            is_last = position == len(starts) - 1
            if micro_batches >= self.gradient_accumulation_steps or is_last:
                scaled = [g / micro_batches for g in accumulated]
                self._apply_update(scaled)
                accumulated = None
                micro_batches = 0

        return sum(epoch_losses) / len(epoch_losses)

    def train(self, inputs, targets, epochs: int = 1, batch_size: int = 64) -> List[float]:
        if len(inputs) != len(targets):
            raise ValueError("Inputs e targets devono avere la stessa lunghezza.")

        if epochs < 1:
            raise ValueError("epochs deve essere almeno 1.")

        history = []
        for _ in range(epochs):
            average_loss = self._run_epoch(inputs, targets, batch_size, training=True)
            history.append(average_loss)
            self.epoch += 1

        return history

    def evaluate(self, inputs, targets, batch_size: int = 64) -> float:
        if len(inputs) != len(targets):
            raise ValueError("Inputs e targets devono avere la stessa lunghezza.")

        if len(inputs) == 0:
            raise ValueError("Il dataset di valutazione non può essere vuoto.")

        return self._run_epoch(inputs, targets, batch_size, training=False, shuffle=False)

    def train_with_validation(
        self,
        training_inputs,
        training_targets,
        validation_inputs,
        validation_targets,
        epochs: int = 1,
        batch_size: int = 64,
        patience: int = 5,
        checkpoint_path: Optional[str] = None,
        last_checkpoint_path: Optional[str] = None,
        resume_path: Optional[str] = None,
        on_epoch_end=None,
    ) -> List[dict]:
        if len(training_inputs) != len(training_targets):
            raise ValueError("Training inputs e targets devono avere la stessa lunghezza.")

        if len(validation_inputs) != len(validation_targets):
            raise ValueError("Validation inputs e targets devono avere la stessa lunghezza.")

        if len(training_inputs) == 0:
            raise ValueError("Il Training Dataset non può essere vuoto.")

        if len(validation_inputs) == 0:
            raise ValueError("Il Validation Dataset non può essere vuoto.")

        if epochs < 1:
            raise ValueError("epochs deve essere almeno 1.")

        if resume_path is not None:
            self.load_checkpoint(resume_path)

        history = []
        best_params = None
        patience_counter = 0

        for _ in range(epochs):
            train_loss = self._run_epoch(training_inputs, training_targets, batch_size, training=True)
            val_loss = self.evaluate(validation_inputs, validation_targets, batch_size=batch_size)
            self.epoch += 1

            entry = {
                "epoch": self.epoch,
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": getattr(self.optimizer, "learning_rate", None),
            }
            history.append(entry)
            self.history.append(entry)

            is_best = val_loss < self.best_validation_loss
            if is_best:
                self.best_validation_loss = val_loss
                best_params = [np.asarray(p).copy() for p in self.model.get_parameters()]
                patience_counter = 0
                if checkpoint_path is not None:
                    self._save_best_checkpoint(checkpoint_path, best_params)
            else:
                patience_counter += 1

            if on_epoch_end is not None:
                on_epoch_end(
                    epoch=self.epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    trainer=self,
                    is_best=is_best,
                    patience_counter=patience_counter,
                )

            if patience_counter >= patience:
                break

        if last_checkpoint_path is not None:
            self.save_checkpoint(last_checkpoint_path)

        if best_params is not None:
            self.model.set_parameters(best_params)

        return history

    # ------------------------------------------------------------------
    # Persistenza
    # ------------------------------------------------------------------

    def _model_config(self) -> dict:
        if hasattr(self.model, "config"):
            return dict(self.model.config())
        return {}

    def _save_best_checkpoint(self, path, params):
        """Salva il best model in formato JSON leggibile."""
        checkpoint = {
            "model_name": "jarvis_neural_core",
            "model_class": type(self.model).__name__,
            "parameters": {
                "config": self._model_config(),
                "best_validation_loss": self.best_validation_loss,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "tensors": [np.asarray(p).tolist() for p in params],
            },
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(checkpoint, file)
        return str(path)

    def load_best_checkpoint(self, path):
        """Carica i pesi dal best checkpoint JSON."""
        with open(path, "r", encoding="utf-8") as file:
            checkpoint = json.load(file)

        parameters = checkpoint.get("parameters", {})
        tensors = parameters.get("tensors")
        if tensors is None:
            raise ValueError("Best checkpoint senza tensori.")

        self.model.set_parameters([np.asarray(t, dtype=float) for t in tensors])
        self.best_validation_loss = parameters.get("best_validation_loss", self.best_validation_loss)
        return checkpoint

    def state_dict(self) -> dict:
        """Stato completo del training, sufficiente per il resume."""
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "model_class": type(self.model).__name__,
            "model_config": self._model_config(),
            "parameters": [np.asarray(p).copy() for p in self.model.get_parameters()],
            "optimizer_class": type(self.optimizer).__name__,
            "optimizer_state": (
                self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None
            ),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_validation_loss": self.best_validation_loss,
            "history": list(self.history),
            "tokenizer_state": (
                self.tokenizer.state_dict()
                if self.tokenizer is not None and hasattr(self.tokenizer, "state_dict")
                else None
            ),
        }

    def save_checkpoint(self, path) -> str:
        """
        Salva un checkpoint completo: modello, ottimizzatore, scheduler,
        contatori e tokenizer. Permette il resume esatto del training.
        """
        target = Path(path)
        if str(target.parent):
            target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as handle:
            pickle.dump(self.state_dict(), handle)
        return str(target)

    def load_checkpoint(self, path, load_tokenizer: bool = False) -> dict:
        """
        Ripristina un checkpoint completo salvato con `save_checkpoint()`.

        Il modello deve avere la stessa architettura del checkpoint.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")

        with open(source, "rb") as handle:
            state = pickle.load(handle)

        if not isinstance(state, dict) or state.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError("Formato di checkpoint non riconosciuto.")

        expected_class = state.get("model_class")
        if expected_class and expected_class != type(self.model).__name__:
            raise ValueError(
                f"Il checkpoint appartiene a {expected_class}, "
                f"il modello corrente è {type(self.model).__name__}."
            )

        model_config = state.get("model_config") or {}
        current_config = self._model_config()

        for key in ("vocab_size", "d_model", "num_heads", "d_ff", "num_layers", "max_len"):
            expected = model_config.get(key)
            current = current_config.get(key)
            if expected is not None and current is not None and expected != current:
                raise ValueError(
                    f"Checkpoint mismatch per {key}: atteso {expected}, "
                    f"corrente {current}."
                )

        current_params = self.model.get_parameters()
        saved_params = state.get("parameters", [])
        if len(saved_params) != len(current_params):
            raise ValueError(
                f"Numero di parametri incompatibile: checkpoint ha "
                f"{len(saved_params)}, modello ha {len(current_params)}."
            )
        for i, (saved, current) in enumerate(zip(saved_params, current_params)):
            if np.asarray(saved).shape != np.asarray(current).shape:
                raise ValueError(
                    f"Forma parametro {i} incompatibile: checkpoint "
                    f"{np.asarray(saved).shape}, modello "
                    f"{np.asarray(current).shape}."
                )

        tokenizer_state = state.get("tokenizer_state")
        if tokenizer_state and hasattr(self.tokenizer, "state_dict") and not load_tokenizer:
            current_tokenizer_state = self.tokenizer.state_dict()
            for key in ("format", "version", "mode", "special_tokens"):
                saved = tokenizer_state.get(key)
                current = current_tokenizer_state.get(key)
                if saved is not None and current is not None and saved != current:
                    raise ValueError(
                        f"Tokenizer mismatch per {key}: atteso {saved!r}, "
                        f"corrente {current!r}."
                    )
            saved_vocab = tokenizer_state.get("vocabulary") or {}
            current_vocab = current_tokenizer_state.get("vocabulary") or {}
            if saved_vocab and saved_vocab != current_vocab:
                raise ValueError(
                    "Il vocabolario del tokenizer nel checkpoint non coincide con quello corrente."
                )

        self.model.set_parameters([np.asarray(p) for p in state["parameters"]])

        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None and hasattr(self.optimizer, "load_state_dict"):
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = state.get("scheduler_state")
        if scheduler_state is not None:
            if self.scheduler is None:
                self.scheduler = LearningRateScheduler(
                    base_learning_rate=scheduler_state.get("base_learning_rate", 0.001)
                )
            self.scheduler.load_state_dict(scheduler_state)

        self.gradient_accumulation_steps = int(
            state.get("gradient_accumulation_steps", self.gradient_accumulation_steps)
        )
        self.max_grad_norm = state.get("max_grad_norm", self.max_grad_norm)
        self.seed = state.get("seed", self.seed)
        self.global_step = int(state.get("global_step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.best_validation_loss = state.get("best_validation_loss", float("inf"))
        self.history = list(state.get("history") or [])

        if load_tokenizer and tokenizer_state and hasattr(self.tokenizer, "load_state_dict"):
            self.tokenizer.load_state_dict(tokenizer_state)

        return state

    def resume(self, path) -> dict:
        """Alias esplicito di `load_checkpoint()`."""
        return self.load_checkpoint(path)

    def save_tokenizer(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.tokenizer, file)

    def load_tokenizer(self, path: str) -> None:
        with open(path, "rb") as file:
            self.tokenizer = pickle.load(file)

    def save_weights(self, path: str) -> None:
        """
        Salva i pesi del modello.

        Il formato include `model_class`, `config` e la lista dei parametri; per
        `NextTokenModel` vengono mantenute anche le chiavi storiche per
        retrocompatibilità.
        """
        payload = {
            "model_class": type(self.model).__name__,
            "config": self._model_config(),
            "parameters": [np.asarray(p).copy() for p in self.model.get_parameters()],
        }

        if isinstance(self.model, NextTokenModel):
            payload.update({
                "embeddings": self.model.embeddings,
                "W1": self.model.W1,
                "b1": self.model.b1,
                "W2": self.model.W2,
                "b2": self.model.b2,
                "vocab_size": self.model.vocab_size,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "dropout": self.model.dropout,
            })

        with open(path, "wb") as file:
            pickle.dump(payload, file)

    def load_weights(self, path: str) -> None:
        with open(path, "rb") as file:
            params = pickle.load(file)

        if not isinstance(params, dict):
            raise ValueError("File dei pesi non valido.")

        if isinstance(params.get("parameters"), list):
            self.model.set_parameters([np.asarray(p) for p in params["parameters"]])
        elif "embeddings" in params:
            self.model.embeddings = np.asarray(params["embeddings"])
            self.model.W1 = np.asarray(params["W1"])
            self.model.b1 = np.asarray(params["b1"])
            self.model.W2 = np.asarray(params["W2"])
            self.model.b2 = np.asarray(params["b2"])
        else:
            raise ValueError("Il file dei pesi non contiene parametri riconoscibili.")

        for attribute in ("vocab_size", "embedding_dim", "hidden_dim", "dropout"):
            if attribute in params and hasattr(self.model, attribute):
                setattr(self.model, attribute, params[attribute])
