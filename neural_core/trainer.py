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
import io
import json
import os
import pickle
import pickle as _pickle_module
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from neural_core.dataset import Dataset
from neural_core.tokenizer import Tokenizer
from neural_core.loss import CrossEntropyLoss
from neural_core.model import gelu, gelu_derivative
from neural_core.optimizer import Adam, LearningRateScheduler


_ALLOWED_PICKLE_MODULES = {
    "numpy",
    "numpy.core",
    "numpy.core.multiarray",
    "builtins",
    "collections",
    "collections.abc",
    "typing",
}

_ALLOWED_BUILTIN_CLASSES = {
    "list", "dict", "tuple", "set", "frozenset",
    "str", "int", "float", "bool", "bytes",
    "complex", "range", "slice",
    "OrderedDict", "defaultdict", "Counter",
}

_BLOCKED_BUILTIN_NAMES = {
    "eval", "exec", "compile", "__import__", "open",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "input",
    "os", "sys", "subprocess", "pickle", "marshal",
}


class SafeUnpickler(_pickle_module.Unpickler):
    """Unpickler that forbids arbitrary code execution.

    Only allows deserialization of:
    - numpy arrays and scalars (via numpy and numpy._core internal classes)
    - Basic builtins (list, dict, tuple, set, etc.)
    - collections (OrderedDict, defaultdict, Counter)

    Raises ``PermissionError`` for any other class, effectively neutralizing
    ``__reduce__``-based RCE payloads (e.g. ``os.system``, ``eval``,
    ``__import__``).
    """

    def find_class(self, module, name):
        # Allow numpy and its internal submodules (e.g. numpy._core.numeric) —
        # these are needed to reconstruct ndarray/scalar objects.
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        if module in _ALLOWED_PICKLE_MODULES:
            if module == "builtins" and name in _BLOCKED_BUILTIN_NAMES:
                raise PermissionError(
                    f"Deserializzazione pickle bloccata: builtins.{name} è pericoloso"
                )
            if module == "collections" and name in _BLOCKED_BUILTIN_NAMES:
                raise PermissionError(
                    f"Deserializzazione pickle bloccata: collections.{name}"
                )
            if module == "builtins" and name in _ALLOWED_BUILTIN_CLASSES:
                return super().find_class(module, name)
            if module == "collections" and name in _ALLOWED_BUILTIN_CLASSES:
                return super().find_class(module, name)
            if module == "typing":
                return super().find_class(module, name)
            # Other allowed modules (e.g. collections.abc) — permit
            return super().find_class(module, name)
        raise PermissionError(
            f"Deserializzazione pickle bloccata per modulo sconosciuto: {module}.{name}"
        )


def safe_pickle_load(file_obj, max_bytes: int = 256 * 1024 * 1024) -> Any:
    """Deserialize a pickle stream using SafeUnpickler (blocks RCE payloads)."""
    return SafeUnpickler(file_obj).load()


def safe_pickle_loads(data: bytes, max_bytes: int = 256 * 1024 * 1024) -> Any:
    """Deserialize a pickle byte string using SafeUnpickler (blocks RCE payloads)."""
    return SafeUnpickler(io.BytesIO(data)).load()


def _is_trusted_internal_path(path: str, allowed_roots: Optional[List[str]] = None) -> bool:
    """Constrain weight/checkpoint/tokenizer loading to trusted internal dirs."""
    if allowed_roots is None:
        repo_root = str(Path(__file__).resolve().parents[1])
        allowed_roots = [
            str(Path(repo_root) / "neural_core" / "data"),
            str(Path(repo_root) / "neural_core"),
            str(Path(repo_root) / "CORE"),
        ]
    try:
        real = str(Path(path).resolve())
        for root in allowed_roots:
            if real.startswith(str(Path(root).resolve())):
                return True
    except Exception:
        pass
    return False


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


def _save_checkpoint_safe(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(state), f)


def _load_checkpoint_safe(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _from_serializable(json.load(f))


def _safe_weights_path(path: str) -> tuple:
    base = os.path.splitext(path)[0]
    return base + ".json", base + ".npz"


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "dtype": str(obj.dtype), "shape": list(obj.shape), "data": obj.tolist()}
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _from_serializable(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.array(obj["data"], dtype=obj.get("dtype", "float64")).reshape(obj["shape"])
    elif isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    return obj


def _save_weights_safe(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f)


def _load_weights_safe(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _from_serializable(json.load(f))


def load_model_from_weights(path, vocab_size: Optional[int] = None):
    """
    Ricostruisce un modello dai pesi salvati da `Trainer.save_weights()`.

    Supporta sia il formato corrente (con `model_class` e `parameters`) sia il
    formato storico di `NextTokenModel` (`embeddings`, `W1`, `b1`, `W2`, `b2`).

    Returns:
        Il modello con i pesi caricati.
    """
    try:
        payload = _load_weights_safe(path)
    except Exception:
        with open(path, "rb") as handle:
            payload = safe_pickle_load(handle)

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
                 gradient_accumulation_steps: int = 1, max_grad_norm=None, seed=None,
                 ewc_lambda: float = 0.5):
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
        self.ewc_lambda = float(ewc_lambda)

        # Elastic Weight Consolidation (EWC) state
        self.fisher_matrix = None   # dict: param_name -> np.ndarray (Fisher info)
        self.optimal_weights = None # dict: param_name -> np.ndarray (W_A snapshot)

        self.global_step = 0
        self.epoch = 0
        self.best_validation_loss = float("inf")
        self.history: List[dict] = []

        # Episodic replay buffer: list of (inputs, targets) tuples
        self.replay_buffer = None

    # ------------------------------------------------------------------
    # Elastic Weight Consolidation (EWC) & Continual Learning
    # ------------------------------------------------------------------

    def _all_param_names(self):
        """Mappa i nomi di TUTTI i parametri del modello alle chiavi di EWC."""
        if hasattr(self.model, "named_parameters"):
            try:
                return list(self.model.named_parameters().keys())
            except Exception:
                pass
        params = self.model.get_parameters()
        return [f"param_{i}" for i in range(len(params))]

    def _fisher_param_names(self):
        """
        Restituisce la lista dei nomi dei parametri per la Fisher matrix.

        Con LoRA attivo (modello con `trainable_param_names`), restituisce SOLO
        i nomi dei parametri addestrabili (le matrici LoRA A/B). I parametri
        congelati (2.7M) non entrano mai nella Fisher matrix, riducendone il
        costo di >90% rispetto alla versione full-parameter.
        """
        if hasattr(self.model, "trainable_param_names"):
            try:
                names = self.model.trainable_param_names()
                if names:
                    return names
            except Exception:
                pass
        if hasattr(self.model, "named_parameters"):
            try:
                return list(self.model.named_parameters().keys())
            except Exception:
                pass
        params = self.model.get_parameters()
        return [f"param_{i}" for i in range(len(params))]

    def _trainable_indices(self):
        """Indici in `get_parameters()` dei parametri addestrabili."""
        if hasattr(self.model, "trainable_param_indices"):
            try:
                return list(self.model.trainable_param_indices())
            except Exception:
                pass
        return list(range(len(self.model.get_parameters())))

    def compute_fisher_matrix(self, dataset, num_samples=100):
        """
        Calcola la Micro-Fisher Matrix sul SOLE spazio dei parametri
        addestrabili (matrici LoRA A/B).

        I gradienti del log-likelihood vengono accumulati come quadrati e
        normalizzati su `num_samples`. I parametri congelati (embedding,
        attention, LayerNorm, output) non vengono calcolati: la loro importanza
        è fissata dai pesi pre-addestrati e non deve essere penalizzata.

        Salva i pesi attuali in `self.optimal_weights` e la Fisher in
        `self.fisher_matrix` (solo per i parametri addestrabili).
        """
        if dataset is None:
            raise ValueError("dataset è richiesto per calcolare la Fisher matrix.")

        inputs, targets = self._dataset_components(dataset)
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        n = len(inputs)
        if n == 0:
            raise ValueError("Il dataset per la Fisher matrix non può essere vuoto.")

        sample_count = min(int(num_samples), n)
        indices = np.arange(n)
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
        indices = indices[:sample_count]

        trainable_idx = self._trainable_indices()
        param_names = self._fisher_param_names()

        # Accumulatore: somma dei quadrati dei gradienti per ogni parametro addestrabile
        fisher_accum = None

        for idx in indices:
            if inputs.ndim >= 2:
                batch_inputs = inputs[idx:idx + 1]
                batch_targets = targets[idx:idx + 1]
            else:
                batch_inputs = np.array([inputs[idx]])
                batch_targets = np.array([targets[idx]])

            logits = self.model.forward(batch_inputs, training=False)
            loss_value = float(self.loss.forward(logits, batch_targets))
            probs = getattr(self.loss, "get_probabilities", lambda: None)()
            if probs is not None:
                gradients = self.model.backward(batch_targets, probs=probs)
            else:
                gradients = self.model.backward(batch_targets)

            gradients = [np.asarray(g, dtype=float) for g in gradients]
            # Riduci ai soli parametri addestrabili
            trainable_grads = [gradients[i] for i in trainable_idx]
            if fisher_accum is None:
                fisher_accum = [np.square(g) for g in trainable_grads]
            else:
                for i in range(len(fisher_accum)):
                    fisher_accum[i] += np.square(trainable_grads[i])

        if fisher_accum is None:
            raise ValueError("Impossibile accumulare gradienti per la Fisher matrix.")

        # Salva i pesi ottimali correnti (solo i parametri addestrabili)
        all_params = self.model.get_parameters()
        self.optimal_weights = {
            name: np.asarray(all_params[i]).copy()
            for i, name in zip(trainable_idx, param_names)
        }

        # Normalizza e memorizza la Micro-Fisher matrix
        self.fisher_matrix = {
            name: fisher_accum[i] / float(sample_count)
            for i, name in enumerate(param_names)
        }

        # Ricorda su quale task e' stata ancorata la Fisher, per poter
        # ri-ancorarla automaticamente quando viene registrato un nuovo task.
        self._ewc_anchor_task = getattr(self.model, "active_task", None)

        return self.fisher_matrix

    def _ensure_ewc_anchor(self):
        """
        Ri-ancora la Micro-Fisher al task attivo se il task e' cambiato.

        Con il design a adapter per-task, quando si registra un nuovo task il
        suo adapter parte dai pesi dell'adapter precedente (warm start). La
        Fisher calcolata sul task precedente vive nello stesso spazio di
        parametri (stesse matrici A/B, stesso rango) e puo' quindi vincolare
        direttamente il nuovo adapter: e' esattamente l'ipotesi
        theta_{k+1} = theta_k su cui si basa l'EWC full-parameter, proiettata
        nello spazio LoRA.

        Senza questa ri-ancoratura i nomi della Fisher (riferiti al task
        precedente) non corrisponderebbero ai nomi dei parametri addestrabili
        del nuovo task e la penalita' sarebbe silenziosamente nulla.
        """
        if self.fisher_matrix is None or self.optimal_weights is None:
            return
        current_task = getattr(self.model, "active_task", None)
        anchor_task = getattr(self, "_ewc_anchor_task", None)
        if current_task is None or anchor_task is None or current_task == anchor_task:
            return

        old_names = list(self.fisher_matrix.keys())
        new_names = self._fisher_param_names()
        if len(old_names) != len(new_names):
            # Spazi di parametri diversi: la Fisher precedente non e' trasferibile.
            self.fisher_matrix = None
            self.optimal_weights = None
            self._ewc_anchor_task = None
            return

        all_params = self.model.get_parameters()
        trainable_idx = self._trainable_indices()
        self.fisher_matrix = dict(zip(new_names, self.fisher_matrix.values()))
        # L'ancoraggio e' il punto di partenza del nuovo adapter, che per il
        # warm start coincide con l'ottimo del task precedente.
        self.optimal_weights = {
            name: np.asarray(all_params[i]).copy()
            for name, i in zip(new_names, trainable_idx)
        }
        self._ewc_anchor_task = current_task

    def _dataset_components(self, dataset):
        """Estrae inputs e targets da un Dataset o da una tupla (inputs, targets)."""
        if hasattr(dataset, "get_inputs") and hasattr(dataset, "get_targets"):
            return dataset.get_inputs(), dataset.get_targets()
        if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            return np.asarray(dataset[0]), np.asarray(dataset[1])
        raise ValueError("Dataset non riconosciuto per la Fisher matrix.")

    def _compute_ewc_loss(self):
        """
        Calcola la penalità Micro-EWC corrente (0 se non c'è Fisher matrix).

        La penalità è applicata ESCLUSIVAMENTE sui parametri addestrabili
        (matrici LoRA A/B). I parametri congelati non sono penalizzati.
        """
        if self.fisher_matrix is None or self.optimal_weights is None:
            return 0.0
        ewc_loss = 0.0
        param_names = self._fisher_param_names()
        all_params = self.model.get_parameters()
        trainable_idx = self._trainable_indices()
        for name, idx in zip(param_names, trainable_idx):
            if name in self.fisher_matrix and name in self.optimal_weights:
                fisher = self.fisher_matrix[name]
                opt_weight = self.optimal_weights[name]
                current = np.asarray(all_params[idx])
                ewc_loss += float((fisher * (current - opt_weight) ** 2).sum())
        return ewc_loss

    def set_replay_buffer(self, replay_buffer):
        """
        Imposta il buffer di replay episodico per il continual learning.
        `replay_buffer` deve essere una lista di tuple (inputs, targets).
        """
        if replay_buffer is None:
            self.replay_buffer = None
            return
        if not isinstance(replay_buffer, (list, tuple)):
            raise ValueError("replay_buffer deve essere una lista di tuple (inputs, targets).")
        self.replay_buffer = list(replay_buffer)

    def _sample_replay(self, current_inputs, current_targets, replay_ratio=0.2,
                       replay_buffer=None):
        """
        Miscela dati attuali con campioni dal replay buffer.

        Restituisce (inputs, targets) miscelati con un rapporto
        `replay_ratio` di dati provenienti dal buffer.

        Args:
            replay_buffer: buffer da usare per questo batch. Se `None`, si
                ripiega su `self.replay_buffer` impostato da `set_replay_buffer`.
                Senza questo fallback il parametro `replay_buffer` passato a
                `train()` / `train_step()` veniva ignorato e il replay
                episodico non veniva mai applicato.
        """
        buffer = replay_buffer if replay_buffer is not None else self.replay_buffer
        if not buffer:
            return current_inputs, current_targets

        n_current = len(current_inputs)
        n_replay = max(1, int(round(n_current * replay_ratio / (1.0 - replay_ratio))))
        n_replay = min(n_replay, len(buffer))

        if self.seed is not None:
            rng = np.random.RandomState(self.seed + self.global_step)
        else:
            rng = np.random

        indices = rng.choice(len(buffer), size=n_replay, replace=True)
        replay_inputs = []
        replay_targets = []
        current_inputs_arr = np.asarray(current_inputs)
        current_ndim = current_inputs_arr.ndim
        for idx in indices:
            r_inputs, r_targets = buffer[int(idx)]
            r_inputs = np.asarray(r_inputs)
            r_targets = np.asarray(r_targets)
            # Un campione del buffer puo' essere una singola sequenza 1-D
            # (context_size,) mentre il batch corrente e' 2-D (batch, seq).
            # Senza questa normalizzazione il controllo di ndim scartava
            # TUTTI i campioni e il replay episodico non avveniva mai.
            if r_inputs.ndim == current_ndim - 1:
                r_inputs = np.expand_dims(r_inputs, axis=0)
                r_targets = np.expand_dims(r_targets, axis=0)
            if r_inputs.ndim != current_ndim or r_targets.ndim != r_inputs.ndim:
                continue
            if r_inputs.shape[1:] != current_inputs_arr.shape[1:]:
                continue
            replay_inputs.append(r_inputs)
            replay_targets.append(r_targets)

        if not replay_inputs:
            return current_inputs, current_targets

        try:
            replay_inputs = np.concatenate(replay_inputs, axis=0)
            replay_targets = np.concatenate(replay_targets, axis=0)
        except ValueError:
            return current_inputs, current_targets

        mixed_inputs = np.concatenate([current_inputs, replay_inputs], axis=0)
        mixed_targets = np.concatenate([current_targets, replay_targets], axis=0)

        perm = np.arange(len(mixed_inputs))
        if self.seed is not None:
            np.random.RandomState(self.seed + self.global_step).shuffle(perm)
        else:
            np.random.shuffle(perm)
        return mixed_inputs[perm], mixed_targets[perm]

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
        """
        Applica un aggiornamento ai parametri del modello tramite l'ottimizzatore.

        Con LoRA attivo, l'ottimizzatore riceve esclusivamente il sottoinsieme
        dei parametri addestrabili (le matrici A/B dell'adapter del task
        attivo) e i relativi gradienti. I parametri congelati non vengono
        neppure passati all'ottimizzatore, quindi non possono essere spostati
        dai momenti `m`/`v` di Adam.

        Questo e' necessario perche' la registrazione di un nuovo task cambia
        la lunghezza della lista completa dei parametri: i momenti dell'ottimizzatore
        riuserebbero altrimenti a non combaciare. L'ordinamento del sottoinsieme
        addestrabile (A, B per ogni layer, nell'ordine dei layer) e' invece
        stabile tra un task e l'altro, quindi lo stato dell'ottimizzatore resta
        valido e riutilizzabile.

        Modelli senza LoRA seguono esattamente il comportamento legacy
        full-parameter.
        """
        gradients, _ = self._clip_gradients(gradients)
        self._current_learning_rate()
        parameters = self.model.get_parameters()
        trainable_idx = self._trainable_indices()

        if len(trainable_idx) == len(parameters):
            # Nessun parametro congelato: percorso legacy invariato.
            updated = self.optimizer.update(parameters, gradients)
        else:
            trainable_params = [parameters[i] for i in trainable_idx]
            trainable_grads = [gradients[i] for i in trainable_idx]
            updated_trainable = self.optimizer.update(trainable_params, trainable_grads)
            updated = list(parameters)
            for pos, i in enumerate(trainable_idx):
                updated[i] = updated_trainable[pos]

        self.model.set_parameters(updated)
        self.global_step += 1

    # ------------------------------------------------------------------
    # Step di training
    # ------------------------------------------------------------------

    def train_step(self, batch_inputs, batch_targets, replay_buffer=None) -> float:
        """
        Un singolo step di training (forward + backward + update immediato).

        Args:
            batch_inputs, batch_targets: batch corrente.
            replay_buffer: opzionale buffer di replay episodico; se fornito,
                i dati vengono miscelati con un 20% di campioni dal buffer.

        Returns:
            La loss del batch (comprensiva di eventuale penalità EWC).
        """
        self._ensure_ewc_anchor()
        if replay_buffer is not None or self.replay_buffer:
            batch_inputs, batch_targets = self._sample_replay(
                batch_inputs, batch_targets, replay_ratio=0.2,
                replay_buffer=replay_buffer,
            )

        input_ids, target_ids = self._prepare_batch(batch_inputs, batch_targets)
        logits = self.model.forward(input_ids, training=True)
        loss_value = float(self.loss.forward(logits, target_ids))
        ewc_loss = self._compute_ewc_loss()
        total_loss = loss_value + (self.ewc_lambda / 2.0) * ewc_loss

        probs = getattr(self.loss, "get_probabilities", lambda: None)()
        if probs is not None:
            gradients = self.model.backward(target_ids, probs=probs)
        else:
            gradients = self.model.backward(target_ids)

        # Aggiungi il gradiente della penalità EWC ai gradienti del modello
        if ewc_loss > 0.0:
            ewc_gradients = self._compute_ewc_gradients()
            gradients = [g + eg for g, eg in zip(gradients, ewc_gradients)]

        self._apply_update([np.asarray(g, dtype=float) for g in gradients])
        return total_loss

    def _compute_ewc_gradients(self):
        """
        Calcola il gradiente della penalità Micro-EWC rispetto ai parametri
        del modello.

        d/dW [ (lambda/2) * sum F * (W - W_A)^2 ] = lambda * F * (W - W_A)

        applicato SOLO ai parametri addestrabili (matrici LoRA A/B).
        I parametri congelati ricevono gradiente zero.
        """
        if self.fisher_matrix is None or self.optimal_weights is None:
            return []
        all_params = self.model.get_parameters()
        trainable_idx = self._trainable_indices()
        param_names = self._fisher_param_names()

        ewc_gradients = [np.zeros_like(np.asarray(p)) for p in all_params]
        for name, idx in zip(param_names, trainable_idx):
            if name in self.fisher_matrix and name in self.optimal_weights:
                fisher = self.fisher_matrix[name]
                opt_weight = self.optimal_weights[name]
                diff = np.asarray(all_params[idx]) - opt_weight
                ewc_gradients[idx] = self.ewc_lambda * fisher * diff
        return ewc_gradients

    def _run_epoch(self, inputs, targets, batch_size, training=True, shuffle=True, replay_buffer=None):
        if training:
            self._ensure_ewc_anchor()
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

            # Miscela episodica replay (80% dati nuovi, 20% dal buffer)
            if training and (replay_buffer is not None or self.replay_buffer):
                batch_inputs, batch_targets = self._sample_replay(
                    batch_inputs, batch_targets, replay_ratio=0.2,
                    replay_buffer=replay_buffer,
                )

            logits = self.model.forward(batch_inputs, training=training)
            loss_value = float(self.loss.forward(logits, batch_targets))
            ewc_loss = self._compute_ewc_loss() if training else 0.0
            total_loss = loss_value + (self.ewc_lambda / 2.0) * ewc_loss
            epoch_losses.append(total_loss)

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

            # Aggiungi il gradiente della penalità EWC
            if self.fisher_matrix is not None and self.optimal_weights is not None:
                ewc_gradients = self._compute_ewc_gradients()
                if len(ewc_gradients) == len(gradients):
                    gradients = [g + eg for g, eg in zip(gradients, ewc_gradients)]

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

    def train(self, inputs, targets, epochs: int = 1, batch_size: int = 64,
              replay_buffer=None) -> List[float]:
        if len(inputs) != len(targets):
            raise ValueError("Inputs e targets devono avere la stessa lunghezza.")

        if epochs < 1:
            raise ValueError("epochs deve essere almeno 1.")

        history = []
        for _ in range(epochs):
            average_loss = self._run_epoch(
                inputs, targets, batch_size, training=True,
                replay_buffer=replay_buffer,
            )
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
        replay_buffer=None,
        ewc_lambda: Optional[float] = None,
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

        if ewc_lambda is not None:
            self.ewc_lambda = float(ewc_lambda)

        if resume_path is not None:
            self.load_checkpoint(resume_path)

        history = []
        best_params = None
        patience_counter = 0

        for _ in range(epochs):
            train_loss = self._run_epoch(
                training_inputs, training_targets, batch_size, training=True,
                replay_buffer=replay_buffer,
            )
            val_loss = self.evaluate(validation_inputs, validation_targets, batch_size=batch_size)
            self.epoch += 1

            entry = {
                "epoch": self.epoch,
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": getattr(self.optimizer, "learning_rate", None),
                "ewc_loss": self._compute_ewc_loss(),
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
        ewc_state = None
        if self.fisher_matrix is not None or self.optimal_weights is not None:
            ewc_state = {
                "ewc_lambda": self.ewc_lambda,
                "fisher_matrix": (
                    {k: np.asarray(v).tolist() for k, v in self.fisher_matrix.items()}
                    if self.fisher_matrix is not None else None
                ),
                "optimal_weights": (
                    {k: np.asarray(v).tolist() for k, v in self.optimal_weights.items()}
                    if self.optimal_weights is not None else None
                ),
                "param_names": self._fisher_param_names(),
            }
        replay_payload = None
        if self.replay_buffer is not None:
            replay_payload = [
                [np.asarray(inp).tolist(), np.asarray(tgt).tolist()]
                for inp, tgt in self.replay_buffer
            ]
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
            "ewc_state": ewc_state,
            "replay_buffer": replay_payload,
        }

    def save_checkpoint(self, path) -> str:
        """
        Salva un checkpoint completo: modello, ottimizzatore, scheduler,
        contatori e tokenizer. Permette il resume esatto del training.
        """
        target = Path(path)
        if str(target.parent):
            target.parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint_safe(str(target), self.state_dict())
        return str(target)

    def load_checkpoint(self, path, load_tokenizer: bool = False) -> dict:
        """
        Ripristina un checkpoint completo salvato con `save_checkpoint()`.

        Il modello deve avere la stessa architettura del checkpoint.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")

        try:
            state = _load_checkpoint_safe(str(source))
        except Exception:
            with open(source, "rb") as handle:
                state = safe_pickle_load(handle)

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

        # Ripristina lo stato EWC (Fisher matrix + optimal weights) se presente
        ewc_state = state.get("ewc_state")
        if isinstance(ewc_state, dict):
            if ewc_state.get("ewc_lambda") is not None:
                self.ewc_lambda = float(ewc_state["ewc_lambda"])
            fisher_data = ewc_state.get("fisher_matrix")
            opt_data = ewc_state.get("optimal_weights")
            param_names = ewc_state.get("param_names") or self._fisher_param_names()
            if isinstance(fisher_data, dict):
                self.fisher_matrix = {
                    name: np.asarray(fisher_data[name], dtype=float)
                    for name in param_names
                    if name in fisher_data
                }
            elif isinstance(fisher_data, list):
                self.fisher_matrix = {
                    name: np.asarray(fisher_data[i], dtype=float)
                    for i, name in enumerate(param_names)
                    if i < len(fisher_data)
                }
            if isinstance(opt_data, dict):
                self.optimal_weights = {
                    name: np.asarray(opt_data[name], dtype=float)
                    for name in param_names
                    if name in opt_data
                }
            elif isinstance(opt_data, list):
                self.optimal_weights = {
                    name: np.asarray(opt_data[i], dtype=float)
                    for i, name in enumerate(param_names)
                    if i < len(opt_data)
                }

        # Ripristina il replay buffer episodico se presente
        replay_data = state.get("replay_buffer")
        if replay_data is not None:
            self.replay_buffer = [
                (np.asarray(inp), np.asarray(tgt))
                for inp, tgt in replay_data
            ]

        if load_tokenizer and tokenizer_state and hasattr(self.tokenizer, "load_state_dict"):
            self.tokenizer.load_state_dict(tokenizer_state)

        return state

    def resume(self, path) -> dict:
        """Alias esplicito di `load_checkpoint()`."""
        return self.load_checkpoint(path)

    def save_tokenizer(self, path: str) -> None:
        if hasattr(self.tokenizer, "save"):
            self.tokenizer.save(path)
        else:
            with open(path, "wb") as file:
                pickle.dump(self.tokenizer, file)

    def load_tokenizer(self, path: str) -> None:
        if path.endswith(".json") or Path(path).suffix == ".json":
            self.tokenizer = Tokenizer.load(path)
        else:
            with open(path, "rb") as file:
                self.tokenizer = safe_pickle_load(file)

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

        _save_weights_safe(path, payload)

    def load_weights(self, path: str) -> None:
        source = Path(path)
        if source.exists():
            try:
                params = _load_weights_safe(str(source))
            except Exception:
                with open(path, "rb") as file:
                    params = safe_pickle_load(file)
        else:
            raise FileNotFoundError(f"File dei pesi non trovato: {path}")

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
