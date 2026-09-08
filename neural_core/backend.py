"""
Backend abstraction for JARVIS Neural Core (Phase 2A).

Provides a uniform interface over two isolated backends:

    - ``NumpyBackend``  -> CPU reference (``neural_core.network.MicroTransformer``,
                           float64, authoritative baseline).
    - ``TorchBackend``  -> GPU backend (``neural_core.torch_model.TorchMicroTransformer``,
                           float32, PyTorch/MPS).

The CPU reference is never modified. The factory ``get_backend`` selects the
backend by name. Both backends expose the same forward / backward / gradient
API so the parity tests and benchmarks can treat them interchangeably.
"""

import abc
import numpy as np

from neural_core.network import MicroTransformer
from neural_core.torch_model import TorchMicroTransformer


class NeuralBackend(abc.ABC):
    """Common interface implemented by both backends."""

    name = "abstract"
    device = "abstract"
    dtype = "abstract"

    @abc.abstractmethod
    def set_parameters(self, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, token_ids, training=False):
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, token_ids, targets, training=False):
        """Return ``(loss_value, gradients_list)``."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameter_count(self):
        raise NotImplementedError

    @abc.abstractmethod
    def config(self):
        raise NotImplementedError


class NumpyBackend(NeuralBackend):
    """CPU reference backend (float64, NumPy)."""

    name = "numpy_cpu"
    device = "cpu"
    dtype = "float64"

    def __init__(self, config):
        self.config = dict(config)
        self.model = MicroTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            max_len=config["max_len"],
            dropout=config.get("dropout", 0.0),
            seed=config.get("seed", None),
            dtype=np.float64,
        )
        from neural_core.loss import CrossEntropyLoss
        self._loss = CrossEntropyLoss()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self):
        return self.model.get_parameters()

    def forward(self, token_ids, training=False):
        return np.asarray(self.model.forward(token_ids, training=training))

    def backward(self, token_ids, targets, training=False):
        logits = self.model.forward(token_ids, training=training)
        loss = self._loss.forward(logits, targets)
        grads = self.model.backward(targets)
        return float(loss), [np.asarray(g, dtype=np.float64) for g in grads]

    def parameter_count(self):
        return self.model.parameter_count()


class TorchBackend(NeuralBackend):
    """GPU backend (float32, PyTorch/MPS)."""

    name = "torch_gpu"
    device = "mps"

    def __init__(self, config, device=None):
        if device is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.dtype = "float32"
        self.config = dict(config)
        import torch
        self._torch = torch
        self.model = TorchMicroTransformer(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            max_len=config["max_len"],
            dropout=config.get("dropout", 0.0),
            seed=config.get("seed", None),
            dtype=torch.float32,
            device=device,
        )

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self):
        return self.model.get_parameters()

    def forward(self, token_ids, training=False):
        with self._torch.no_grad():
            logits = self.model.forward(token_ids, training=training)
        return logits.detach().cpu().numpy()

    def backward(self, token_ids, targets, training=False):
        self._torch.manual_seed(0)
        logits = self.model.forward(token_ids, training=training)
        flat_logits = logits.reshape(-1, self.model.vocab_size)
        flat_targets = self._torch.as_tensor(
            np.asarray(targets).reshape(-1), dtype=self._torch.long, device=self.model.device)
        loss = self._torch.nn.functional.cross_entropy(
            flat_logits, flat_targets, ignore_index=-100, reduction="mean")
        self.model.zero_grad()
        loss.backward()
        grads = self.model.get_gradients()
        return float(loss), [np.asarray(g, dtype=np.float32) for g in grads]

    def parameter_count(self):
        return self.model.parameter_count()


def get_backend(name, config, device=None):
    """Factory: ``"cpu"``/``"numpy"`` -> NumpyBackend, ``"gpu"``/``"torch"`` -> TorchBackend."""
    if name in ("cpu", "numpy", "numpy_cpu"):
        return NumpyBackend(config)
    if name in ("gpu", "torch", "torch_gpu", "mps"):
        return TorchBackend(config, device=device)
    raise ValueError(f"Unknown backend: {name}")
