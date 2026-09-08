import numpy as np


class MSELoss:
    def forward(self, prediction, target):
        error = prediction - target
        return error ** 2

    def backward(self, prediction, target):
        return 2 * (prediction - target)


class CrossEntropyLoss:
    """
    Cross entropy con softmax integrata.

    Supporta:
        - logits 1D `(V,)`         con target scalare o `(1,)`;
        - logits 2D `(N, V)`       con target `(N,)`;
        - logits 3D `(B, T, V)`    con target `(B, T)`  -> modello autoregressivo.

    `ignore_index` permette di escludere posizioni di padding dal calcolo.

    Le probabilità calcolate in `forward()` vengono cachate e riutilizzate in
    `backward()` per evitare il calcolo duplicato della softmax.
    """

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self._cached_probs = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(logits, targets):
        logits = np.asarray(logits, dtype=float)
        targets = np.asarray(targets, dtype=int)

        original_shape = logits.shape

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        elif logits.ndim > 2:
            logits = logits.reshape(-1, original_shape[-1])

        targets = targets.reshape(-1)

        if targets.shape[0] != logits.shape[0]:
            raise ValueError(
                "Il numero di target deve coincidere con il numero di predizioni."
            )

        return logits, targets, original_shape

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, logits, targets):
        logits, targets, _ = self._flatten(logits, targets)
        probs = self._softmax(logits)
        self._cached_probs = probs

        valid = targets != self.ignore_index
        if not np.any(valid):
            return 0.0

        safe_targets = np.where(valid, targets, 0)
        picked = probs[np.arange(logits.shape[0]), safe_targets]
        nll = -np.log(picked + 1e-12)
        return float(np.mean(nll[valid]))

    def backward(self, logits, targets, probs=None):
        flat_logits, flat_targets, original_shape = self._flatten(logits, targets)

        if probs is None:
            probs = self._softmax(flat_logits)

        valid = flat_targets != self.ignore_index
        valid_count = int(np.sum(valid))

        if valid_count == 0:
            return np.zeros(original_shape, dtype=float)

        safe_targets = np.where(valid, flat_targets, 0)
        probs[np.arange(flat_logits.shape[0]), safe_targets] -= 1
        probs[~valid] = 0.0
        grad = probs / valid_count

        if len(original_shape) > 2:
            return grad.reshape(original_shape)

        return grad

    def get_probabilities(self):
        """Restituisce le probabilità cachate dall'ultimo forward."""
        return self._cached_probs
