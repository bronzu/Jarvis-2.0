import math

import numpy as np

from neural_core.loss import MSELoss
from neural_core.optimizer import SGD
from neural_core.checkpoint import Checkpoint


def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    x = np.asarray(x, dtype=float)
    tanh_arg = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * (1.0 + np.tanh(tanh_arg)) + 0.5 * x * (1.0 - np.tanh(tanh_arg) ** 2) * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)


def softmax(values, axis=-1):
    """
    Softmax numericamente stabile.

    Args:
        values: array di input.
        axis: asse di normalizzazione. `None` normalizza su tutto l'array.
    """
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=axis is not None)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=axis is not None)


def softmax_backward(probs, grad_output, axis=-1):
    """
    Gradiente della softmax dato il gradiente in uscita.

    d/dz = p * (g - sum(p * g))
    """
    probs = np.asarray(probs, dtype=float)
    grad_output = np.asarray(grad_output, dtype=float)
    weighted = np.sum(probs * grad_output, axis=axis, keepdims=True)
    return probs * (grad_output - weighted)


def causal_mask(seq_len, dtype=float):
    """
    Maschera causale additiva `(1, 1, seq_len, seq_len)`.

    Le posizioni future valgono `-inf`, quelle consentite `0`.
    Costruita in modo vettorizzato (nessun doppio loop Python).
    """
    if seq_len <= 0:
        raise ValueError("seq_len deve essere maggiore di 0.")

    allowed = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    mask = np.where(allowed, 0.0, -np.inf).astype(dtype)
    return mask.reshape(1, 1, seq_len, seq_len)


class SelfAttention:
    """
    Self attention single-head con supporto al causal masking.

    Mantiene la firma storica `forward(query, key, value)` e aggiunge:
        - `causal=True` per il masking autoregressivo;
        - `backward(grad_output)` per l'addestramento.
    """

    def __init__(self, scale=None, causal=False):
        self.scale = scale
        self.causal = causal
        self._cached = {}

    def forward(self, query, key, value, causal=None, mask=None):
        q = np.asarray(query, dtype=float)
        k = np.asarray(key, dtype=float)
        v = np.asarray(value, dtype=float)

        if q.ndim == 1:
            q = q.reshape(1, -1)
        if k.ndim == 1:
            k = k.reshape(1, -1)
        if v.ndim == 1:
            v = v.reshape(1, -1)

        if q.shape[-1] != k.shape[-1]:
            raise ValueError("Le dimensioni di query e key devono coincidere.")

        if k.shape[0] != v.shape[0]:
            raise ValueError("Key e value devono avere lo stesso numero di posizioni.")

        use_causal = self.causal if causal is None else causal

        scale = self.scale if self.scale is not None else math.sqrt(k.shape[-1])
        scores = (q @ k.T) / scale

        if use_causal:
            if q.shape[0] != k.shape[0]:
                raise ValueError(
                    "Il causal masking richiede query e key della stessa lunghezza."
                )
            scores = scores + causal_mask(q.shape[0])[0, 0]

        if mask is not None:
            scores = scores + np.asarray(mask, dtype=float)

        attention_weights = softmax(scores, axis=-1)
        output = attention_weights @ v

        self._cached = {
            "q": q,
            "k": k,
            "v": v,
            "scale": scale,
            "weights": attention_weights,
        }
        return output

    def backward(self, grad_output):
        """
        Gradienti rispetto a query, key e value.

        Returns:
            (grad_query, grad_key, grad_value)
        """
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=float)
        q = self._cached["q"]
        k = self._cached["k"]
        v = self._cached["v"]
        scale = self._cached["scale"]
        weights = self._cached["weights"]

        grad_weights = grad_output @ v.T
        grad_v = weights.T @ grad_output

        grad_scores = softmax_backward(weights, grad_weights, axis=-1) / scale
        grad_q = grad_scores @ k
        grad_k = grad_scores.T @ q

        return grad_q, grad_k, grad_v

    def attention_weights(self):
        """Pesi di attenzione dell'ultima forward."""
        if not self._cached:
            raise RuntimeError("Nessuna forward() eseguita.")
        return self._cached["weights"]


class Decoder:
    @staticmethod
    def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
        if temperature <= 0:
            raise ValueError("temperature deve essere positiva.")

        logits = np.asarray(logits, dtype=float)
        logits = logits / temperature

        if top_k is not None:
            top_k = max(1, int(top_k))
            if top_k < logits.size:
                candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
                logits_original = logits.copy()
                logits = np.full_like(logits, -np.inf)
                logits[candidate_indices] = logits_original[candidate_indices]
                logits = np.where(np.isfinite(logits), logits, -1e9)

        if top_p is not None:
            top_p = float(top_p)
            if not 0 < top_p <= 1:
                raise ValueError("top_p deve essere tra 0 e 1.")
            sorted_idx = np.argsort(logits)[::-1]
            cumulative = np.cumsum(np.exp(logits[sorted_idx]))
            cutoff = cumulative <= top_p * cumulative[-1]
            if cutoff.any():
                keep = sorted_idx[: int(np.argmax(np.where(cutoff, 1, 0)) + 1)]
                filtered = np.full_like(logits, -np.inf)
                filtered[keep] = logits[keep]
                logits = filtered

        distribution = np.exp(logits - np.max(logits))
        distribution = distribution / np.sum(distribution)
        rng = np.random if rng is None else rng
        return int(rng.choice(len(distribution), p=distribution))


class TextGenerator:
    def __init__(self, decoder=None):
        self.decoder = decoder or Decoder()

    def generate(self, logits, temperature=1.0, top_k=None, top_p=None):
        return self.decoder.sample(logits, temperature, top_k, top_p)


class Neuron:
    def __init__(self, input_size, learning_rate=0.01, hidden_sizes=None):
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes or [])
        self.weights = [0.1] * (self.hidden_sizes[-1] if self.hidden_sizes else input_size)
        self.bias = 0.0
        self.hidden_weights = []
        self.hidden_biases = []

        for layer_in, layer_out in zip([input_size] + self.hidden_sizes, self.hidden_sizes + [1]):
            if layer_in == layer_out:
                continue
            self.hidden_weights.append([0.1 for _ in range(layer_in * layer_out)])
            self.hidden_biases.append([0.0 for _ in range(layer_out)])

        self.loss = MSELoss()
        self.optimizer = SGD(learning_rate)
        self.checkpoint = Checkpoint()

    def _apply_hidden_layers(self, inputs):
        current = list(inputs)
        for index, hidden_size in enumerate(self.hidden_sizes):
            weight_matrix = np.asarray(self.hidden_weights[index], dtype=float).reshape(len(current), hidden_size)
            bias_vector = np.asarray(self.hidden_biases[index], dtype=float)
            current = gelu(np.dot(weight_matrix.T, np.asarray(current, dtype=float)) + bias_vector).tolist()
        return current

    def forward(self, inputs):
        if self.hidden_sizes:
            hidden_output = self._apply_hidden_layers(inputs)
            return sum(w * x for w, x in zip(self.weights, hidden_output)) + self.bias
        return sum(w * x for w, x in zip(self.weights, inputs)) + self.bias

    def backward(self, inputs, loss_gradient):
        weight_gradients = [loss_gradient * x for x in inputs]
        bias_gradient = loss_gradient
        return weight_gradients, bias_gradient

    def train_step(self, inputs, target, checkpoint_path=None):
        prediction = self.forward(inputs)
        loss = self.loss.forward(prediction, target)
        loss_gradient = self.loss.backward(prediction, target)
        weight_gradients, bias_gradient = self.backward(inputs, loss_gradient)

        self.weights, self.bias = self.optimizer.update(
            self.weights,
            self.bias,
            weight_gradients,
            bias_gradient
        )

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)

        return prediction, loss

    def train(self, inputs, targets, epochs=1, checkpoint_path=None):
        if len(inputs) != len(targets):
            raise ValueError("Inputs e targets devono avere la stessa lunghezza.")

        if epochs < 1:
            raise ValueError("epochs deve essere almeno 1.")

        history = []

        for _ in range(epochs):
            epoch_losses = []
            for input_data, target in zip(inputs, targets):
                _, loss = self.train_step(input_data, target, checkpoint_path=None)
                epoch_losses.append(loss)

            average_loss = sum(epoch_losses) / len(epoch_losses)
            history.append(average_loss)

            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)

        return history

    def evaluate(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError("Inputs e targets devono avere la stessa lunghezza.")

        if not inputs:
            raise ValueError("Il dataset di valutazione non può essere vuoto.")

        losses = []
        for input_data, target in zip(inputs, targets):
            prediction = self.forward(input_data)
            losses.append(self.loss.forward(prediction, target))
        return sum(losses) / len(losses)

    def train_with_validation(self, training_inputs, training_targets, validation_inputs, validation_targets, epochs=1, checkpoint_path=None):
        if len(training_inputs) != len(training_targets):
            raise ValueError("Training inputs e targets devono avere la stessa lunghezza.")

        if len(validation_inputs) != len(validation_targets):
            raise ValueError("Validation inputs e targets devono avere la stessa lunghezza.")

        if not training_inputs:
            raise ValueError("Il Training Dataset non può essere vuoto.")

        if not validation_inputs:
            raise ValueError("Il Validation Dataset non può essere vuoto.")

        if epochs < 1:
            raise ValueError("epochs deve essere almeno 1.")

        history = []
        best_validation_loss = None

        for _ in range(epochs):
            training_losses = []
            for input_data, target in zip(training_inputs, training_targets):
                _, loss = self.train_step(input_data, target)
                training_losses.append(loss)

            training_loss = sum(training_losses) / len(training_losses)
            validation_loss = self.evaluate(validation_inputs, validation_targets)
            history.append({"training_loss": training_loss, "validation_loss": validation_loss})

            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)

        return history

    def create_checkpoint(self):
        parameters = {
            "input_size": self.input_size,
            "weights": self.weights,
            "bias": self.bias,
            "hidden_sizes": self.hidden_sizes,
            "hidden_weights": self.hidden_weights,
            "hidden_biases": self.hidden_biases,
        }
        return self.checkpoint.create(parameters)

    def save_checkpoint(self, filepath):
        self.checkpoint.save(self.create_checkpoint(), filepath)

    def load_checkpoint(self, filepath):
        checkpoint = self.checkpoint.load(filepath)
        parameters = checkpoint["parameters"]

        self.input_size = parameters.get("input_size", self.input_size)
        self.weights = parameters["weights"]
        self.bias = parameters["bias"]
        self.hidden_sizes = parameters.get("hidden_sizes", [])
        self.hidden_weights = parameters.get("hidden_weights", [])
        self.hidden_biases = parameters.get("hidden_biases", [])