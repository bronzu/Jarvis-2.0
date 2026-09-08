"""
Transformer proprietario del Neural Core di JARVIS.

Architettura completamente interna: nessun modello, embedding o tokenizer
pre-addestrato, nessuna libreria di deep learning esterna. Solo NumPy.

Componenti:
    - token embedding            (`neural_core.embeddings.Embeddings`)
    - positional encoding        (`PositionalEncoding`, sinusoidale)
    - self attention             (`neural_core.model.SelfAttention`)
    - multi head attention       (`MultiHeadAttention`)
    - feed forward network       (`FeedForward`)
    - layer normalization        (`LayerNorm`)
    - residual connection        (`TransformerBlock`)
    - causal masking             (`neural_core.model.causal_mask`)

Il modello è autoregressivo: dati i token precedenti predice il token
successivo. Ogni componente implementa `forward` e `backward` per rendere il
modello addestrabile dal `Trainer` interno.
"""

import hashlib
import math

import numpy as np

from neural_core.embeddings import Embeddings
from neural_core.model import (
    causal_mask,
    gelu,
    gelu_derivative,
    softmax,
    softmax_backward,
)


def _derive_seed(master_seed, salt):
    """Deriva un seed deterministico da un master seed e un salt."""
    if master_seed is None:
        return None
    digest = hashlib.md5(f"{master_seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)

#: Context window minima supportata dal Neural Core.
MIN_CONTEXT_SIZE = 128

#: Context window consigliate/configurabili.
SUPPORTED_CONTEXT_SIZES = (128, 256, 512, 1024)

#: Context window di default.
DEFAULT_CONTEXT_SIZE = 256


class LayerNorm:
    """Layer normalization con parametri addestrabili `gamma` e `beta`."""

    def __init__(self, d_model, eps=1e-6, dtype=np.float64):
        if d_model <= 0:
            raise ValueError("d_model deve essere maggiore di 0")
        self.d_model = d_model
        self.eps = eps
        self.dtype = np.dtype(dtype)
        self.gamma = np.ones(d_model, dtype=self.dtype)
        self.beta = np.zeros(d_model, dtype=self.dtype)
        self.grad_gamma = np.zeros(d_model, dtype=self.dtype)
        self.grad_beta = np.zeros(d_model, dtype=self.dtype)
        self._cached = {}

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        self._cached = {"x_hat": x_hat, "std": std}
        return self.gamma * x_hat + self.beta

    def backward(self, grad_output):
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        x_hat = self._cached["x_hat"]
        std = self._cached["std"]
        d = self.d_model

        flat_grad = grad_output.reshape(-1, d)
        flat_x_hat = x_hat.reshape(-1, d)
        self.grad_gamma = np.sum(flat_grad * flat_x_hat, axis=0)
        self.grad_beta = np.sum(flat_grad, axis=0)

        grad_x_hat = grad_output * self.gamma
        sum_grad = np.sum(grad_x_hat, axis=-1, keepdims=True)
        sum_grad_x_hat = np.sum(grad_x_hat * x_hat, axis=-1, keepdims=True)
        return (d * grad_x_hat - sum_grad - x_hat * sum_grad_x_hat) / (d * std)

    def parameters(self):
        return [self.gamma, self.beta]

    def gradients(self):
        return [self.grad_gamma, self.grad_beta]

    def set_parameters(self, parameters):
        if len(parameters) != 2:
            raise ValueError("LayerNorm richiede 2 parametri")
        self.gamma = np.asarray(parameters[0], dtype=np.float64)
        self.beta = np.asarray(parameters[1], dtype=np.float64)


class MultiHeadAttention:
    """
    Multi head self attention con causal masking.

    Input e output hanno forma `(batch, seq_len, d_model)`.
    """

    def __init__(self, d_model, num_heads, seed=None, dtype=np.float64):
        if d_model <= 0:
            raise ValueError("d_model deve essere maggiore di 0")
        if num_heads <= 0:
            raise ValueError("num_heads deve essere maggiore di 0")
        if d_model % num_heads != 0:
            raise ValueError("d_model deve essere divisibile per num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        scale = 1.0 / math.sqrt(d_model)

        self.W_q = (rng.randn(d_model, d_model) * scale).astype(self.dtype)
        self.W_k = (rng.randn(d_model, d_model) * scale).astype(self.dtype)
        self.W_v = (rng.randn(d_model, d_model) * scale).astype(self.dtype)
        self.W_o = (rng.randn(d_model, d_model) * scale).astype(self.dtype)
        self.b_q = np.zeros(d_model, dtype=self.dtype)
        self.b_k = np.zeros(d_model, dtype=self.dtype)
        self.b_v = np.zeros(d_model, dtype=self.dtype)
        self.b_o = np.zeros(d_model, dtype=self.dtype)

        self._zero_gradients()
        self._cached = {}

    def _zero_gradients(self):
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_o = np.zeros_like(self.b_o)

    def _split_heads(self, x, batch_size, seq_len):
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x, batch_size, seq_len):
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 3:
            raise ValueError("MultiHeadAttention richiede input (batch, seq_len, d_model).")

        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("La dimensione dell'input non coincide con d_model.")

        x_flat = x.reshape(-1, d_model)
        q = (x_flat @ self.W_q + self.b_q).reshape(batch_size, seq_len, d_model)
        k = (x_flat @ self.W_k + self.b_k).reshape(batch_size, seq_len, d_model)
        v = (x_flat @ self.W_v + self.b_v).reshape(batch_size, seq_len, d_model)

        q_heads = self._split_heads(q, batch_size, seq_len)
        k_heads = self._split_heads(k, batch_size, seq_len)
        v_heads = self._split_heads(v, batch_size, seq_len)

        scores = (q_heads @ k_heads.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + np.asarray(mask, dtype=self.dtype)

        attention = softmax(scores, axis=-1)
        context = attention @ v_heads
        merged = self._merge_heads(context, batch_size, seq_len)
        output = (merged.reshape(-1, d_model) @ self.W_o + self.b_o).reshape(batch_size, seq_len, d_model)

        self._cached = {
            "x_flat": x_flat,
            "q_heads": q_heads,
            "k_heads": k_heads,
            "v_heads": v_heads,
            "attention": attention,
            "merged_flat": merged.reshape(-1, d_model),
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return output

    def backward(self, grad_output):
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        cache = self._cached
        batch_size = cache["batch_size"]
        seq_len = cache["seq_len"]
        d_model = self.d_model

        grad_flat = grad_output.reshape(-1, d_model)
        self.grad_W_o = cache["merged_flat"].T @ grad_flat
        self.grad_b_o = np.sum(grad_flat, axis=0)

        grad_merged = (grad_flat @ self.W_o.T).reshape(batch_size, seq_len, d_model)
        grad_context = self._split_heads(grad_merged, batch_size, seq_len)

        attention = cache["attention"]
        v_heads = cache["v_heads"]
        k_heads = cache["k_heads"]
        q_heads = cache["q_heads"]

        grad_attention = grad_context @ v_heads.transpose(0, 1, 3, 2)
        grad_v_heads = attention.transpose(0, 1, 3, 2) @ grad_context

        grad_scores = softmax_backward(attention, grad_attention, axis=-1) / math.sqrt(self.d_k)
        grad_q_heads = grad_scores @ k_heads
        grad_k_heads = grad_scores.transpose(0, 1, 3, 2) @ q_heads

        grad_q_flat = self._merge_heads(grad_q_heads, batch_size, seq_len).reshape(-1, d_model)
        grad_k_flat = self._merge_heads(grad_k_heads, batch_size, seq_len).reshape(-1, d_model)
        grad_v_flat = self._merge_heads(grad_v_heads, batch_size, seq_len).reshape(-1, d_model)

        x_flat = cache["x_flat"]
        self.grad_W_q = x_flat.T @ grad_q_flat
        self.grad_W_k = x_flat.T @ grad_k_flat
        self.grad_W_v = x_flat.T @ grad_v_flat
        self.grad_b_q = np.sum(grad_q_flat, axis=0)
        self.grad_b_k = np.sum(grad_k_flat, axis=0)
        self.grad_b_v = np.sum(grad_v_flat, axis=0)

        grad_x_flat = (
            grad_q_flat @ self.W_q.T
            + grad_k_flat @ self.W_k.T
            + grad_v_flat @ self.W_v.T
        )
        return grad_x_flat.reshape(batch_size, seq_len, d_model)

    def attention_weights(self):
        """Pesi di attenzione dell'ultima forward, forma `(B, H, T, T)`."""
        if not self._cached:
            raise RuntimeError("Nessuna forward() eseguita.")
        return self._cached["attention"]

    def parameters(self):
        return [
            self.W_q, self.b_q,
            self.W_k, self.b_k,
            self.W_v, self.b_v,
            self.W_o, self.b_o,
        ]

    def gradients(self):
        return [
            self.grad_W_q, self.grad_b_q,
            self.grad_W_k, self.grad_b_k,
            self.grad_W_v, self.grad_b_v,
            self.grad_W_o, self.grad_b_o,
        ]

    def set_parameters(self, parameters):
        if len(parameters) != 8:
            raise ValueError("MultiHeadAttention richiede 8 parametri")
        (
            self.W_q, self.b_q,
            self.W_k, self.b_k,
            self.W_v, self.b_v,
            self.W_o, self.b_o,
        ) = [np.asarray(p, dtype=np.float64) for p in parameters]


class FeedForward:
    """Feed forward network posizionale con attivazione GELU."""

    def __init__(self, d_model, d_ff, seed=None, dtype=np.float64):
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model e d_ff devono essere maggiori di 0")

        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.W1 = (rng.randn(d_model, d_ff) / math.sqrt(d_model)).astype(self.dtype)
        self.b1 = np.zeros(d_ff, dtype=self.dtype)
        self.W2 = (rng.randn(d_ff, d_model) / math.sqrt(d_ff)).astype(self.dtype)
        self.b2 = np.zeros(d_model, dtype=self.dtype)

        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
        self._cached = {}

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        z1 = x_flat @ self.W1 + self.b1
        h1 = gelu(z1)
        out_flat = h1 @ self.W2 + self.b2
        self._cached = {"x_flat": x_flat, "z1": z1, "h1": h1, "shape": shape}
        return out_flat.reshape(shape)

    def backward(self, grad_output):
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        cache = self._cached
        grad_flat = grad_output.reshape(-1, self.d_model)

        self.grad_W2 = cache["h1"].T @ grad_flat
        self.grad_b2 = np.sum(grad_flat, axis=0)

        grad_h1 = grad_flat @ self.W2.T
        grad_z1 = grad_h1 * gelu_derivative(cache["z1"])

        self.grad_W1 = cache["x_flat"].T @ grad_z1
        self.grad_b1 = np.sum(grad_z1, axis=0)

        grad_x_flat = grad_z1 @ self.W1.T
        return grad_x_flat.reshape(cache["shape"])

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def gradients(self):
        return [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]

    def set_parameters(self, parameters):
        if len(parameters) != 4:
            raise ValueError("FeedForward richiede 4 parametri")
        self.W1, self.b1, self.W2, self.b2 = [
            np.asarray(p, dtype=np.float64) for p in parameters
        ]


class TransformerBlock:
    """
    Blocco Transformer pre-norm con residual connection.

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, seed=None, dtype=np.float64):
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve essere compreso tra 0 e 1")

        self.d_model = d_model
        self.dropout = float(dropout)
        self.dtype = np.dtype(dtype)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._initial_rng_state = self.rng.get_state()
        self.attn = MultiHeadAttention(d_model, num_heads, seed=seed, dtype=dtype)
        self.norm1 = LayerNorm(d_model, dtype=dtype)
        self.ff = FeedForward(d_model, d_ff, seed=seed, dtype=dtype)
        self.norm2 = LayerNorm(d_model, dtype=dtype)
        self._cached = {}

    def _dropout_mask(self, shape):
        keep = 1.0 - self.dropout
        return (self.rng.binomial(1, keep, size=shape) / keep).astype(self.dtype)

    def forward(self, x, mask=None, training=False):
        x = np.asarray(x, dtype=self.dtype)

        if training and self.dropout > 0:
            self.rng.set_state(self._initial_rng_state)

        attn_out = self.attn.forward(self.norm1.forward(x), mask=mask)
        attn_mask = None
        if training and self.dropout > 0:
            attn_mask = self._dropout_mask(attn_out.shape)
            attn_out = attn_out * attn_mask
        hidden = x + attn_out

        ff_out = self.ff.forward(self.norm2.forward(hidden))
        ff_mask = None
        if training and self.dropout > 0:
            ff_mask = self._dropout_mask(ff_out.shape)
            ff_out = ff_out * ff_mask
        output = hidden + ff_out

        self._cached = {"attn_mask": attn_mask, "ff_mask": ff_mask}
        return output

    def backward(self, grad_output):
        grad_output = np.asarray(grad_output, dtype=self.dtype)
        attn_mask = self._cached.get("attn_mask")
        ff_mask = self._cached.get("ff_mask")

        grad_ff = grad_output if ff_mask is None else grad_output * ff_mask
        grad_hidden = grad_output + self.norm2.backward(self.ff.backward(grad_ff))

        grad_attn = grad_hidden if attn_mask is None else grad_hidden * attn_mask
        grad_x = grad_hidden + self.norm1.backward(self.attn.backward(grad_attn))
        return grad_x

    def parameters(self):
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.ff.parameters()
        )

    def gradients(self):
        return (
            self.norm1.gradients()
            + self.attn.gradients()
            + self.norm2.gradients()
            + self.ff.gradients()
        )

    def set_parameters(self, parameters):
        expected = 2 + 8 + 2 + 4
        if len(parameters) != expected:
            raise ValueError(f"TransformerBlock richiede {expected} parametri")
        self.norm1.set_parameters(parameters[0:2])
        self.attn.set_parameters(parameters[2:10])
        self.norm2.set_parameters(parameters[10:12])
        self.ff.set_parameters(parameters[12:16])


class PositionalEncoding:
    """Positional encoding sinusoidale deterministico (senza parametri)."""

    def __init__(self, d_model, max_len=DEFAULT_CONTEXT_SIZE, dtype=np.float64):
        if d_model <= 0:
            raise ValueError("d_model deve essere maggiore di 0")
        if max_len <= 0:
            raise ValueError("max_len deve essere maggiore di 0")

        self.d_model = d_model
        self.max_len = max_len
        self.dtype = np.dtype(dtype)

        pe = np.zeros((max_len, d_model), dtype=self.dtype)
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        cos_columns = pe[:, 1::2].shape[1]
        pe[:, 1::2] = np.cos(position * div_term[:cos_columns])
        self.pe = pe

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        seq_len = x.shape[-2]
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequenza di {seq_len} token oltre la context window di {self.max_len}."
            )
        return x + self.pe[:seq_len]

    def backward(self, grad_output):
        """Il positional encoding è additivo e senza parametri."""
        return np.asarray(grad_output, dtype=self.dtype)


class MicroTransformer:
    """
    Transformer decoder-only autoregressivo di JARVIS.

        input:  sequenza di token precedenti  `(batch, seq_len)`
        output: logits del token successivo   `(batch, seq_len, vocab_size)`

    La posizione `t` dell'output predice il token `t + 1` dell'input, grazie al
    causal masking. Il modello è interamente proprietario e addestrabile:
    espone `forward`, `backward`, `get_parameters` e `set_parameters`, lo stesso
    protocollo usato dal `Trainer` interno.

    Context window (`max_len`): minimo 128 token, configurabile a 256 o 512.
    """

    def __init__(
        self,
        vocab_size,
        d_model=64,
        num_heads=2,
        d_ff=128,
        num_layers=2,
        max_len=DEFAULT_CONTEXT_SIZE,
        dropout=0.0,
        seed=None,
        dtype=np.float64,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size deve essere maggiore di 0")
        if num_layers <= 0:
            raise ValueError("num_layers deve essere maggiore di 0")
        if max_len <= 0:
            raise ValueError("max_len deve essere maggiore di 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve essere compreso tra 0 e 1")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = float(dropout)
        self.seed = seed
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random

        embedding_seed = _derive_seed(seed, "embedding")
        self.embedding = Embeddings(
            vocab_size,
            d_model,
            init_scale=1.0 / math.sqrt(d_model),
            seed=embedding_seed,
            dtype=dtype,
        )
        self.embedding_multiplier = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dtype=dtype)
        self.layers = [
            TransformerBlock(
                d_model, num_heads, d_ff,
                dropout=dropout,
                seed=_derive_seed(seed, f"layer{i}"),
                dtype=dtype,
            )
            for i in range(num_layers)
        ]
        self.norm = LayerNorm(d_model, dtype=dtype)
        output_seed = _derive_seed(seed, "output")
        output_rng = np.random.RandomState(output_seed) if output_seed is not None else np.random
        self.output_layer = (output_rng.randn(d_model, vocab_size) / math.sqrt(d_model)).astype(self.dtype)
        self.output_bias = np.zeros(vocab_size, dtype=self.dtype)

        self.grad_output_layer = np.zeros_like(self.output_layer)
        self.grad_output_bias = np.zeros_like(self.output_bias)
        self.grad_embeddings = np.zeros_like(self.embedding.weights)

        self._mask_cache = {}
        self._cached = {}

    # ------------------------------------------------------------------
    # Context window
    # ------------------------------------------------------------------

    @property
    def context_size(self):
        """Alias leggibile della context window."""
        return self.max_len

    def supports_context(self, seq_len):
        """True se il modello può processare `seq_len` token."""
        return 0 < int(seq_len) <= self.max_len

    def config(self):
        """Iperparametri necessari a ricostruire il modello."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "dtype": str(self.dtype),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _normalize_token_ids(self, token_ids):
        if isinstance(token_ids, (int, np.integer)):
            ids = np.asarray([[token_ids]], dtype=np.int64)
        else:
            ids = np.asarray(token_ids, dtype=np.int64)
            if ids.ndim == 0:
                ids = ids.reshape(1, 1)
            elif ids.ndim == 1:
                ids = ids.reshape(1, -1)
            elif ids.ndim != 2:
                raise ValueError("token_ids deve avere forma (seq_len,) o (batch, seq_len).")

        if ids.shape[1] == 0:
            raise ValueError("La sequenza di input non può essere vuota.")

        if ids.shape[1] > self.max_len:
            raise ValueError(
                f"Sequenza di {ids.shape[1]} token oltre la context window di {self.max_len}."
            )

        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise IndexError("token_ids contiene indici fuori dal vocabolario.")

        return ids

    def _causal_mask(self, seq_len):
        """Maschera causale additiva `(1, 1, seq_len, seq_len)` con cache."""
        cached = self._mask_cache.get(seq_len)
        if cached is None:
            cached = causal_mask(seq_len)
            self._mask_cache[seq_len] = cached
        return cached

    def forward(self, token_ids, training=False):
        ids = self._normalize_token_ids(token_ids)
        batch_size, seq_len = ids.shape

        mask = self._causal_mask(seq_len)

        x = self.embedding.forward_matrix(ids) * self.embedding_multiplier
        x = self.pos_encoding.forward(x)

        for layer in self.layers:
            x = layer.forward(x, mask=mask, training=training)

        x = self.norm.forward(x)
        x_flat = x.reshape(-1, self.d_model)
        logits = (x_flat @ self.output_layer + self.output_bias).reshape(
            batch_size, seq_len, self.vocab_size
        )

        self._cached = {
            "token_ids": ids,
            "x_flat": x_flat,
            "logits": logits,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return logits

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, targets, ignore_index=-100, probs=None):
        """
        Backpropagation della cross entropy sul token successivo.

        Args:
            targets: `(batch, seq_len)` oppure `(seq_len,)`.
            ignore_index: posizioni da escludere dalla loss.
            probs: probabilità opzionali calcolate in `forward()` per riutilizzare
                la softmax e evitare il calcolo duplicato.

        Returns:
            Lista di gradienti nello stesso ordine di `get_parameters()`.
        """
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        batch_size = self._cached["batch_size"]
        seq_len = self._cached["seq_len"]
        logits = self._cached["logits"]

        targets = np.asarray(targets, dtype=np.int64)
        if targets.ndim == 1:
            targets = targets.reshape(batch_size, seq_len)
        if targets.shape != (batch_size, seq_len):
            raise ValueError(
                "targets deve avere la stessa forma dell'input (batch, seq_len)."
            )

        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = targets.reshape(-1)

        if probs is None:
            probs = softmax(flat_logits, axis=-1)
        else:
            probs = probs.copy()

        valid = flat_targets != ignore_index
        valid_count = int(np.sum(valid))
        if valid_count == 0:
            raise ValueError("Nessun target valido nella batch.")

        safe_targets = np.where(valid, flat_targets, 0)
        probs[np.arange(flat_logits.shape[0]), safe_targets] -= 1.0
        probs[~valid] = 0.0
        grad_logits_flat = probs / valid_count

        self.grad_output_layer = self._cached["x_flat"].T @ grad_logits_flat
        self.grad_output_bias = np.sum(grad_logits_flat, axis=0)

        grad_x = (grad_logits_flat @ self.output_layer.T).reshape(
            batch_size, seq_len, self.d_model
        )
        grad_x = self.norm.backward(grad_x)

        for layer in reversed(self.layers):
            grad_x = layer.backward(grad_x)

        grad_x = self.pos_encoding.backward(grad_x)
        grad_embed_output = grad_x * self.embedding_multiplier
        self.grad_embeddings = self.embedding.backward(
            self._cached["token_ids"], grad_embed_output
        )

        return self.get_gradients()

    # ------------------------------------------------------------------
    # Parametri
    # ------------------------------------------------------------------

    def get_parameters(self):
        parameters = [self.embedding.weights]
        for layer in self.layers:
            parameters.extend(layer.parameters())
        parameters.extend(self.norm.parameters())
        parameters.extend([self.output_layer, self.output_bias])
        return parameters

    def get_gradients(self):
        gradients = [self.grad_embeddings]
        for layer in self.layers:
            gradients.extend(layer.gradients())
        gradients.extend(self.norm.gradients())
        gradients.extend([self.grad_output_layer, self.grad_output_bias])
        return gradients

    def set_parameters(self, parameters):
        parameters = list(parameters)
        expected = len(self.get_parameters())
        if len(parameters) != expected:
            raise ValueError(f"MicroTransformer richiede {expected} parametri")

        self.embedding.weights = np.asarray(parameters[0], dtype=self.dtype)

        index = 1
        block_size = 16
        for layer in self.layers:
            layer.set_parameters(parameters[index:index + block_size])
            index += block_size

        self.norm.set_parameters(parameters[index:index + 2])
        index += 2

        self.output_layer = np.asarray(parameters[index], dtype=self.dtype)
        self.output_bias = np.asarray(parameters[index + 1], dtype=self.dtype)

    def parameter_count(self):
        """Numero totale di parametri addestrabili."""
        return int(sum(np.asarray(p).size for p in self.get_parameters()))

    def parameter_count_breakdown(self):
        total = self.parameter_count()
        embedding = int(self.embedding.weights.size)
        attention = 0
        ffn = 0
        layernorm = 0
        output = int(self.output_layer.size + self.output_bias.size)

        for layer in self.layers:
            attention += int(sum(np.asarray(p).size for p in layer.attn.parameters()))
            ffn += int(sum(np.asarray(p).size for p in layer.ff.parameters()))
            layernorm += int(sum(np.asarray(p).size for p in layer.norm1.parameters()))
            layernorm += int(sum(np.asarray(p).size for p in layer.norm2.parameters()))

        layernorm += int(sum(np.asarray(p).size for p in self.norm.parameters()))

        return {
            "total": total,
            "trainable": total,
            "non_trainable": 0,
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "layernorm": layernorm,
            "output": output,
        }

    def attention_weights(self, layer_index=0):
        """Pesi di attenzione di un layer dopo l'ultima forward."""
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("layer_index fuori intervallo.")
        return self.layers[layer_index].attn.attention_weights()
