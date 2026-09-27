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


# ----------------------------------------------------------------------
# Causal Linear Attention — O(N) replacement for Scaled Dot-Product
# ----------------------------------------------------------------------

#: Numerical epsilon for linear attention denominator stability.
_LINEAR_EPS = 1e-6


def elu_plus_one(x):
    """Feature map positiva phi(x) = ELU(x) + 1.

    ELU(x) + 1 = x + 1   se x >= 0
                exp(x)   se x < 0

    La mappa è sempre strettamente positiva (range (0, +inf)), garantendo
    che i prodotti cumsum siano ben condizionati e la causalità preservata.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, x + 1.0, np.exp(x))


def elu_plus_one_derivative(x):
    """Derivata di phi(x) = ELU(x) + 1.

    phi'(x) = 1   se x >= 0
             = exp(x)  se x < 0
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0, np.exp(x))


def _reverse_cumsum(tensor, axis=2):
    """Inverse dell'operatore cumsum lungo *axis*.

    Se ``cum = np.cumsum(x, axis=axis)`` allora
    ``_reverse_cumsum(cum, axis) == x``.

    Implementato come cumsum sull'array invertito e poi ri-invertito,
    mantenendo la complessità O(N).
    """
    sl = [slice(None)] * tensor.ndim
    sl[axis] = slice(None, None, -1)
    sl = tuple(sl)
    return np.cumsum(tensor[sl], axis=axis)[sl]


class MultiHeadLinearAttention:
    """
    Causal Linear Attention multi-head — sostituzione O(N) per
    :class:`MultiHeadAttention`.

    Invece di calcolare la matrice di attenzione N×N tramite
    ``softmax(QK^T)V`` (O(N²)), sfrutta la proprietà associativa
    della moltiplicazione e le somme cumulative:

        1. Feature map positiva:  phi(x) = ELU(x) + 1
        2. Numeratore:  phi(Q_i) · Σ_{j≤i} phi(K_j)^T V_j   (cumsum)
        3. Denominatore: phi(Q_i) · Σ_{j≤i} phi(K_j)^T      (cumsum)
        4. Output:  num_i / (den_i + eps)

    La causalità è intrinseca nella cumsum: nessun token futuro avvia
    informazione su posizioni precedenti (no future leakage).

    La classe espone la **stessa interfaccia** di ``MultiHeadAttention``
    (forward(x, mask), backward, parameters, gradients, set_parameters,
    attention_weights), in modo da essere un drop-in replacement nel
    :class:`TransformerBlock` e nel :class:`MicroTransformer`.

    Input / Output shape: ``(batch, seq_len, d_model)`` — identica a
    :class:`MultiHeadAttention`.
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

    # ------------------------------------------------------------------
    # Gradienti
    # ------------------------------------------------------------------

    def _zero_gradients(self):
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_o = np.zeros_like(self.b_o)

    # ------------------------------------------------------------------
    # Head splitting / merging (identici a MultiHeadAttention)
    # ------------------------------------------------------------------

    def _split_heads(self, x, batch_size, seq_len):
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x, batch_size, seq_len):
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, mask=None):
        """Forward causal linear attention.

        Args:
            x: tensore input ``(batch, seq_len, d_model)``.
            mask: accettato per compatibilità API (``causal_mask``).
                  La causalità è già garantita dalla cumsum; il mask
                  viene ignorato in quanto ridondante.

        Returns:
            Output ``(batch, seq_len, d_model)`` — stessa forma di input.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 3:
            raise ValueError("MultiHeadLinearAttention richiede input (batch, seq_len, d_model).")

        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError("La dimensione dell'input non coincide con d_model.")

        x_flat = x.reshape(-1, d_model)
        q = (x_flat @ self.W_q + self.b_q).reshape(batch_size, seq_len, d_model)
        k = (x_flat @ self.W_k + self.b_k).reshape(batch_size, seq_len, d_model)
        v = (x_flat @ self.W_v + self.b_v).reshape(batch_size, seq_len, d_model)

        q_heads = self._split_heads(q, batch_size, seq_len)  # (B, H, T, d_k)
        k_heads = self._split_heads(k, batch_size, seq_len)  # (B, H, T, d_k)
        v_heads = self._split_heads(v, batch_size, seq_len)  # (B, H, T, d_k)

        # 1. Feature map positiva phi(x) = ELU(x) + 1
        phi_q = elu_plus_one(q_heads)   # (B, H, T, d_k) — positiva
        phi_k = elu_plus_one(k_heads)   # (B, H, T, d_k) — positiva

        # 2. Somme cumulative per causalità (cumsum lungo T = asse 2)
        cumsum_phi_k = np.cumsum(phi_k, axis=2)                     # (B, H, T, d_k)
        outer_prod = np.einsum('bhtd,bhtv->bhtdv', phi_k, v_heads)  # (B, H, T, d_k, d_k)
        cumsum_kv = np.cumsum(outer_prod, axis=2)                   # (B, H, T, d_k, d_k)

        # 3. Numeratore e denominatore
        numerator = np.einsum('bhtd,bhtdv->bhtv', phi_q, cumsum_kv)   # (B, H, T, d_k)
        denominator = np.einsum('bhtd,bhtd->bht', phi_q, cumsum_phi_k)  # (B, H, T)

        # 4. Output normalizzato
        attention_output = numerator / (denominator[:, :, :, None] + _LINEAR_EPS)

        # 5. Merge heads + proiezione output
        merged = self._merge_heads(attention_output, batch_size, seq_len)
        output = (merged.reshape(-1, d_model) @ self.W_o + self.b_o).reshape(
            batch_size, seq_len, d_model
        )

        self._cached = {
            'x_flat': x_flat,
            'q_heads': q_heads,
            'k_heads': k_heads,
            'v_heads': v_heads,
            'phi_q': phi_q,
            'phi_k': phi_k,
            'cumsum_phi_k': cumsum_phi_k,
            'cumsum_kv': cumsum_kv,
            'outer_prod': outer_prod,
            'numerator': numerator,
            'denominator': denominator,
            'attention_output': attention_output,
            'merged_flat': merged.reshape(-1, d_model),
            'batch_size': batch_size,
            'seq_len': seq_len,
        }
        return output

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, grad_output):
        """Backpropagation dell'attention lineare causale.

        Returns:
            Gradiente rispetto all'input ``x``, forma ``(batch, seq_len, d_model)``.
            Popola internamente ``grad_W_q``, ``grad_W_k``, ``grad_W_v``,
            ``grad_W_o`` e i bias gradienti.
        """
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        cache = self._cached
        batch_size = cache["batch_size"]
        seq_len = cache["seq_len"]
        d_model = self.d_model

        phi_q = cache['phi_q']
        phi_k = cache['phi_k']
        q_heads = cache['q_heads']
        k_heads = cache['k_heads']
        v_heads = cache['v_heads']
        cumsum_phi_k = cache['cumsum_phi_k']
        cumsum_kv = cache['cumsum_kv']
        numerator = cache['numerator']
        denominator = cache['denominator']
        attention_output = cache['attention_output']
        merged_flat = cache['merged_flat']

        # --- W_o projection (identico a MultiHeadAttention) ---
        grad_flat = grad_output.reshape(-1, d_model)
        self.grad_W_o = merged_flat.T @ grad_flat
        self.grad_b_o = np.sum(grad_flat, axis=0)

        grad_merged = (grad_flat @ self.W_o.T).reshape(batch_size, seq_len, d_model)
        grad_attn_output = self._split_heads(grad_merged, batch_size, seq_len)  # (B, H, T, d_k)

        # --- Output = num / (den + eps) ---
        den_exp = denominator[:, :, :, None] + _LINEAR_EPS
        d_numerator = grad_attn_output / den_exp  # (B, H, T, d_k)
        d_denominator = -np.sum(
            grad_attn_output * attention_output / den_exp, axis=-1
        )  # (B, H, T)

        # --- Numerator: einsum('bhtd,bhtdv->bhtv', phi_q, cumsum_kv) ---
        grad_phi_q_from_num = np.einsum('bhtv,bhtdv->bhtd', d_numerator, cumsum_kv)  # (B, H, T, d_k)
        grad_cumsum_kv = np.einsum('bhtd,bhtv->bhtdv', phi_q, d_numerator)  # (B, H, T, d_k, d_k)

        # --- Denominator: einsum('bhtd,bhtd->bht', phi_q, cumsum_phi_k) ---
        grad_phi_q_from_den = d_denominator[:, :, :, None] * cumsum_phi_k  # (B, H, T, d_k)
        grad_cumsum_phi_k = d_denominator[:, :, :, None] * phi_q  # (B, H, T, d_k)

        # --- Total phi_q gradient ---
        grad_phi_q = grad_phi_q_from_num + grad_phi_q_from_den

        # --- Reverse cumsum: recupera gradienti rispetto ai valori per-token ---
        grad_phi_k_from_cumsum = _reverse_cumsum(grad_cumsum_phi_k, axis=2)  # (B, H, T, d_k)
        grad_outer_prod = _reverse_cumsum(grad_cumsum_kv, axis=2)  # (B, H, T, d_k, d_k)

        # --- Outer product: phi_k ⊗ v_heads ---
        # outer[b,h,t,d,v] = phi_k[b,h,t,d] * v_heads[b,h,t,v]
        grad_phi_k_from_outer = np.einsum('bhtdv,bhtv->bhtd', grad_outer_prod, v_heads)  # (B, H, T, d_k)
        grad_v_heads = np.einsum('bhtdv,bhtd->bhtv', grad_outer_prod, phi_k)  # (B, H, T, d_k)

        # --- Total k gradient ---
        grad_phi_k = grad_phi_k_from_cumsum + grad_phi_k_from_outer

        # --- Through feature map phi(x) = ELU(x) + 1 ---
        grad_q_heads = grad_phi_q * elu_plus_one_derivative(q_heads)  # (B, H, T, d_k)
        grad_k_heads = grad_phi_k * elu_plus_one_derivative(k_heads)  # (B, H, T, d_k)
        # V non ha feature map — grad_v_heads è già corretto

        # --- Merge heads ---
        grad_q = self._merge_heads(grad_q_heads, batch_size, seq_len)  # (B, T, d_model)
        grad_k = self._merge_heads(grad_k_heads, batch_size, seq_len)
        grad_v = self._merge_heads(grad_v_heads, batch_size, seq_len)

        # --- W_q, W_k, W_v projections (identico a MultiHeadAttention) ---
        grad_q_flat = grad_q.reshape(-1, d_model)
        grad_k_flat = grad_k.reshape(-1, d_model)
        grad_v_flat = grad_v.reshape(-1, d_model)

        x_flat = cache['x_flat']
        self.grad_W_q = x_flat.T @ grad_q_flat
        self.grad_W_k = x_flat.T @ grad_k_flat
        self.grad_W_v = x_flat.T @ grad_v_flat
        self.grad_b_q = np.sum(grad_q_flat, axis=0)
        self.grad_b_k = np.sum(grad_k_flat, axis=0)
        self.grad_b_v = np.sum(grad_v_flat, axis=0)

        # --- Input gradient ---
        grad_x_flat = (
            grad_q_flat @ self.W_q.T
            + grad_k_flat @ self.W_k.T
            + grad_v_flat @ self.W_v.T
        )
        return grad_x_flat.reshape(batch_size, seq_len, d_model)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def attention_weights(self):
        """Pesi di attenzione efficaci ``(batch, num_heads, seq_len, seq_len)``.

        Nota: la computazione esplicita della matrice di attenzione è O(N²)
        e viene eseguita solo su richiesta per ispezione/debug. Il forward
        pass mantiene comunque complessità O(N) grazie alla cumsum.
        """
        if not self._cached:
            raise RuntimeError("Nessuna forward() eseguita.")

        phi_q = self._cached['phi_q']        # (B, H, T, d_k)
        phi_k = self._cached['phi_k']         # (B, H, T, d_k)
        cumsum_phi_k = self._cached['cumsum_phi_k']  # (B, H, T, d_k)
        denominator = self._cached['denominator']    # (B, H, T)

        # w_ij = phi_q[i] · phi_k[j] / den_i   per j <= i (causale)
        raw = np.einsum('bhtd,bhsd->bhts', phi_q, phi_k)  # (B, H, T, T)
        causal_raw = np.tril(raw, k=0)
        weights = causal_raw / (denominator[:, :, :, None] + _LINEAR_EPS)
        return weights

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
            raise ValueError("MultiHeadLinearAttention richiede 8 parametri")
        (
            self.W_q, self.b_q,
            self.W_k, self.b_k,
            self.W_v, self.b_v,
            self.W_o, self.b_o,
        ) = [np.asarray(p, dtype=np.float64) for p in parameters]

    def get_parameters(self):
        """Alias di :meth:`parameters` per compatibilità con MicroTransformer."""
        return self.parameters()


class LoRALinear:
    """
    Low-Rank Adaptation (LoRA) per un layer lineare congelato.

    Per una matrice `W` (d_out, d_in) congelata, la correzione è:

        Delta = B @ A     con A: (rank, d_in), B: (d_out, rank)
        output_correzione = (B @ A) @ x * scaling

    dove `scaling = alpha / rank`.

    Convenzione di inizializzazione standard LoRA:
        - A: inizializzata con Kaiming uniform (random, addestrabile)
        - B: inizializzata a ZERO (addestrabile)

    Questa scelta garantisce che all'inizio del training la correzione sia
    nulla (Delta = 0) e il comportamento sia identico al modello base, mentre
    nel corso del training solo A e B vengono aggiornati.

    W e b del layer host sono CONGELATI: non ricevono gradienti e non vengono
    aggiornati dall'ottimizzatore. Solo A e B sono parametri addestrabili.

    numero di parametri addestrabili = rank * (d_in + d_out)
    vs d_in * d_out della matrice piena — riduzione >99% per rank=8 su d=256.

    Interfaccia identica alle altre layer (forward/backward/parameters/gradients/
    set_parameters) per essere inserita in FeedForward in trasparenza.
    """

    def __init__(self, d_in, d_out, rank=8, alpha=16.0, seed=None, dtype=np.float64):
        if rank <= 0:
            raise ValueError("rank deve essere maggiore di 0")
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in e d_out devono essere maggiori di 0")
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        # A: (rank, d_in) — Kaiming uniform
        limit_a = np.sqrt(6.0 / (self.rank + self.d_in))
        self.A = (rng.uniform(-limit_a, limit_a, size=(self.rank, self.d_in))).astype(self.dtype)
        # B: (d_out, rank) — inizializzato a ZERO (convenzione LoRA)
        self.B = np.zeros((self.d_out, self.rank), dtype=self.dtype)

        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)
        self._cached = {}

    def _zero_gradients(self):
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)

    def forward(self, x):
        """
        Calcola la correzione LoRA: scaling * (B @ A) @ x.

        Args:
            x: (..., d_in)

        Returns:
            (..., d_out) — correzione da sommare all'output del layer congelato.
        """
        x = np.asarray(x, dtype=self.dtype)
        shape = x.shape
        x_flat = x.reshape(-1, self.d_in)
        # Ax = x @ A.T   -> (N, rank)
        # BAx = Ax @ B.T  -> (N, d_out)
        Ax = x_flat @ self.A.T
        BAx = Ax @ self.B.T
        correction = (BAx * self.scaling).reshape(shape[:-1] + (self.d_out,))
        self._cached = {"x_flat": x_flat, "Ax": Ax}
        return correction

    def backward(self, grad_output):
        """
        Gradiente della correzione LoRA rispetto a input, A e B.

        Correzione = scaling * B @ A @ x.
        dL/dB = (dL/dcorrezione_flat).T @ Ax
        dL/dA = ((dL/dcorrezione_flat) @ B).T @ x_flat
        dL/dx = ((dL/dcorrezione_flat) @ B) @ A
        """
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")
        grad_output = np.asarray(grad_output, dtype=self.dtype)
        shape = grad_output.shape
        grad_flat = grad_output.reshape(-1, self.d_out) * self.scaling
        cache = self._cached
        Ax = cache["Ax"]
        x_flat = cache["x_flat"]

        d_BAx = grad_flat  # (N, d_out)
        d_Ax = d_BAx @ self.B  # (N, rank)
        self.grad_A = d_Ax.T @ x_flat  # (rank, d_in)
        self.grad_B = d_BAx.T @ Ax  # (d_out, rank)
        d_x = (d_Ax @ self.A).reshape(shape[:-1] + (self.d_in,))
        return d_x

    def parameters(self):
        return [self.A, self.B]

    def gradients(self):
        return [self.grad_A, self.grad_B]

    def set_parameters(self, parameters):
        if len(parameters) != 2:
            raise ValueError("LoRALinear richiede 2 parametri (A, B)")
        self.A = np.asarray(parameters[0], dtype=self.dtype)
        self.B = np.asarray(parameters[1], dtype=self.dtype)

    def get_parameters(self):
        """Alias di :meth:`parameters` per compatibilità con MicroTransformer."""
        return self.parameters()

    def parameter_count(self):
        return int(self.A.size + self.B.size)


class FeedForward:
    """Feed forward network posizionale con attivazione GELU.

    Se `lora_rank > 0`, inietta una matrice LoRA sul layer finale W2
    (d_ff -> d_model):  output = h1 @ W2 + b2 + scaling * B @ A @ h1.
    I pesi W1, b1, W2, b2 sono congelati; solo A e B sono addestrabili.
    """

    def __init__(self, d_model, d_ff, seed=None, dtype=np.float64,
                 lora_rank: int = 0, lora_alpha: float = 16.0):
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model e d_ff devono essere maggiori di 0")
        if lora_rank < 0:
            raise ValueError("lora_rank deve essere >= 0 (0 = disattivato)")

        self.d_model = d_model
        self.d_ff = d_ff
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.W1 = (rng.randn(d_model, d_ff) / math.sqrt(d_model)).astype(self.dtype)
        self.b1 = np.zeros(d_ff, dtype=self.dtype)
        self.W2 = (rng.randn(d_ff, d_model) / math.sqrt(d_ff)).astype(self.dtype)
        self.b2 = np.zeros(d_model, dtype=self.dtype)

        # Stack di adattatori LoRA, uno per task (design option (b)).
        # LoRA su W2 (d_ff -> d_model): A (rank, d_ff), B (d_model, rank).
        # Il forward applica la SOMMA delle correzioni di tutti gli adapter
        # registrati: il modello deployed e' un unico oggetto
        #   output = FFN_base(x) + sum_k scaling * B_k @ A_k @ GELU(W1 x + b1)
        # e non richiede di sapere a quale task appartiene la query.
        self.lora_stack = []
        if self.lora_rank > 0:
            lora_seed = _derive_seed(seed, f"lora-{d_model}-{d_ff}")
            self.lora_stack.append(LoRALinear(
                d_in=d_ff, d_out=d_model, rank=self.lora_rank,
                alpha=self.lora_alpha, seed=lora_seed, dtype=self.dtype,
            ))

        self._zero_gradients()
        self._cached = {}

    @property
    def lora(self):
        """Primo adapter dello stack (None se LoRA disattivata)."""
        return self.lora_stack[0] if self.lora_stack else None

    @property
    def num_adapters(self):
        return len(self.lora_stack)

    def add_adapter(self, warm_start=None, seed=None):
        """
        Aggiunge un adapter LoRA dedicato a un nuovo task.

        Args:
            warm_start: se `True` (default quando non specificato), il nuovo
                adapter inizializzato con i pesi dell'ultimo adapter attivo.
                E' la condizione che permette al Micro-EWC di avere un
                significato: il nuovo task parte dall'ottimo del task
                precedente nello spazio dei parametri, esattamente come
                l'EWC full-parameter assume theta_{k+1} = theta_k.
                Se `False`, il nuovo adapter parte da B=0 (adattamento nullo).
        """
        if self.lora_rank <= 0:
            raise ValueError("add_adapter() richiede lora_rank > 0")
        warm = self.lora_stack[-1] if self.lora_stack else None
        if warm_start is None:
            warm_start = True
        if warm_start and warm is not None:
            new = LoRALinear(
                d_in=self.d_ff, d_out=self.d_model, rank=self.lora_rank,
                alpha=self.lora_alpha, seed=seed, dtype=self.dtype,
            )
            new.A = warm.A.copy()
            new.B = warm.B.copy()
        else:
            s = seed if seed is not None else _derive_seed(seed, f"lora-adapter-{len(self.lora_stack)}")
            new = LoRALinear(
                d_in=self.d_ff, d_out=self.d_model, rank=self.lora_rank,
                alpha=self.lora_alpha, seed=s, dtype=self.dtype,
            )
        self.lora_stack.append(new)
        return new

    def _zero_gradients(self):
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
        for adapter in self.lora_stack:
            adapter._zero_gradients()

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        shape = x.shape
        x_flat = x.reshape(-1, self.d_model)
        z1 = x_flat @ self.W1 + self.b1
        h1 = gelu(z1)
        out_flat = h1 @ self.W2 + self.b2
        # Somma delle correzioni di tutti gli adapter registrati (stacking).
        for adapter in self.lora_stack:
            out_flat = out_flat + adapter.forward(h1)
        self._cached = {"x_flat": x_flat, "z1": z1, "h1": h1, "shape": shape}
        return out_flat.reshape(shape)

    def backward(self, grad_output):
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        cache = self._cached
        grad_flat = grad_output.reshape(-1, self.d_model)

        if self.lora_stack:
            # Con LoRA attivo W1, b1, W2, b2 sono congelati: i loro gradienti
            # sono azzerati. Il gradiente verso h1 passa comunque attraverso
            # W2.T (corretto per la backpropagation verso gli strati precedenti)
            # e riceve in piu' il contributo delle correzioni LoRA.
            self.grad_W1 = np.zeros_like(self.W1)
            self.grad_b1 = np.zeros_like(self.b1)
            self.grad_W2 = np.zeros_like(self.W2)
            self.grad_b2 = np.zeros_like(self.b2)

            grad_h1 = grad_flat @ self.W2.T
            # Somma dei gradienti di tutti gli adapter. I gradienti degli
            # adapter non attivi vengono azzerati dal MicroTransformer
            # (un solo adapter e' addestrabile per volta).
            for adapter in self.lora_stack:
                grad_h1 = grad_h1 + adapter.backward(grad_output).reshape(grad_h1.shape)
        else:
            # Senza LoRA: comportamento legacy full-parameter invariato.
            self.grad_W2 = cache["h1"].T @ grad_flat
            self.grad_b2 = np.sum(grad_flat, axis=0)
            grad_h1 = grad_flat @ self.W2.T

        grad_z1 = grad_h1 * gelu_derivative(cache["z1"])

        if self.lora is None:
            self.grad_W1 = cache["x_flat"].T @ grad_z1
            self.grad_b1 = np.sum(grad_z1, axis=0)

        grad_x_flat = grad_z1 @ self.W1.T
        return grad_x_flat.reshape(cache["shape"])

    def parameters(self):
        params = [self.W1, self.b1, self.W2, self.b2]
        for adapter in self.lora_stack:
            params.extend(adapter.parameters())
        return params

    def gradients(self):
        grads = [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]
        for adapter in self.lora_stack:
            grads.extend(adapter.gradients())
        return grads

    def set_parameters(self, parameters):
        parameters = list(parameters)
        n_base = 4
        n_expected = n_base + 2 * len(self.lora_stack)
        if len(parameters) != n_expected:
            raise ValueError(
                f"FeedForward con {len(self.lora_stack)} adapter LoRA richiede "
                f"{n_expected} parametri, ricevuti {len(parameters)}"
            )
        self.W1, self.b1, self.W2, self.b2 = [
            np.asarray(p, dtype=np.float64) for p in parameters[:n_base]
        ]
        idx = n_base
        for adapter in self.lora_stack:
            adapter.set_parameters(parameters[idx:idx + 2])
            idx += 2


class TransformerBlock:
    """
    Blocco Transformer pre-norm con residual connection.

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0, seed=None, dtype=np.float64,
                 lora_rank: int = 0, lora_alpha: float = 16.0):
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve essere compreso tra 0 e 1")
        if lora_rank < 0:
            raise ValueError("lora_rank deve essere >= 0")

        self.d_model = d_model
        self.dropout = float(dropout)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.dtype = np.dtype(dtype)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._initial_rng_state = self.rng.get_state()
        self.attn = MultiHeadAttention(d_model, num_heads, seed=seed, dtype=dtype)
        self.norm1 = LayerNorm(d_model, dtype=dtype)
        self.ff = FeedForward(
            d_model, d_ff, seed=seed, dtype=dtype,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )
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
        parameters = list(parameters)
        ff_size = 4 + 2 * self.ff.num_adapters
        expected = 2 + 8 + 2 + ff_size
        if len(parameters) != expected:
            raise ValueError(
                f"TransformerBlock richiede {expected} parametri "
                f"({self.ff.num_adapters} adapter LoRA), "
                f"ricevuti {len(parameters)}"
            )
        self.norm1.set_parameters(parameters[0:2])
        self.attn.set_parameters(parameters[2:10])
        self.norm2.set_parameters(parameters[10:12])
        self.ff.set_parameters(parameters[12:12 + ff_size])


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
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size deve essere maggiore di 0")
        if num_layers <= 0:
            raise ValueError("num_layers deve essere maggiore di 0")
        if max_len <= 0:
            raise ValueError("max_len deve essere maggiore di 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve essere compreso tra 0 e 1")
        if lora_rank < 0:
            raise ValueError("lora_rank deve essere >= 0 (0 = disattivato)")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = float(dropout)
        self.seed = seed
        self.dtype = np.dtype(dtype)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        # Task attivamente addestrabile sugli adapter LoRA. Con LoRA disattivata
        # resta None: in quel caso valgono i parametri full-parameter legacy.
        self._active_task = 0 if lora_rank > 0 else None

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
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
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
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
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

        gradients = self.get_gradients()
        return self._mask_frozen_gradients(gradients)

    def _mask_frozen_gradients(self, gradients):
        """
        Azzera i gradienti dei parametri congelati quando LoRA e' attivo.

        Senza LoRA non viene applicata alcuna maschera: tutti i parametri restano
        addestrabili e il comportamento e' quello legacy full-parameter.

        Con LoRA attivo, i gradienti dei parametri pre-addestrati (embedding,
        attention, LayerNorm, W1/b1/W2/b2, output) e degli adapter LoRA dei
        task precedenti sono azzerati gia' alla fonte. Questo rende il
        congelamento una proprieta' del MODELLO e non solo del Trainer: anche un
        chiamante esterno che usi `backward()` direttamente riceve gradienti
        nulli per i pesi congelati.
        """
        if self.lora_rank <= 0:
            return gradients
        trainable_set = set(self._trainable_indices_cached())
        return [
            g if i in trainable_set else np.zeros_like(np.asarray(g))
            for i, g in enumerate(gradients)
        ]

    def _trainable_indices_cached(self):
        return getattr(self, "_trainable_idx_cache", None)

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
        # Ogni TransformerBlock: 16 parametri base + 2 per adapter LoRA
        block_size = 16 + 2 * self.num_lora_tasks
        for layer in self.layers:
            layer.set_parameters(parameters[index:index + block_size])
            index += block_size

        self.norm.set_parameters(parameters[index:index + 2])
        index += 2

        self.output_layer = np.asarray(parameters[index], dtype=self.dtype)
        self.output_bias = np.asarray(parameters[index + 1], dtype=self.dtype)

    def trainable_parameters(self):
        """
        Restituisce la lista dei parametri ADDESTRABILI (non congelati).

        Con LoRA attivo, i parametri congelati sono:
            - embedding.weights
            - tutti i W_q/W_k/W_v/W_o, W1, W2, b1, b2 dei layer
            - norm1/norm2 gamma/beta, norm gamma/beta
            - output_layer, output_bias
            - gli adapter LoRA dei task PRECEDENTI a quello attivo

        Gli unici parametri addestrabili sono le matrici A/B dell'adapter
        del task attualmente attivo.
        """
        if self.lora_rank <= 0 or not self.layers:
            return list(self.get_parameters())
        task = self._normalize_task(self._active_task)
        trainable = []
        for layer in self.layers:
            adapter = layer.ff.lora_stack[task]
            trainable.extend(adapter.parameters())
        return trainable

    def trainable_parameter_count(self):
        """
        Numero di parametri addestrabili.

        Con LoRA attivo: solo le matrici A/B dell'adapter del task attivo.
        Senza LoRA: tutti i parametri (comportamento legacy full-parameter).
        """
        if self.lora_rank <= 0 or not self.layers:
            return self.parameter_count()
        return int(sum(np.asarray(p).size for p in self.trainable_parameters()))

    def frozen_parameter_count(self):
        """Numero di parametri congelati (pre-addestrati + adapter precedenti)."""
        return self.parameter_count() - self.trainable_parameter_count()

    def lora_parameters(self):
        """Lista delle sole matrici LoRA A/B di tutti i layer."""
        return self.trainable_parameters()

    # ------------------------------------------------------------------
    # Gestione degli adapter per-task (design option (b))
    # ------------------------------------------------------------------

    @property
    def num_lora_tasks(self):
        """Numero di task registrati (uguale al numero di adapter per layer)."""
        if not self.layers or not self.layers[0].ff.lora_stack:
            return 0
        return self.layers[0].ff.num_adapters

    @property
    def active_task(self):
        """Indice del task attualmente addestrabile (None se LoRA disattivata)."""
        return self._active_task

    def _normalize_task(self, task):
        n = self.num_lora_tasks
        if task is None:
            return n - 1
        task = int(task)
        if not 0 <= task < n:
            raise ValueError(f"task {task} non valido: dev'essere in [0, {n - 1}]")
        return task

    def add_task_adapter(self, task=None, warm_start=None, seed=None):
        """
        Registra un nuovo task e attiva il suo adapter LoRA dedicato.

        Crea un adapter per-task in ogni TransformerBlock. L'adapter
        precedente resta nello stack e nel forward pass ma non e' piu'
        addestrabile. Il forward somma le correzioni di tutti gli adapter,
        quindi il modello deployed resta un oggetto unico.

        Args:
            task: indice del task da attivare. Se `None`, registra un task
                in coda (incrementa il contatore).
            warm_start: se `True` (default), il nuovo adapter parte dai pesi
                dell'adapter precedente. E' l'analogo, nello spazio LoRA,
                dell'assunzione theta_{k+1} = theta_k dell'EWC full-parameter,
                ed e' la condizione che rende il Micro-EWC significativo.
            seed: seed opzionale per l'inizializzazione a freddo.

        Returns:
            L'indice del task attivo.
        """
        if self.lora_rank <= 0:
            raise ValueError("add_task_adapter() richiede lora_rank > 0")
        if task is None:
            task = self.num_lora_tasks
        else:
            # Deve essere successivo all'ultimo registrato: gli adapter sono
            # append-only, per non invalidare i pesi gia' addestrati.
            if int(task) < self.num_lora_tasks:
                raise ValueError(
                    f"task {task} gia' registrato: gli adapter sono append-only"
                )
        for i, layer in enumerate(self.layers):
            layer.ff.add_adapter(
                warm_start=warm_start,
                seed=None if seed is None else int(seed) + i,
            )
        self._active_task = self.num_lora_tasks - 1
        self._invalidate_trainable_cache()
        return self._active_task

    def set_active_task(self, task):
        """
        Attiva un task gia' registrato (per addestrare o valutare su quel task).

        Solo l'adapter del task attivo riceve gradienti e viene aggiornato.
        """
        self._active_task = self._normalize_task(task)
        self._invalidate_trainable_cache()
        return self._active_task

    def _invalidate_trainable_cache(self):
        if hasattr(self, "_trainable_idx_cache"):
            del self._trainable_idx_cache

    def _trainable_indices_cached(self):
        cache = getattr(self, "_trainable_idx_cache", None)
        if cache is None:
            cache = self.trainable_param_indices()
            self._trainable_idx_cache = cache
        return cache

    def named_parameters(self):
        """
        Mappa nome -> array per ogni parametro del modello.

        I nomi sono deterministici e descrittivi, nello stesso ordine di
        `get_parameters()`:
            - embedding
            - layer{i}_norm1_gamma, layer{i}_norm1_beta
            - layer{i}_W_q, layer{i}_b_q, ..., layer{i}_b_o
            - layer{i}_norm2_gamma, layer{i}_norm2_beta
            - layer{i}_W1, layer{i}_b1, layer{i}_W2, layer{i}_b2
            - layer{i}_lora_{task}_{A,B}   (un blocco per task registrato)
            - norm_gamma, norm_beta, output_layer, output_bias
        """
        names = ["embedding"]
        for i, layer in enumerate(self.layers):
            names.extend([
                f"layer{i}_norm1_gamma", f"layer{i}_norm1_beta",
                f"layer{i}_W_q", f"layer{i}_b_q",
                f"layer{i}_W_k", f"layer{i}_b_k",
                f"layer{i}_W_v", f"layer{i}_b_v",
                f"layer{i}_W_o", f"layer{i}_b_o",
                f"layer{i}_norm2_gamma", f"layer{i}_norm2_beta",
                f"layer{i}_W1", f"layer{i}_b1",
                f"layer{i}_W2", f"layer{i}_b2",
            ])
            for t, adapter in enumerate(layer.ff.lora_stack):
                names.extend([f"layer{i}_lora_{t}_A", f"layer{i}_lora_{t}_B"])
        names.extend(["norm_gamma", "norm_beta", "output_layer", "output_bias"])
        params = self.get_parameters()
        return {n: p for n, p in zip(names, params)}

    def trainable_param_names(self):
        """
        Nomi dei parametri addestrabili: le matrici A/B del SOLO adapter
        del task attivo. Gli adapter dei task precedenti sono nello stack e
        nel forward pass, ma congelati.
        """
        if self.lora_rank <= 0 or not self.layers:
            return []
        task = self._normalize_task(self._active_task)
        return [f"layer{i}_lora_{task}_{ab}"
                for i in range(self.num_layers) for ab in ("A", "B")]

    def trainable_param_indices(self):
        """
        Indici in `get_parameters()` dei parametri addestrabili.

        Con LoRA attivo: solo le matriche A/B dell'adapter del task attivo.
        Senza LoRA (`lora_rank == 0`): tutti gli indici, cioe' comportamento
        full-parameter legacy, identico a prima dell'introduzione di LoRA.
        """
        if self.lora_rank <= 0 or not self.layers:
            return list(range(len(self.get_parameters())))

        n_tasks = self.num_lora_tasks
        task = self._normalize_task(self._active_task)
        block_size = 16 + 2 * n_tasks
        indices = []
        for i in range(self.num_layers):
            base = 1 + i * block_size
            offset = 16 + 2 * task
            indices.extend([base + offset, base + offset + 1])
        return indices

    def frozen_param_names(self):
        """Nomi dei parametri congelati (vuoto se LoRA non attivo)."""
        if self.lora_rank <= 0:
            return []
        all_names = list(self.named_parameters().keys())
        trainable = set(self.trainable_param_names())
        return [n for n in all_names if n not in trainable]

    def parameter_count(self):
        """Numero totale di parametri (congelati + addestrabili)."""
        return int(sum(np.asarray(p).size for p in self.get_parameters()))

    def parameter_count_breakdown(self):
        total = self.parameter_count()
        embedding = int(self.embedding.weights.size)
        attention = 0
        ffn = 0
        ffn_lora = 0
        layernorm = 0
        output = int(self.output_layer.size + self.output_bias.size)

        for layer in self.layers:
            attention += int(sum(np.asarray(p).size for p in layer.attn.parameters()))
            # FFN base (W1, b1, W2, b2) esclusa la correzione LoRA
            ffn += int(layer.ff.W1.size + layer.ff.b1.size
                       + layer.ff.W2.size + layer.ff.b2.size)
            if layer.ff.lora is not None:
                ffn_lora += int(layer.ff.lora.parameter_count())
            layernorm += int(sum(np.asarray(p).size for p in layer.norm1.parameters()))
            layernorm += int(sum(np.asarray(p).size for p in layer.norm2.parameters()))

        layernorm += int(sum(np.asarray(p).size for p in self.norm.parameters()))

        return {
            "total": total,
            "trainable": self.trainable_parameter_count(),
            "non_trainable": self.frozen_parameter_count(),
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "ffn_lora": ffn_lora,
            "layernorm": layernorm,
            "output": output,
        }

    def attention_weights(self, layer_index=0):
        """Pesi di attenzione di un layer dopo l'ultima forward."""
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("layer_index fuori intervallo.")
        return self.layers[layer_index].attn.attention_weights()
