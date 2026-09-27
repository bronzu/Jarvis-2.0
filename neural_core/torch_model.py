"""
GPU backend model for JARVIS Neural Core (Phase 2A).

This module implements a PyTorch/MPS port of the CPU reference transformer
``neural_core.network.MicroTransformer``. It is intentionally a SEPARATE,
ISOLATED implementation: the NumPy CPU reference (``network.py``) is never
modified and remains the authoritative baseline (float64).

Design constraints (Phase 2A authorization):
    - Same architecture, same math, same parameter ordering.
    - GPU runs at float32 (MPS has no float64 support on Apple Silicon).
    - GELU uses the SAME tanh approximation as the NumPy reference
      (NOT PyTorch's default erf GELU) to preserve the model math.
    - LayerNorm is computed manually with eps=1e-6 to match the NumPy
      population-variance formula exactly.
    - No model-scaling: dimensions are taken from the supplied config.

Numerical parity (CPU float64 vs GPU float32) is validated by the parity
test suite; only documented tolerances are expected, never bitwise identity.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with the exact tanh approximation used by the NumPy reference."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _manual_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
    """LayerNorm matching ``neural_core.network.LayerNorm`` (population var)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


def _pos_encoding(d_model: int, max_len: int, dtype, device):
    """Sinusoidal positional encoding identical to ``PositionalEncoding``."""
    pe = torch.zeros((max_len, d_model), dtype=dtype, device=device)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    cos_columns = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term[:cos_columns])
    return pe


class TorchMicroTransformer(torch.nn.Module):
    """
    PyTorch/MPS port of ``MicroTransformer``.

    Parameter ordering is identical to ``MicroTransformer.get_parameters`` so
    that weights can be transferred losslessly (float64 -> float32) from the
    CPU reference for parity testing.
    """

    def __init__(self, vocab_size, d_model=64, num_heads=2, d_ff=128, num_layers=2,
                 max_len=256, dropout=0.0, seed=None, dtype=torch.float32,
                 device="cpu", lora_rank: int = 0, lora_alpha: float = 16.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if lora_rank < 0:
            raise ValueError("lora_rank must be >= 0")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_k = self.d_model // self.num_heads
        self.d_ff = int(d_ff)
        self.num_layers = int(num_layers)
        self.max_len = int(max_len)
        self.dropout = float(dropout)
        self.dtype = dtype
        self.device = torch.device(device)
        self.embedding_multiplier = math.sqrt(self.d_model)
        self.eps = 1e-6
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 0.0
        # Task attivamente addestrabile sugli adapter LoRA.
        self._active_task = 0 if lora_rank > 0 else None

        self.embedding = torch.nn.Parameter(
            torch.zeros(self.vocab_size, self.d_model, dtype=self.dtype, device=self.device))

        self.pos_encoding = _pos_encoding(self.d_model, self.max_len, self.dtype, self.device)
        self.register_buffer("pe", self.pos_encoding, persistent=False)

        self.blocks = torch.nn.ModuleList()
        gen = torch.Generator(device="cpu").manual_seed(1234 if seed is None else int(seed))
        for block_idx in range(self.num_layers):
            block = torch.nn.ParameterDict({
                "norm1_gamma": torch.nn.Parameter(torch.ones(self.d_model, dtype=self.dtype, device=self.device)),
                "norm1_beta": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "W_q": torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, dtype=self.dtype, device=self.device)),
                "b_q": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "W_k": torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, dtype=self.dtype, device=self.device)),
                "b_k": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "W_v": torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, dtype=self.dtype, device=self.device)),
                "b_v": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "W_o": torch.nn.Parameter(torch.zeros(self.d_model, self.d_model, dtype=self.dtype, device=self.device)),
                "b_o": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "norm2_gamma": torch.nn.Parameter(torch.ones(self.d_model, dtype=self.dtype, device=self.device)),
                "norm2_beta": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
                "W1": torch.nn.Parameter(torch.zeros(self.d_model, self.d_ff, dtype=self.dtype, device=self.device)),
                "b1": torch.nn.Parameter(torch.zeros(self.d_ff, dtype=self.dtype, device=self.device)),
                "W2": torch.nn.Parameter(torch.zeros(self.d_ff, self.d_model, dtype=self.dtype, device=self.device)),
                "b2": torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device)),
            })
            if self.lora_rank > 0:
                # Blocco adapter per-task 0. A: (rank, d_ff) Kaiming uniform;
                # B: (d_model, rank) zero. Stessa convenzione della reference NumPy.
                self._new_lora_params(gen, block, task=0)
            self.blocks.append(block)

        self.norm_gamma = torch.nn.Parameter(torch.ones(self.d_model, dtype=self.dtype, device=self.device))
        self.norm_beta = torch.nn.Parameter(torch.zeros(self.d_model, dtype=self.dtype, device=self.device))
        self.output_layer = torch.nn.Parameter(
            torch.zeros(self.d_model, self.vocab_size, dtype=self.dtype, device=self.device))
        self.output_bias = torch.nn.Parameter(torch.zeros(self.vocab_size, dtype=self.dtype, device=self.device))

        self._set_trainable_flags()

        self._mask_cache = {}

    # ------------------------------------------------------------------
    # LoRA: blocco adapter per-task (parita' con MicroTransformer)
    # ------------------------------------------------------------------

    def _set_trainable_flags(self):
        """
        Congelamento selettivo: requires_grad=True SOLO sulle matrici A/B
        dell'adapter del task ATTIVAMENTE addestrato. Tutto il resto
        (embedding, attention, LayerNorm, W1/b1/W2/b2, output, e gli adapter
        dei task precedenti) resta in requires_grad=False, cosi' l'update
        Micro-EWC non puo' toccarli.

        Va eseguito DOPO aver creato tutti i parametri. Si usa esplicitamente
        nn.Module.named_parameters per non passare per metodi sovrascritti.
        """
        if self.lora_rank <= 0:
            for p in torch.nn.Module.parameters(self):
                p.requires_grad_(True)
            return
        active = self._normalize_task(self._active_task)
        for name, p in torch.nn.Module.named_parameters(self):
            p.requires_grad_(name.endswith(f"lora_{active}_A")
                             or name.endswith(f"lora_{active}_B"))

    def _new_lora_params(self, gen, block, task, warm_start_block=None):
        """Crea (o clona) le matrici LoRA di un task dentro `block`."""
        A = torch.empty(self.lora_rank, self.d_ff, dtype=self.dtype, device=self.device)
        bound = math.sqrt(6.0 / (self.lora_rank + self.d_ff))
        A.uniform_(-bound, bound, generator=gen)
        B = torch.zeros(self.d_model, self.lora_rank, dtype=self.dtype, device=self.device)
        if warm_start_block is not None:
            A = warm_start_block[f"lora_{task - 1}_A"].detach().clone()
            B = warm_start_block[f"lora_{task - 1}_B"].detach().clone()
        block[f"lora_{task}_A"] = torch.nn.Parameter(A)
        block[f"lora_{task}_B"] = torch.nn.Parameter(B)

    @property
    def num_lora_tasks(self):
        """Numero di task registrati (0 se LoRA disattivata)."""
        if not self.blocks or self.lora_rank <= 0:
            return 0
        n = 0
        while f"lora_{n}_A" in self.blocks[0]:
            n += 1
        return n

    @property
    def active_task(self):
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

        Gli adapter sono append-only: registrare un task gia' esistente e'
        un errore, per non invalidare pesi gia' addestrati. Il forward somma
        le correzioni di tutti gli adapter (stacking), quindi il modello
        resta un oggetto unico.
        """
        if self.lora_rank <= 0:
            raise ValueError("add_task_adapter() richiede lora_rank > 0")
        current = self.num_lora_tasks
        if task is None:
            task = current
        elif int(task) < current:
            raise ValueError(f"task {task} gia' registrato: gli adapter sono append-only")
        gen = torch.Generator(device="cpu").manual_seed(
            (1234 if seed is None else int(seed)) + 7919 * current
        )
        # `warm_start=None` deve valere `True`, come nella reference NumPy:
        # il nuovo adapter riparte dai pesi dell'ultimo registrato.
        if warm_start is None:
            warm_start = True
        for block in self.blocks:
            self._new_lora_params(
                gen, block, task=current,
                warm_start_block=block if warm_start else None,
            )
        self._active_task = self.num_lora_tasks - 1
        self._set_trainable_flags()
        return self._active_task

    def set_active_task(self, task):
        self._active_task = self._normalize_task(task)
        self._set_trainable_flags()
        return self._active_task

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "max_len": self.max_len,
            "dropout": self.dropout,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }

    @property
    def context_size(self):
        return self.max_len

    # ------------------------------------------------------------------
    # Parameter transfer (CPU float64 -> GPU float32)
    # ------------------------------------------------------------------
    def set_parameters(self, parameters):
        """Load parameters from a list ordered like ``MicroTransformer.get_parameters``."""
        parameters = list(parameters)
        block_size = 16 + 2 * self.num_lora_tasks
        expected = 1 + self.num_layers * block_size + 2 + 2
        if len(parameters) != expected:
            raise ValueError(f"Expected {expected} parameter arrays, got {len(parameters)}")

        with torch.no_grad():
            self.embedding.copy_(torch.as_tensor(parameters[0], dtype=self.dtype, device=self.device))
            idx = 1
            for block in self.blocks:
                for key in self._block_keys():
                    block[key].copy_(torch.as_tensor(parameters[idx], dtype=self.dtype, device=self.device))
                    idx += 1
            self.norm_gamma.copy_(torch.as_tensor(parameters[idx], dtype=self.dtype, device=self.device)); idx += 1
            self.norm_beta.copy_(torch.as_tensor(parameters[idx], dtype=self.dtype, device=self.device)); idx += 1
            self.output_layer.copy_(torch.as_tensor(parameters[idx], dtype=self.dtype, device=self.device)); idx += 1
            self.output_bias.copy_(torch.as_tensor(parameters[idx], dtype=self.dtype, device=self.device)); idx += 1

    def _block_keys(self):
        """Chiavi di un blocco nell'ordine canonico di `get_parameters`."""
        keys = [
            "norm1_gamma", "norm1_beta",
            "W_q", "b_q", "W_k", "b_k", "W_v", "b_v", "W_o", "b_o",
            "norm2_gamma", "norm2_beta",
            "W1", "b1", "W2", "b2",
        ]
        for t in range(self.num_lora_tasks):
            keys.extend([f"lora_{t}_A", f"lora_{t}_B"])
        return keys

    def get_parameters(self):
        """Return parameters as a list of numpy arrays in reference order."""
        out = [self.embedding.detach().cpu().numpy()]
        for block in self.blocks:
            for key in self._block_keys():
                out.append(block[key].detach().cpu().numpy())
        out.extend([
            self.norm_gamma.detach().cpu().numpy(),
            self.norm_beta.detach().cpu().numpy(),
            self.output_layer.detach().cpu().numpy(),
            self.output_bias.detach().cpu().numpy(),
        ])
        return out

    def canonical_parameter_names(self):
        """
        Nomi canonici dei parametri, nello stesso ordine di `get_parameters()`.

        Non sovrascrive `nn.Module.named_parameters` per non rompere l'internals
        di PyTorch: il metodo ha quindi un nome diverso.
        """
        out = ["embedding"]
        for i, block in enumerate(self.blocks):
            out.extend([f"layer{i}_{key}" for key in self._block_keys()])
        out.extend(["norm_gamma", "norm_beta", "output_layer", "output_bias"])
        return out

    def trainable_param_names(self):
        """Nomi dei parametri addestrabili: A/B del SOLO adapter del task attivo."""
        if self.lora_rank <= 0:
            return []
        task = self._normalize_task(self._active_task)
        return [f"layer{i}_lora_{task}_{ab}"
                for i in range(self.num_layers) for ab in ("A", "B")]

    def trainable_param_indices(self):
        """
        Indici in `get_parameters()` dei parametri addestrabili.

        Con LoRA attivo: solo A/B. Senza LoRA: tutti (comportamento legacy).
        """
        if self.lora_rank <= 0:
            return list(range(len(self.get_parameters())))
        task = self._normalize_task(self._active_task)
        indices = []
        block_size = 16 + 2 * self.num_lora_tasks
        for i in range(self.num_layers):
            base = 1 + i * block_size
            offset = 16 + 2 * task
            indices.extend([base + offset, base + offset + 1])
        return indices

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_parameter_count(self):
        """Parametri addestrabili: solo l'adapter del task attivo, o tutti."""
        if self.lora_rank <= 0:
            return self.parameter_count()
        return int(sum(np.asarray(self.get_parameters()[i]).size
                       for i in self.trainable_param_indices()))

    def frozen_parameter_count(self):
        return self.parameter_count() - self.trainable_parameter_count()

    def get_gradients(self):
        """Return gradients as numpy arrays in reference parameter order.

        NOTE: ``nn.ParameterDict`` iterates keys alphabetically, so gradients are
        gathered explicitly to match ``get_parameters`` / the NumPy reference order.
        """
        grads = [self.embedding.grad]
        for block in self.blocks:
            for key in self._block_keys():
                grads.append(block[key].grad)
        grads.extend([
            self.norm_gamma.grad,
            self.norm_beta.grad,
            self.output_layer.grad,
            self.output_bias.grad,
        ])
        return [g.detach().cpu().numpy() if g is not None else None for g in grads]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _causal_mask(self, seq_len):
        cached = self._mask_cache.get(seq_len)
        if cached is None:
            ones = torch.ones((seq_len, seq_len), dtype=self.dtype, device=self.device)
            mask = torch.triu(ones, diagonal=1)
            mask = torch.where(mask.bool(), torch.tensor(float("-inf"), dtype=self.dtype, device=self.device),
                               torch.zeros((), dtype=self.dtype, device=self.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            self._mask_cache[seq_len] = mask
        return self._mask_cache[seq_len]

    def _split_heads(self, x, batch_size, seq_len):
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x, batch_size, seq_len):
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

    def forward(self, token_ids, training=False):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.reshape(1, 1)
        elif ids.ndim == 1:
            ids = ids.reshape(1, -1)
        elif ids.ndim != 2:
            raise ValueError("token_ids must have shape (seq_len,) or (batch, seq_len).")
        if ids.shape[1] == 0:
            raise ValueError("Input sequence cannot be empty.")
        if int(ids.shape[1]) > self.max_len:
            raise ValueError("Sequence exceeds context window.")

        batch_size, seq_len = ids.shape
        mask = self._causal_mask(int(seq_len))

        x = F.embedding(ids, self.embedding) * self.embedding_multiplier
        x = x + self.pe[:seq_len]

        for block in self.blocks:
            # pre-norm attention
            h_norm = _manual_layernorm(x, block["norm1_gamma"], block["norm1_beta"], self.eps)
            q = (h_norm.reshape(-1, self.d_model) @ block["W_q"] + block["b_q"]).reshape(batch_size, seq_len, self.d_model)
            k = (h_norm.reshape(-1, self.d_model) @ block["W_k"] + block["b_k"]).reshape(batch_size, seq_len, self.d_model)
            v = (h_norm.reshape(-1, self.d_model) @ block["W_v"] + block["b_v"]).reshape(batch_size, seq_len, self.d_model)

            q_heads = self._split_heads(q, batch_size, seq_len)
            k_heads = self._split_heads(k, batch_size, seq_len)
            v_heads = self._split_heads(v, batch_size, seq_len)

            scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores + mask
            attention = F.softmax(scores, dim=-1)
            context = attention @ v_heads
            merged = self._merge_heads(context, batch_size, seq_len)
            attn_out = (merged.reshape(-1, self.d_model) @ block["W_o"] + block["b_o"]).reshape(batch_size, seq_len, self.d_model)
            if training and self.dropout > 0:
                attn_out = F.dropout(attn_out, self.dropout, training=True)
            hidden = x + attn_out

            # pre-norm feed-forward
            h2_norm = _manual_layernorm(hidden, block["norm2_gamma"], block["norm2_beta"], self.eps)
            z1 = h2_norm.reshape(-1, self.d_model) @ block["W1"] + block["b1"]
            h1 = _gelu_tanh(z1)
            ff_out = (h1 @ block["W2"] + block["b2"]).reshape(batch_size, seq_len, self.d_model)
            if self.lora_rank > 0:
                # Correzione LoRA: scaling * (B @ A) @ h1  (torch.matmul)
                for t in range(self.num_lora_tasks):
                    lora_delta = torch.matmul(
                        torch.matmul(h1, block[f"lora_{t}_A"].T), block[f"lora_{t}_B"].T)
                    ff_out = ff_out + self.lora_scaling * lora_delta.reshape(
                        batch_size, seq_len, self.d_model)

            if training and self.dropout > 0:
                ff_out = F.dropout(ff_out, self.dropout, training=True)
            x = hidden + ff_out

        x = _manual_layernorm(x, self.norm_gamma, self.norm_beta, self.eps)
        x_flat = x.reshape(-1, self.d_model)
        logits = (x_flat @ self.output_layer + self.output_bias).reshape(batch_size, seq_len, self.vocab_size)
        return logits

    def loss_value(self, logits, targets, ignore_index=-100):
        """Cross-entropy value (mean over valid targets), mirrors ``CrossEntropyLoss``."""
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_targets = torch.as_tensor(np.asarray(targets).reshape(-1), dtype=torch.long, device=self.device)
        return float(F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean").detach())

    @torch.no_grad()
    def forward_components(self, token_ids):
        """
        Forward returning named intermediate activations (CPU-float32 numpy).

        Mirrors the component decomposition used by the NumPy reference for
        deep parity tests: embedding, per-layer norm1/attn/norm2/ff, final_norm,
        and logits.
        """
        ids = torch.as_tensor(np.asarray(token_ids), dtype=torch.long, device=self.device)
        if ids.ndim == 0:
            ids = ids.reshape(1, 1)
        elif ids.ndim == 1:
            ids = ids.reshape(1, -1)
        elif ids.ndim != 2:
            raise ValueError("token_ids must have shape (seq_len,) or (batch, seq_len).")
        batch_size, seq_len = ids.shape
        mask = self._causal_mask(int(seq_len))

        x = F.embedding(ids, self.embedding) * self.embedding_multiplier
        x = x + self.pe[:seq_len]
        comps = {"embedding": x.clone()}

        for i, block in enumerate(self.blocks):
            n1 = _manual_layernorm(x, block["norm1_gamma"], block["norm1_beta"], self.eps)
            q = (n1.reshape(-1, self.d_model) @ block["W_q"] + block["b_q"]).reshape(batch_size, seq_len, self.d_model)
            k = (n1.reshape(-1, self.d_model) @ block["W_k"] + block["b_k"]).reshape(batch_size, seq_len, self.d_model)
            v = (n1.reshape(-1, self.d_model) @ block["W_v"] + block["b_v"]).reshape(batch_size, seq_len, self.d_model)
            q_heads = self._split_heads(q, batch_size, seq_len)
            k_heads = self._split_heads(k, batch_size, seq_len)
            v_heads = self._split_heads(v, batch_size, seq_len)
            scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores + mask
            attention = F.softmax(scores, dim=-1)
            context = attention @ v_heads
            merged = self._merge_heads(context, batch_size, seq_len)
            attn_out = (merged.reshape(-1, self.d_model) @ block["W_o"] + block["b_o"]).reshape(batch_size, seq_len, self.d_model)
            hidden = x + attn_out
            n2 = _manual_layernorm(hidden, block["norm2_gamma"], block["norm2_beta"], self.eps)
            z1 = n2.reshape(-1, self.d_model) @ block["W1"] + block["b1"]
            h1 = _gelu_tanh(z1)
            ff_out = (h1 @ block["W2"] + block["b2"]).reshape(batch_size, seq_len, self.d_model)
            if self.lora_rank > 0:
                # Correzione LoRA: scaling * (B @ A) @ h1  (torch.matmul)
                for t in range(self.num_lora_tasks):
                    lora_delta = torch.matmul(
                        torch.matmul(h1, block[f"lora_{t}_A"].T), block[f"lora_{t}_B"].T)
                    ff_out = ff_out + self.lora_scaling * lora_delta.reshape(
                        batch_size, seq_len, self.d_model)

            out = hidden + ff_out

            comps[f"layer{i}_norm1"] = n1
            comps[f"layer{i}_attn"] = attn_out
            comps[f"layer{i}_norm2"] = n2
            comps[f"layer{i}_ff"] = ff_out
            x = out

        xn = _manual_layernorm(x, self.norm_gamma, self.norm_beta, self.eps)
        x_flat = xn.reshape(-1, self.d_model)
        logits = (x_flat @ self.output_layer + self.output_bias).reshape(batch_size, seq_len, self.vocab_size)
        comps["final_norm"] = xn
        comps["logits"] = logits
        return {key: value.detach().cpu().numpy() for key, value in comps.items()}
