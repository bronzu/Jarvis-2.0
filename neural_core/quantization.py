"""
Quantizzazione INT8 per il Neural Core di JARVIS.

Implementa Symmetric Absmax Quantization per i pesi dei layer lineari
(Dense/Linear), permettendo l'inferenza con pesi a 8 bit con riduzione
della memoria del 75% (FP32 → INT8) e velocizzando l'operazione
matriciale su hardware con VRAM limitata.

Formula di quantizzazione:
    S = max(|W|) / 127
    W_q = round(W / S)  →  cast a np.int8
    W_dq = W_q * S      (dequantizzazione)

Classi:
    QuantizedLinear   — layer lineare con quantizzazione integrata
    QuantizedModel    — wrapper per gestire un modello interamente quantizzato

Funzioni standalone:
    quantize_int8_symmetric   — quantizza un array NumPy a int8
    dequantize_int8           — ricostruisce float da int8 + scale
    load_weights              — carica pesi FP32 o INT8 in base alla config
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

#: Valore massimo rappresentabile in int8 signed.
_INT8_MAX = 127


# ----------------------------------------------------------------------
# Funzioni standalone di quantizzazione
# ----------------------------------------------------------------------

def quantize_int8_symmetric(
    weights: np.ndarray,
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric absmax quantization per un array di pesi.

    Args:
        weights: array di pesi (float32 o float64).
        axis: se ``None`` usa scala per-tensore (una sola S per l'intero array).
              Se specificato, quantizza per-canale lungo quell'asse
              (es. ``axis=1`` per (d_in, d_out) → una scala per ogni colonna).

    Returns:
        (quantized_int8, scale) dove ``scale`` ha la stessa forma di
        ``weights`` se per-canale, o è uno scalare se per-tensore.
    """
    weights = np.asarray(weights, dtype=np.float64)

    if axis is not None:
        max_abs = np.max(np.abs(weights), axis=axis, keepdims=True)
    else:
        max_abs = np.max(np.abs(weights))

    # Evita divisione per zero: se tutti i pesi sono 0, scale = 1
    scale = max_abs / float(_INT8_MAX)
    if np.isscalar(scale):
        scale = max(scale, 1e-8)
    else:
        scale = np.where(scale == 0, 1e-8, scale)

    # Arrotondamento e clipping all'intervallo int8
    quantized = np.round(weights / scale)
    quantized = np.clip(quantized, -_INT8_MAX, _INT8_MAX)
    return quantized.astype(np.int8), scale


def dequantize_int8(
    quantized: np.ndarray,
    scale: Union[float, np.ndarray],
) -> np.ndarray:
    """Ricostruisce i pesi float dall'array int8 e dallo scale."""
    return quantized.astype(np.float64) * np.asarray(scale, dtype=np.float64)


def quantization_error(
    weights: np.ndarray,
    quantized: np.ndarray,
    scale: Union[float, np.ndarray],
) -> float:
    """ERRORE QUADRATICO MEDIO (MSE) tra i pesi originali e dequantizzati."""
    dequant = dequantize_int8(quantized, scale)
    return float(np.mean((weights - dequant) ** 2))


# ----------------------------------------------------------------------
# QuantizedLinear — layer lineare con quantizzazione int8
# ----------------------------------------------------------------------

class QuantizedLinear:
    """
    Layer lineare con pesi quantizzati in INT8 (symmetric absmax).

    In fase di *forward*, i pesi vengono dequantizzati al volo:
    ``output = X @ (W_q * S) + b``. La dequantizzazione è O(d_in * d_out)
    e l'operazione matriciale usa i valori float ricostruiti.

    La *backward* utilizza lo **Straight-Through Estimator (STE)**:
    il gradiente w.r.t. i pesi scorre come se la quantizzazione
    fosse un'operazione identità, permettendo il training aware.

    Interfaccia compatibile con :class:`FeedForward` e
    :class:`MultiHeadAttention` del Neural Core:
        forward(x), backward(grad), parameters(), gradients(),
        set_parameters(), state_dict(), load_state_dict()

    Shapes:
        Input  x:  (batch, ..., d_in)
        Weight W:  (d_in, d_out)
        Output:    (batch, ..., d_out)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        per_channel: bool = True,
        seed=None,
        dtype=np.float64,
    ):
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in e d_out devono essere maggiori di 0")

        self.d_in = d_in
        self.d_out = d_out
        self.per_channel = per_channel
        self.dtype = np.dtype(dtype)

        rng = np.random.RandomState(seed) if seed is not None else np.random
        init_scale = 1.0 / np.sqrt(d_in)

        # Pesi float di riferimento (FP32/FP64)
        self.W = (rng.randn(d_in, d_out) * init_scale).astype(self.dtype)
        self.b = np.zeros(d_out, dtype=self.dtype)

        # Quantizzazione iniziale
        self._quantize()

        # Gradienti (STE: scorrono come float)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self._cached = {}

    # ------------------------------------------------------------------
    # Quantizzazione / Dequantizzazione
    # ------------------------------------------------------------------

    def _quantize(self):
        """Quantizza i pesi attuali in INT8."""
        axis = 1 if self.per_channel else None
        self.W_q, self.scale = quantize_int8_symmetric(self.W, axis=axis)

    def dequantize(self) -> np.ndarray:
        """Restituisce i pesi dequantizzati (float64)."""
        return dequantize_int8(self.W_q, self.scale)

    def quantize_weights(self, weights: np.ndarray):
        """Quantizza un array di pesi esterno e lo imposta come W."""
        self.W = np.asarray(weights, dtype=self.dtype)
        self._quantize()

    @property
    def weight_bits(self) -> int:
        """Numero di bit per peso: sempre 8."""
        return 8

    @property
    def weight_bytes_per_element(self) -> int:
        """Byte occupati per peso quantizzato: 1 (int8)."""
        return 1

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward usando i pesi float ``self.W``.

        Durante il training i gradienti scorrono sui pesi float originali
        (Straight-Through Estimator): la quantizzazione è trasparente per
        il gradiente.  Per l'inferenza con INT8 usare
        :meth:`quantized_forward`.

        Args:
            x: tensore input ``(batch, ..., d_in)``.

        Returns:
            Output ``(batch, ..., d_out)`` — stessa shape tranne l'ultima.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"Input d_in ({x.shape[-1]}) != weight d_in ({self.d_in})."
            )

        shape = x.shape
        x_flat = x.reshape(-1, self.d_in)
        output_flat = x_flat @ self.W + self.b
        self._cached = {
            'x_flat': x_flat,
            'shape': shape,
            'W': self.W.copy(),
        }
        return output_flat.reshape(*shape[:-1], self.d_out)

    def quantized_forward(self, x):
        """Forward con dequantizzazione al volo (INT8 inference).

        Usa ``W_q * scale`` al posto di ``self.W``, simulando
        l'inferenza su hardware con pesi a 8 bit.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"Input d_in ({x.shape[-1]}) != weight d_in ({self.d_in})."
            )

        W_dq = self.dequantize()  # (d_in, d_out) — float64
        shape = x.shape
        x_flat = x.reshape(-1, self.d_in)
        output_flat = x_flat @ W_dq + self.b
        return output_flat.reshape(*shape[:-1], self.d_out)

    def backward(self, grad_output):
        """Backward con Straight-Through Estimator (STE).

        Il gradiente w.r.t. W scorre attraverso W_dq (float), ma il
        gradiente non aggiorna W_q. L'update dei pesi avviene in float
        e viene ribilanciata in INT8 a intervalli.
        """
        if not self._cached:
            raise RuntimeError("backward() richiede una forward() precedente.")

        grad_output = np.asarray(grad_output, dtype=self.dtype)
        cache = self._cached
        shape = cache['shape']

        grad_flat = grad_output.reshape(-1, self.d_out)
        x_flat = cache['x_flat']

        # Gradienti (STE: come se usassimo W direttamente)
        self.grad_W = x_flat.T @ grad_flat
        self.grad_b = np.sum(grad_flat, axis=0)

        # Gradiente w.r.t. input — usa i pesi float usati in forward
        grad_x = (grad_flat @ cache['W'].T).reshape(*shape[:-1], self.d_in)
        return grad_x

    # ------------------------------------------------------------------
    # Interfaccia parametri
    # ------------------------------------------------------------------

    def parameters(self) -> List[np.ndarray]:
        """Restituisce i pesi float (per compatibilità con optimizer)."""
        return [self.W, self.b]

    def get_parameters(self) -> List[np.ndarray]:
        return self.parameters()

    def gradients(self) -> List[np.ndarray]:
        return [self.grad_W, self.grad_b]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Imposta i pesi float e ricalcola la quantizzazione."""
        if len(parameters) != 2:
            raise ValueError("QuantizedLinear richiede 2 parametri (W, b)")
        self.W = np.asarray(parameters[0], dtype=self.dtype)
        self.b = np.asarray(parameters[1], dtype=self.dtype)
        self._quantize()

    # ------------------------------------------------------------------
    # Serializzazione / Caricamento
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Serializza i parametri quantizzati."""
        original_bytes = self.W.nbytes + self.b.nbytes
        quantized_bytes = self.W_q.nbytes + (
            self.scale.nbytes if hasattr(self.scale, 'nbytes') else 8
        )
        return {
            'format': 'int8',
            'd_in': self.d_in,
            'd_out': self.d_out,
            'per_channel': self.per_channel,
            'W_q': self.W_q.tolist(),
            'scale': (self.scale.tolist() if hasattr(self.scale, 'tolist') else float(self.scale)),
            'b': self.b.tolist(),
            'dtype': str(self.dtype),
            'weight_bytes': self.W_q.nbytes,
            'compression_ratio': original_bytes / max(quantized_bytes, 1),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Carica i parametri da un dict (INT8 o FP32)."""
        self.d_in = state['d_in']
        self.d_out = state['d_out']
        self.per_channel = state.get('per_channel', True)
        self.dtype = np.dtype(state.get('dtype', str(self.dtype)))

        # Rileva se i pesi sono INT8 (chiave 'W_q' + 'scale') oppure FP32 ('W')
        if 'W_q' in state and 'scale' in state:
            # Caricamento INT8
            self.W_q = np.array(state['W_q'], dtype=np.int8)
            scale_data = state['scale']
            if isinstance(scale_data, list):
                self.scale = np.array(scale_data, dtype=np.float64)
            else:
                self.scale = float(scale_data)
            self.W = self.dequantize().astype(self.dtype)
        elif 'W' in state:
            # Caricamento FP32/FP64 — quantizza adesso
            self.W = np.array(state['W'], dtype=self.dtype)
            self._quantize()
        else:
            raise ValueError("state_dict deve contenere 'W_q'+'scale' (INT8) o 'W' (FP)")

        if 'b' in state:
            self.b = np.array(state['b'], dtype=self.dtype)
        else:
            self.b = np.zeros(self.d_out, dtype=self.dtype)

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self._cached = {}

    def __repr__(self) -> str:
        return (
            f"QuantizedLinear(d_in={self.d_in}, d_out={self.d_out}, "
            f"per_channel={self.per_channel}, bits={self.weight_bits})"
        )


# ----------------------------------------------------------------------
# Utility: caricamento pesi con rilevamento formato
# ----------------------------------------------------------------------

def load_weights(path: str) -> Dict[str, Any]:
    """
    Carica i pesi da disco. Rileva automaticamente se sono:

    - **INT8 quantizzati**: chiave ``'format'`` = ``'int8'``, contiene
      ``W_q`` (int8) e ``scale`` (float o array).
    - **FP32/FP64 standard**: chiave ``'format'`` = ``'fp32'`` (o mancante),
      contiene ``W`` (float).

    Returns:
        Dizionario di stato pronto per ``QuantizedLinear.load_state_dict()``.
    """
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    fmt = state.get('format', 'fp32')
    state['format'] = fmt

    if fmt == 'int8':
        # I pesi sono già quantizzati — nessuna dequantizzazione necessaria
        # finché non serve per l'inferenza.
        return state
    elif fmt == 'fp32':
        return state
    else:
        raise ValueError(f"Formato pesi sconosciuto: {fmt}")


def save_quantized_model(model_or_layers: List[QuantizedLinear], path: str) -> None:
    """
    Salva una lista di layer QuantizedLinear in formato INT8 su disco.

    Ogni layer è serializzato con ``state_dict()`` e il risultato
    viene scritto come JSON con ``format: 'int8'``.
    """
    layers_state = []
    total_original = 0
    total_quantized = 0

    for layer in model_or_layers:
        state = layer.state_dict()
        state['format'] = 'int8'
        state['dtype'] = str(layer.dtype)
        layers_state.append(state)
        total_original += layer.W.nbytes + layer.b.nbytes
        total_quantized += layer.W_q.nbytes + (
            layer.scale.nbytes if hasattr(layer.scale, 'nbytes') else 8
        )

    output = {
        'format': 'int8',
        'layers': layers_state,
        'compression': {
            'original_bytes': total_original,
            'quantized_bytes': total_quantized,
            'ratio': total_original / max(total_quantized, 1),
        },
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)


def build_quantized_model(layers_config: List[Dict[str, Any]]) -> List[QuantizedLinear]:
    """
    Costruisce una lista di QuantizedLinear a partire da una configurazione.

    Args:
        layers_config: lista di dict con chiavi 'd_in', 'd_out', 'seed', ...

    Returns:
        Lista di QuantizedLinear inizializzati e quantizzati.
    """
    layers = []
    for cfg in layers_config:
        layer = QuantizedLinear(
            d_in=cfg['d_in'],
            d_out=cfg['d_out'],
            per_channel=cfg.get('per_channel', True),
            seed=cfg.get('seed', None),
            dtype=cfg.get('dtype', np.float64),
        )
        layers.append(layer)
    return layers


if __name__ == '__main__':
    # Demo: quantizza un layer e mostra le metriche
    layer = QuantizedLinear(d_in=64, d_out=128, per_channel=True, seed=42)
    err = quantization_error(layer.W, layer.W_q, layer.scale)
    print(f"QuantizedLinear(64→128, per-channel)")
    print(f"  Original size:  {layer.W.nbytes} bytes (FP64)")
    print(f"  Quantized size: {layer.W_q.nbytes + layer.scale.nbytes} bytes (INT8 + scale)")
    print(f"  Compression:    {layer.W.nbytes / (layer.W_q.nbytes + layer.scale.nbytes):.2f}x")
    print(f"  MSE:            {err:.6e}")
    print(f"  Max abs error:  {float(np.max(np.abs(layer.W - layer.dequantize()))):.6e}")
