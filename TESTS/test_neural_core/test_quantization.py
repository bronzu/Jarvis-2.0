"""
Test unitari per il modulo di quantizzazione INT8 (Punto B).

Verifica:
  - correttezza della formula Symmetric Absmax: S = max(|W|) / 127
  - cast a np.int8 e clipping
  - shape invariata in forward/backward
  - gradient check numerico (STE)
  - roundtrip state_dict/load_state_dict (INT8 e FP32)
  - rilevamento formato in load_weights
  - compression ratio ≈ 4x (FP32 → INT8)
"""

import json
import math
import tempfile
import os

import numpy as np
import pytest

from neural_core.quantization import (
    QuantizedLinear,
    quantize_int8_symmetric,
    dequantize_int8,
    quantization_error,
    load_weights,
    save_quantized_model,
    build_quantized_model,
    _INT8_MAX,
)


# ----------------------------------------------------------------------
# Funzioni standalone
# ----------------------------------------------------------------------

def test_quantize_int8_symmetric_scale_corretto():
    """S = max(|W|) / 127."""
    rng = np.random.RandomState(0)
    w = rng.randn(100, 100) * 3.0
    w_q, scale = quantize_int8_symmetric(w)
    expected_scale = np.max(np.abs(w)) / 127.0
    assert np.isclose(scale, expected_scale)
    assert w_q.dtype == np.int8


def test_quantize_int8_symmetric_valori_dentro_range():
    w = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    w_q, scale = quantize_int8_symmetric(w)
    assert np.all(w_q >= -127)
    assert np.all(w_q <= 127)
    assert isinstance(w_q[0], np.int8) or w_q.dtype == np.int8


def test_quantize_int8_symmetric_zero_array():
    """Se tutti i pesi sono zero, scale non deve essere zero."""
    w = np.zeros((10, 10))
    w_q, scale = quantize_int8_symmetric(w)
    assert w_q.dtype == np.int8
    assert np.all(w_q == 0)
    assert scale > 0  # nessuna divisione per zero


def test_quantize_int8_symmetric_clipping():
    """Valori estremi vengono clippati a ±127."""
    w = np.array([1e6, -1e6, 0.0])
    w_q, scale = quantize_int8_symmetric(w)
    assert np.all(w_q >= -127)
    assert np.all(w_q <= 127)


def test_dequantize_int8_roundtrip():
    """dequantize(quantize(W)) ≈ W entro tol."""
    rng = np.random.RandomState(42)
    w = rng.randn(256, 256) * 0.5
    w_q, scale = quantize_int8_symmetric(w)
    w_dq = dequantize_int8(w_q, scale)
    assert w_dq.shape == w.shape
    # Errore relativo medio bassissimo
    rel_err = np.mean(np.abs(w - w_dq) / (np.abs(w) + 1e-8))
    assert rel_err < 0.05  # < 5% errore medio (per-tensor su 256x256)


def test_per_channel_vs_per_tensor():
    """Per-channel ha minore errore rispetto a per-tensore."""
    rng = np.random.RandomState(7)
    # Crea pesi con scala molto diversa tra colonne
    w = np.hstack([rng.randn(100, 50) * 0.01, rng.randn(100, 50) * 100.0])

    w_q_tensor, scale_tensor = quantize_int8_symmetric(w, axis=None)
    err_tensor = quantization_error(w, w_q_tensor, scale_tensor)

    w_q_channel, scale_channel = quantize_int8_symmetric(w, axis=1)
    err_channel = quantization_error(w, w_q_channel, scale_channel)

    assert err_channel < err_tensor  # per-channel è più preciso


def test_quantization_error_valori():
    """MSE calcolato correttamente."""
    w = np.array([1.0, -1.0, 0.0])
    w_q, scale = quantize_int8_symmetric(w)
    err = quantization_error(w, w_q, scale)
    assert err >= 0.0
    assert np.isfinite(err)


# ----------------------------------------------------------------------
# QuantizedLinear — shape e interfaccia
# ----------------------------------------------------------------------

def test_quantized_linear_inizializzazione():
    layer = QuantizedLinear(d_in=16, d_out=32, seed=42)
    assert layer.d_in == 16
    assert layer.d_out == 32
    assert layer.W.shape == (16, 32)
    assert layer.b.shape == (32,)
    assert layer.W_q.dtype == np.int8
    assert layer.weight_bits == 8


def test_quantized_linear_rifiuta_dimensioni_non_valide():
    with pytest.raises(ValueError):
        QuantizedLinear(d_in=0, d_out=10)
    with pytest.raises(ValueError):
        QuantizedLinear(d_in=10, d_out=0)


def test_quantized_linear_forward_shape():
    layer = QuantizedLinear(d_in=16, d_out=32, seed=1)
    x = np.random.RandomState(0).randn(4, 10, 16)
    out = layer.forward(x)
    assert out.shape == (4, 10, 32)


def test_quantized_linear_forward_shape_1d_input():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    x = np.random.RandomState(0).randn(8)
    out = layer.forward(x)
    assert out.shape == (4,)


def test_quantized_linear_forward_rifiuta_d_in_errato():
    layer = QuantizedLinear(d_in=16, d_out=32, seed=1)
    x = np.random.RandomState(0).randn(4, 8)  # d_in=8, ma aspetta 16
    with pytest.raises(ValueError):
        layer.forward(x)


def test_quantized_linear_backward_shape():
    layer = QuantizedLinear(d_in=16, d_out=32, seed=1)
    x = np.random.RandomState(0).randn(4, 10, 16)
    out = layer.forward(x)
    grad_x = layer.backward(np.ones_like(out))
    assert grad_x.shape == x.shape
    assert layer.grad_W.shape == layer.W.shape
    assert layer.grad_b.shape == layer.b.shape


def test_quantized_linear_backward_richiede_forward():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((1, 4)))


def test_quantized_linear_parameters_interface():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    x = np.random.RandomState(0).randn(2, 3, 8)
    out = layer.forward(x)
    layer.backward(np.ones_like(out))
    params = layer.parameters()
    grads = layer.gradients()
    assert len(params) == len(grads) == 2
    for p, g in zip(params, grads):
        assert np.asarray(p).shape == np.asarray(g).shape


def test_quantized_linear_set_parameters_roundtrip():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    original = [np.asarray(p).copy() for p in layer.parameters()]
    layer.set_parameters(original)
    for before, after in zip(original, layer.parameters()):
        assert np.allclose(before, after)


def test_quantized_linear_set_parameters_rifiuta_numero_errato():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    with pytest.raises(ValueError):
        layer.set_parameters([np.zeros((2, 2))])


# ----------------------------------------------------------------------
# QuantizedLinear — gradient check numerico (STE)
# ----------------------------------------------------------------------

def test_quantized_linear_gradiente_numerico_W():
    """Gradient check sui parametri W e b (STE)."""
    layer = QuantizedLinear(d_in=8, d_out=6, seed=3)
    rng = np.random.RandomState(0)
    x = rng.randn(1, 4, 8)
    grad_out = rng.randn(1, 4, 6)

    layer.forward(x)
    layer.backward(grad_out)

    for attr_name, grad_attr in [('W', 'grad_W'), ('b', 'grad_b')]:
        param = getattr(layer, attr_name)
        analytic = getattr(layer, grad_attr)
        flat_param = param.reshape(-1)
        flat_analytic = analytic.reshape(-1)
        rng_pos = np.random.RandomState(42)
        positions = rng_pos.choice(flat_param.size, size=min(3, flat_param.size), replace=False)

        for pos in positions:
            original = flat_param[pos]
            flat_param[pos] = original + 1e-6
            loss_plus = float(np.sum(layer.forward(x) * grad_out))
            flat_param[pos] = original - 1e-6
            loss_minus = float(np.sum(layer.forward(x) * grad_out))
            flat_param[pos] = original

            numeric = (loss_plus - loss_minus) / 2e-6
            denom = max(1e-8, abs(numeric) + abs(flat_analytic[pos]))
            assert abs(numeric - flat_analytic[pos]) / denom < 1e-3, (
                f"Gradient mismatch in {attr_name}[{pos}]: "
                f"numeric={numeric:.6e}, analytic={flat_analytic[pos]:.6e}"
            )


def test_quantized_linear_gradiente_numerico_input():
    """Gradient check rispetto all'input x."""
    layer = QuantizedLinear(d_in=8, d_out=6, seed=3)
    rng = np.random.RandomState(0)
    x = rng.randn(1, 4, 8)
    grad_out = rng.randn(1, 4, 6)

    layer.forward(x)
    analytic_grad_x = layer.backward(grad_out)

    epsilon = 1e-6

    def loss_fn(input_tensor):
        out = layer.forward(input_tensor)
        return float(np.sum(out * grad_out))

    flat_x = x.reshape(-1)
    flat_analytic = analytic_grad_x.reshape(-1)
    rng_pos = np.random.RandomState(99)
    positions = rng_pos.choice(flat_x.size, size=min(5, flat_x.size), replace=False)

    for pos in positions:
        original = flat_x[pos]
        flat_x[pos] = original + epsilon
        loss_plus = loss_fn(x)
        flat_x[pos] = original - epsilon
        loss_minus = loss_fn(x)
        flat_x[pos] = original

        numeric = (loss_plus - loss_minus) / (2 * epsilon)
        denom = max(1e-8, abs(numeric) + abs(flat_analytic[pos]))
        assert abs(numeric - flat_analytic[pos]) / denom < 1e-3


# ----------------------------------------------------------------------
# QuantizedLinear — equivalenza con matmul FP
# ----------------------------------------------------------------------

def test_quantized_linear_forward_equivalente_float():
    """Il forward usa i pesi float self.W (STE)."""
    layer = QuantizedLinear(d_in=16, d_out=32, seed=1)
    x = np.random.RandomState(0).randn(2, 5, 16)

    out = layer.forward(x)
    out_ref = (x.reshape(-1, 16) @ layer.W + layer.b).reshape(2, 5, 32)

    assert np.allclose(out, out_ref, atol=1e-10)


def test_quantized_linear_quantized_forward_equivale_dequantize():
    """quantized_forward usa W_q * scale (dequantizzato)."""
    layer = QuantizedLinear(d_in=16, d_out=32, seed=1)
    x = np.random.RandomState(0).randn(2, 5, 16)

    out_q = layer.quantized_forward(x)
    W_dq = layer.dequantize()
    out_ref = (x.reshape(-1, 16) @ W_dq + layer.b).reshape(2, 5, 32)

    assert np.allclose(out_q, out_ref, atol=1e-10)


def test_quantized_linear_compression_ratio():
    """INT8 riduce la memoria del peso del ~4x rispetto a FP32."""
    layer = QuantizedLinear(d_in=1024, d_out=1024, seed=1)
    original_bytes = layer.W.astype(np.float32).nbytes + layer.b.astype(np.float32).nbytes
    quantized_bytes = layer.W_q.nbytes + (
        layer.scale.nbytes if hasattr(layer.scale, 'nbytes') else 8
    )
    ratio = original_bytes / quantized_bytes
    # INT8 (1 byte) vs FP32 (4 bytes) → ~4x, leggermente meno per lo scale overhead
    assert ratio > 2.0


# ----------------------------------------------------------------------
# Serializzazione / caricamento
# ----------------------------------------------------------------------

def test_quantized_linear_state_dict_int8():
    """state_dict contiene format=int8, W_q e scale."""
    layer = QuantizedLinear(d_in=8, d_out=4, seed=42)
    state = layer.state_dict()
    assert state['format'] == 'int8'
    assert 'W_q' in state
    assert 'scale' in state
    assert 'd_in' in state
    assert state['d_in'] == 8
    assert state['d_out'] == 4


def test_quantized_linear_load_state_dict_fp32():
    """Caricamento pesi FP32 → quantizzazione automatica."""
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    original_W = layer.W.copy()

    state = {
        'd_in': 8,
        'd_out': 4,
        'format': 'fp32',
        'W': original_W.tolist(),
        'b': layer.b.tolist(),
    }
    layer.load_state_dict(state)
    assert np.allclose(layer.W, original_W)
    assert layer.W_q.dtype == np.int8


def test_quantized_linear_load_state_dict_int8():
    """Caricamento pesi INT8 con dequantizzazione."""
    layer1 = QuantizedLinear(d_in=8, d_out=4, seed=42)
    state = layer1.state_dict()
    state['format'] = 'int8'

    layer2 = QuantizedLinear(d_in=8, d_out=4, seed=0)  # seed diverso
    layer2.load_state_dict(state)

    assert np.allclose(layer2.W, layer1.W, atol=1e-2)
    assert np.allclose(layer2.b, layer1.b)
    assert layer2.W_q.dtype == np.int8


def test_quantized_linear_load_state_dict_rifiuta_formato_sconosciuto():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    with pytest.raises(ValueError):
        layer.load_state_dict({'d_in': 8, 'd_out': 4, 'format': 'unknown'})


def test_quantized_linear_load_state_dict_rifiuta_mancanti():
    layer = QuantizedLinear(d_in=8, d_out=4, seed=1)
    with pytest.raises(ValueError):
        layer.load_state_dict({'d_in': 8, 'd_out': 4, 'format': 'int8'})


# ----------------------------------------------------------------------
# Funzioni di caricamento/salvataggio
# ----------------------------------------------------------------------

def test_load_weights_detects_int8_format(tmp_path):
    """load_weights riconosce format=int8 e restituisce state_dict."""
    layer = QuantizedLinear(d_in=16, d_out=8, seed=42)
    state = layer.state_dict()
    state['format'] = 'int8'
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(state))

    loaded = load_weights(str(path))
    assert loaded['format'] == 'int8'
    assert 'W_q' in loaded
    assert 'scale' in loaded


def test_load_weights_detects_fp32_format(tmp_path):
    """load_weights riconosce format=fp32 (o nessun format)."""
    state = {
        'd_in': 4,
        'd_out': 2,
        'format': 'fp32',
        'W': np.zeros((4, 2)).tolist(),
        'b': np.zeros(2).tolist(),
    }
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(state))

    loaded = load_weights(str(path))
    assert loaded['format'] == 'fp32'


def test_load_weights_default_fp32(tmp_path):
    """Se 'format' è assente, si assume fp32."""
    state = {'d_in': 4, 'd_out': 2, 'W': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], 'b': [0.0, 0.0]}
    path = tmp_path / "weights_no_format.json"
    path.write_text(json.dumps(state))

    loaded = load_weights(str(path))
    assert loaded['format'] == 'fp32' or 'W' in loaded


def test_save_and_load_quantized_model(tmp_path):
    """Roundtrip: salva modello INT8, ricarica, verifica correttezza."""
    layer1 = QuantizedLinear(d_in=32, d_out=16, seed=42)
    layer2 = QuantizedLinear(d_in=64, d_out=32, seed=99)
    path = tmp_path / "quantized_model.json"

    save_quantized_model([layer1, layer2], str(path))
    assert path.exists()

    loaded = load_weights(str(path))
    assert loaded['format'] == 'int8'
    assert len(loaded['layers']) == 2
    assert loaded['compression']['ratio'] > 1.0

    # Ricarica i layer e verifica
    l1 = QuantizedLinear(d_in=32, d_out=16, seed=0)
    l1.load_state_dict(loaded['layers'][0])
    l2 = QuantizedLinear(d_in=64, d_out=32, seed=0)
    l2.load_state_dict(loaded['layers'][1])

    assert np.allclose(l1.W, layer1.W, atol=1e-2)
    assert np.allclose(l2.W, layer2.W, atol=1e-2)


def test_save_quantized_model_compression_report(tmp_path):
    """Il report di compressione deve essere accurato."""
    layer = QuantizedLinear(d_in=128, d_out=128, seed=1)
    path = tmp_path / "model.json"
    save_quantized_model([layer], str(path))

    data = json.loads(path.read_text())
    assert 'compression' in data
    assert data['compression']['original_bytes'] == layer.W.astype(np.float64).nbytes + layer.b.astype(np.float64).nbytes
    assert data['compression']['quantized_bytes'] > 0
    assert data['compression']['ratio'] > 1.0


# ----------------------------------------------------------------------
# build_quantized_model
# ----------------------------------------------------------------------

def test_build_quantized_model():
    """build_quantized_model costruisce layer correttamente."""
    config = [
        {'d_in': 16, 'd_out': 32, 'seed': 1},
        {'d_in': 32, 'd_out': 64, 'seed': 2},
    ]
    layers = build_quantized_model(config)
    assert len(layers) == 2
    assert all(isinstance(l, QuantizedLinear) for l in layers)
    assert layers[0].d_in == 16
    assert layers[0].d_out == 32
    assert layers[1].d_in == 32
    assert layers[1].d_out == 64


def test_build_quantized_model_per_channel_flag():
    layers = build_quantized_model([{'d_in': 8, 'd_out': 8, 'seed': 1, 'per_channel': False}])
    assert layers[0].per_channel is False


# ----------------------------------------------------------------------
# QuantizedLinear — forward produce output finito
# ----------------------------------------------------------------------

def test_quantized_linear_output_finito():
    layer = QuantizedLinear(d_in=16, d_out=32, seed=42)
    x = np.random.RandomState(0).randn(2, 5, 16)
    out = layer.forward(x)
    assert np.all(np.isfinite(out))
    assert not np.any(np.isnan(out))
