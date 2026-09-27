"""
Test unitari per Causal Linear Attention (MultiHeadLinearAttention).

Verifica:
  - shape invariata rispetto a MultiHeadAttention
  - causalità (nessun future leakage)
  - correttezza analitica del backward via gradient check numerico
  - interfaccia parameters()/gradients()/set_parameters() coerente
  - compatibilità con MascheraCausale
"""

import math

import numpy as np
import pytest

from neural_core.model import causal_mask
from neural_core.network import (
    MultiHeadLinearAttention,
    MultiHeadAttention,
    elu_plus_one,
    elu_plus_one_derivative,
)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _random_input(batch_size=2, seq_len=5, d_model=8, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(batch_size, seq_len, d_model)


# ----------------------------------------------------------------------
# Feature map
# ----------------------------------------------------------------------

def test_elu_plus_one_sempre_positiva():
    x = np.linspace(-10, 10, 100)
    phi = elu_plus_one(x)
    assert np.all(phi > 0)
    assert np.all(np.isfinite(phi))


def test_elu_plus_one_derivativa_corretta():
    x = np.linspace(-10, 10, 100)
    eps = 1e-7
    numeric = (elu_plus_one(x + eps) - elu_plus_one(x - eps)) / (2 * eps)
    analytic = elu_plus_one_derivative(x)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_elu_plus_one_valori_noti():
    assert elu_plus_one(0.0) == 1.0
    assert np.isclose(elu_plus_one(-10.0), math.exp(-10.0))
    assert np.isclose(elu_plus_one(5.0), 6.0)


# ----------------------------------------------------------------------
# Shape & interfaccia
# ----------------------------------------------------------------------

def test_linear_attention_stessa_firma_di_multi_head():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=42)
    ma = MultiHeadAttention(d_model=8, num_heads=2, seed=42)

    x = _random_input(batch_size=2, seq_len=5, d_model=8, seed=99)

    out_la = la.forward(x, mask=causal_mask(5))
    out_ma = ma.forward(x, mask=causal_mask(5))

    assert out_la.shape == out_ma.shape == x.shape


def test_linear_attention_rifiuta_d_model_non_divisibile():
    with pytest.raises(ValueError):
        MultiHeadLinearAttention(d_model=10, num_heads=4)


def test_linear_attention_output_finito():
    la = MultiHeadLinearAttention(d_model=16, num_heads=4, seed=7)
    x = _random_input(seq_len=10, d_model=16, seed=3)
    out = la.forward(x, mask=causal_mask(10))
    assert np.all(np.isfinite(out))
    assert not np.any(np.isnan(out))


def test_linear_attention_num_parametri_uguale_a_multi_head():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=1)
    ma = MultiHeadAttention(d_model=8, num_heads=2, seed=1)
    assert len(la.parameters()) == len(ma.parameters()) == 8
    assert len(la.gradients()) == len(ma.gradients()) == 8


def test_linear_attention_get_set_parameters_roundtrip():
    la = MultiHeadLinearAttention(d_model=12, num_heads=3, seed=11)
    x = _random_input(batch_size=1, seq_len=6, d_model=12, seed=5)
    la.forward(x, mask=causal_mask(6))

    original = [np.asarray(p).copy() for p in la.parameters()]
    la.set_parameters(original)
    for before, after in zip(original, la.parameters()):
        assert np.allclose(before, after)


def test_linear_attention_set_parameters_rifiuta_numero_errato():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=1)
    with pytest.raises(ValueError):
        la.set_parameters([np.zeros((2, 2))])


# ----------------------------------------------------------------------
# Causal masking (no future leakage)
# ----------------------------------------------------------------------

def test_linear_attention_causale_non_dipende_futuro():
    rng = np.random.RandomState(123)
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=123)
    la.forward(rng.randn(1, 6, 8), mask=causal_mask(6))
    weights = la.attention_weights()

    future = np.triu(np.ones((6, 6), dtype=bool), 1)
    assert np.allclose(weights[..., future], 0.0)


def test_linear_attention_pesi_sommanno_a_1():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=42)
    x = _random_input(batch_size=1, seq_len=7, d_model=8, seed=17)
    la.forward(x, mask=causal_mask(7))
    weights = la.attention_weights()

    assert weights.shape == (1, 2, 7, 7)
    assert np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-5)
    future = np.triu(np.ones((7, 7), dtype=bool), 1)
    assert np.allclose(weights[..., future], 0.0)


def test_linear_attention_ignora_token_futuro():
    """La posizione t non cambia se i token successivi vengono modificati."""
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=99)

    base = _random_input(batch_size=1, seq_len=5, d_model=8, seed=42)
    modified = base.copy()
    modified[0, 3] = 99.0
    modified[0, 4] = -99.0

    out_base = la.forward(base, mask=causal_mask(5)).copy()

    la2 = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=99)
    out_modified = la2.forward(modified, mask=causal_mask(5)).copy()

    assert np.allclose(out_base[0, :3], out_modified[0, :3])


def test_linear_attention_attention_weights_richiede_forward():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=1)
    with pytest.raises(RuntimeError):
        la.attention_weights()


# ----------------------------------------------------------------------
# Backward — shape e gradient check numerico
# ----------------------------------------------------------------------

def test_linear_attention_backward_restituisce_shape_corretta():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=3)
    x = _random_input(batch_size=2, seq_len=5, d_model=8, seed=7)
    out = la.forward(x, mask=causal_mask(5))
    grad_x = la.backward(np.ones_like(out))

    assert grad_x.shape == x.shape
    assert np.all(np.isfinite(grad_x))


def test_linear_attention_backward_parametri_coerenti():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=5)
    x = _random_input(batch_size=2, seq_len=5, d_model=8, seed=11)
    out = la.forward(x, mask=causal_mask(5))
    la.backward(np.ones_like(out))

    for param, grad in zip(la.parameters(), la.gradients()):
        assert np.asarray(param).shape == np.asarray(grad).shape
        assert np.all(np.isfinite(grad))


def test_linear_attention_backward_richiede_forward():
    la = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=1)
    with pytest.raises(RuntimeError):
        la.backward(np.ones((1, 4, 8)))


def test_linear_attention_gradiente_numerico_w():
    """Gradient check numerico sui parametri W_q, W_k, W_v, W_o."""
    d_model, num_heads, seq_len = 8, 2, 4
    la = MultiHeadLinearAttention(d_model=d_model, num_heads=num_heads, seed=5)
    x = np.random.RandomState(0).randn(1, seq_len, d_model)

    la.forward(x, mask=causal_mask(seq_len))
    grad_out = np.random.RandomState(0).randn(*la.forward(x, mask=causal_mask(seq_len)).shape)

    def scalar_loss(model):
        out = model.forward(x, mask=causal_mask(seq_len))
        return float(np.sum(out * grad_out))

    la.forward(x, mask=causal_mask(seq_len))
    la.backward(grad_out)

    param_names = ['W_q', 'W_k', 'W_v', 'W_o', 'b_q', 'b_k', 'b_v', 'b_o']
    grad_attrs = ['grad_W_q', 'grad_W_k', 'grad_W_v', 'grad_W_o',
                  'grad_b_q', 'grad_b_k', 'grad_b_v', 'grad_b_o']
    rng = np.random.RandomState(42)

    for name, grad_attr in zip(param_names, grad_attrs):
        param = getattr(la, name)
        analytic = getattr(la, grad_attr)
        flat_param = param.reshape(-1)
        flat_analytic = analytic.reshape(-1)
        positions = rng.choice(flat_param.size, size=min(3, flat_param.size), replace=False)

        for pos in positions:
            original = flat_param[pos]
            flat_param[pos] = original + 1e-6
            loss_plus = scalar_loss(la)
            flat_param[pos] = original - 1e-6
            loss_minus = scalar_loss(la)
            flat_param[pos] = original

            numeric = (loss_plus - loss_minus) / (2e-6)
            denom = max(1e-8, abs(numeric) + abs(flat_analytic[pos]))
            assert abs(numeric - flat_analytic[pos]) / denom < 1e-3, (
                f"Gradient mismatch in {name}[{pos}]: "
                f"numeric={numeric:.6e}, analytic={flat_analytic[pos]:.6e}"
            )


def test_linear_attention_gradiente_numerico_input():
    """Gradient check numerico rispetto all'input x."""
    d_model, num_heads, seq_len = 8, 2, 4
    la = MultiHeadLinearAttention(d_model=d_model, num_heads=num_heads, seed=5)
    rng = np.random.RandomState(0)
    x = rng.randn(1, seq_len, d_model)

    grad_out = rng.randn(1, seq_len, d_model)

    # Analytic gradient
    la.forward(x, mask=causal_mask(seq_len))
    analytic_grad_x = la.backward(grad_out)

    epsilon = 1e-6

    def loss_fn(input_tensor):
        out = la.forward(input_tensor, mask=causal_mask(seq_len))
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
        assert abs(numeric - flat_analytic[pos]) / max(1e-8, abs(numeric) + abs(flat_analytic[pos])) < 1e-3, (
            f"Input gradient mismatch at [{pos}]: "
            f"numeric={numeric:.6e}, analytic={flat_analytic[pos]:.6e}"
        )


def test_linear_attention_backward_passa_through_training_block():
    """Verifica che MultiHeadLinearAttention si integri in TransformerBlock
    mantenendo la forma del tensore."""
    from neural_core.network import TransformerBlock

    d_model, num_heads, d_ff = 8, 2, 16
    original_init = TransformerBlock.__init__

    # Costruisci un blocco e sostituisci l'attenzione
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, seed=42)
    block.attn = MultiHeadLinearAttention(d_model=d_model, num_heads=num_heads, seed=42)

    x = _random_input(batch_size=2, seq_len=6, d_model=d_model, seed=10)
    out = block.forward(x, mask=causal_mask(6))
    grad_x = block.backward(np.ones_like(out))

    assert out.shape == x.shape
    assert grad_x.shape == x.shape


def test_linear_attention_backward_full_model_roundtrip():
    """Verifica che MicroTransformer con MultiHeadLinearAttention produca
    gradienti coerenti con get_parameters()/get_gradients()."""
    from neural_core.network import MicroTransformer

    model = MicroTransformer(
        vocab_size=20, d_model=8, num_heads=2, d_ff=16,
        num_layers=2, max_len=128, seed=5,
    )
    # Sostituisci l'attenzione di ogni blocco
    for i, layer in enumerate(model.layers):
        layer.attn = MultiHeadLinearAttention(d_model=8, num_heads=2, seed=42 + i)

    inputs = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])

    model.forward(inputs)
    gradients = model.backward(targets)
    parameters = model.get_parameters()

    assert len(gradients) == len(parameters)
    for param, grad in zip(parameters, gradients):
        assert np.asarray(param).shape == np.asarray(grad).shape
