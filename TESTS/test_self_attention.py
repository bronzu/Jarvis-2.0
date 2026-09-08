import numpy as np
import pytest

from neural_core.model import SelfAttention, causal_mask, softmax, softmax_backward


def test_self_attention_outputs_expected_shape():
    attention = SelfAttention()
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    k = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    v = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=float)

    output = attention.forward(q, k, v)

    assert output.shape == (2, 2)
    assert np.all(np.isfinite(output))


# ======================================================================
# SOFTMAX
# ======================================================================


def test_softmax_normalizza_sull_ultimo_asse():
    values = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(values)

    assert probs.shape == values.shape
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    assert np.allclose(probs[1], 1.0 / 3.0)


def test_softmax_globale_con_axis_none():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    probs = softmax(values, axis=None)

    assert np.isclose(np.sum(probs), 1.0)


def test_softmax_stabile_con_valori_grandi():
    probs = softmax(np.array([1000.0, 1001.0, 1002.0]))

    assert np.all(np.isfinite(probs))
    assert np.isclose(np.sum(probs), 1.0)


def test_softmax_backward_corrisponde_al_gradiente_numerico():
    logits = np.array([[0.4, -1.2, 0.7]])
    grad_output = np.array([[1.0, 0.0, -0.5]])

    probs = softmax(logits)
    analytic = softmax_backward(probs, grad_output)

    epsilon = 1e-7
    numeric = np.zeros_like(logits)
    for index in range(logits.shape[1]):
        perturbed = logits.copy()
        perturbed[0, index] += epsilon
        plus = float(np.sum(softmax(perturbed) * grad_output))
        perturbed[0, index] -= 2 * epsilon
        minus = float(np.sum(softmax(perturbed) * grad_output))
        numeric[0, index] = (plus - minus) / (2 * epsilon)

    assert np.allclose(analytic, numeric, atol=1e-5)


# ======================================================================
# CAUSAL MASKING
# ======================================================================


def test_self_attention_causal_ignora_il_futuro():
    attention = SelfAttention(causal=True)
    q = np.eye(3)
    k = np.eye(3)
    v = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])

    attention.forward(q, k, v)
    weights = attention.attention_weights()

    assert weights.shape == (3, 3)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)
    assert np.allclose(weights[np.triu_indices(3, 1)], 0.0)
    assert np.isclose(weights[0, 0], 1.0)


def test_self_attention_causal_prima_posizione_copia_il_primo_value():
    attention = SelfAttention(causal=True)
    q = np.eye(2)
    k = np.eye(2)
    v = np.array([[7.0, -3.0], [1.0, 1.0]])

    output = attention.forward(q, k, v)

    assert np.allclose(output[0], v[0])


def test_self_attention_maschera_esplicita():
    attention = SelfAttention()
    q = np.eye(2)
    k = np.eye(2)
    v = np.array([[1.0], [9.0]])

    output = attention.forward(q, k, v, mask=causal_mask(2)[0, 0])

    assert np.isclose(output[0, 0], 1.0)


def test_self_attention_backward_gradiente_numerico():
    rng = np.random.RandomState(0)
    q = rng.randn(3, 4)
    k = rng.randn(3, 4)
    v = rng.randn(3, 2)
    grad_output = rng.randn(3, 2)

    attention = SelfAttention(causal=True)
    attention.forward(q, k, v)
    grad_q, grad_k, grad_v = attention.backward(grad_output)

    assert grad_q.shape == q.shape
    assert grad_k.shape == k.shape
    assert grad_v.shape == v.shape

    def scalar_loss(query, key, value):
        local = SelfAttention(causal=True)
        return float(np.sum(local.forward(query, key, value) * grad_output))

    epsilon = 1e-6
    for tensor, analytic in ((q, grad_q), (k, grad_k), (v, grad_v)):
        flat = tensor.reshape(-1)
        flat_analytic = analytic.reshape(-1)
        for index in (0, flat.size - 1):
            original = flat[index]
            flat[index] = original + epsilon
            plus = scalar_loss(q, k, v)
            flat[index] = original - epsilon
            minus = scalar_loss(q, k, v)
            flat[index] = original
            numeric = (plus - minus) / (2 * epsilon)
            scale = max(1e-8, abs(numeric) + abs(flat_analytic[index]))
            assert abs(numeric - flat_analytic[index]) / scale < 1e-4


def test_self_attention_backward_richiede_forward():
    attention = SelfAttention()
    with pytest.raises(RuntimeError):
        attention.backward(np.ones((2, 2)))


def test_self_attention_attention_weights_richiede_forward():
    attention = SelfAttention()
    with pytest.raises(RuntimeError):
        attention.attention_weights()


def test_self_attention_dimensioni_incoerenti():
    attention = SelfAttention()

    with pytest.raises(ValueError):
        attention.forward(np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 2)))

    with pytest.raises(ValueError):
        attention.forward(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 2)))

    with pytest.raises(ValueError):
        attention.forward(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((3, 2)), causal=True)


def test_self_attention_accetta_vettori_monodimensionali():
    attention = SelfAttention()
    output = attention.forward(np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([4.0, 2.0]))

    assert output.shape == (1, 2)
    assert np.allclose(output[0], [4.0, 2.0])
