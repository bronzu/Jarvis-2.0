"""
Numerical gradient checking for the Neural Core.

Compares analytical gradients from MicroTransformer.backward() with
numerical gradients computed via finite differences.

The test uses a small model to keep runtime low.
"""

import numpy as np
import pytest

from neural_core.network import MicroTransformer
from neural_core.loss import CrossEntropyLoss


@pytest.fixture(scope="module")
def small_model_and_data():
    vocab_size = 50
    d_model = 16
    num_heads = 2
    d_ff = 32
    num_layers = 1
    max_len = 16
    seed = 42

    model = MicroTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        dropout=0.0,
        seed=seed,
    )

    rng = np.random.RandomState(seed)
    token_ids = rng.randint(0, vocab_size, size=(2, 8)).tolist()
    targets = rng.randint(0, vocab_size, size=(2, 8)).tolist()

    return model, token_ids, targets


def _numerical_gradient(model, token_ids, targets, eps=1e-5):
    """Compute numerical gradients for all parameters via central differences."""
    params = model.get_parameters()
    num_grads = [np.zeros_like(p) for p in params]

    for idx, param in enumerate(params):
        flat_param = param.ravel()
        flat_grad = num_grads[idx].ravel()
        for i in range(flat_param.size):
            original = flat_param[i]

            flat_param[i] = original + eps
            logits_plus = model.forward(token_ids, training=False)
            loss_plus = float(CrossEntropyLoss().forward(logits_plus, targets))

            flat_param[i] = original - eps
            logits_minus = model.forward(token_ids, training=False)
            loss_minus = float(CrossEntropyLoss().forward(logits_minus, targets))

            flat_grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
            flat_param[i] = original

    return num_grads


def test_numerical_gradient_matches_analytical(small_model_and_data):
    model, token_ids, targets = small_model_and_data

    model.forward(token_ids, training=True)
    analytical_grads = model.backward(targets)
    numerical_grads = _numerical_gradient(model, token_ids, targets)

    assert len(analytical_grads) == len(numerical_grads), "Gradient count mismatch"

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for a_grad, n_grad in zip(analytical_grads, numerical_grads):
        a = np.asarray(a_grad, dtype=np.float64)
        n = np.asarray(n_grad, dtype=np.float64)
        diff = np.abs(a - n)
        max_abs_diff = max(max_abs_diff, float(diff.max()))

        denom = np.maximum(np.abs(a) + np.abs(n), 1e-8)
        rel_diff = diff / denom
        max_rel_diff = max(max_rel_diff, float(rel_diff.max()))

    assert max_abs_diff < 1e-5, (
        f"Numerical gradient check failed: max absolute diff {max_abs_diff:.3e}"
    )
    assert max_rel_diff < 5e-3, (
        f"Numerical gradient check failed: max relative diff {max_rel_diff:.3e}"
    )
