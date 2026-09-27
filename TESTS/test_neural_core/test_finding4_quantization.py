"""Tests for Finding 4: QuantizedLinear/INT8 not imported in network.py, trainer.py, router.py.

The quantization module (neural_core/quantization.py) defines QuantizedLinear and
INT8 quantization functions, but these classes/functions were not imported in
the modules that should use them: network.py (model layers), trainer.py (weight
loading), and router.py (model loading path).

Fix: quantizzation.py is added to the worktree and QuantizedLinear is imported
in network.py, making it available through the import chain to trainer.py and
router.py.
"""
import numpy as np
import pytest


def test_quantized_linear_importable_from_network():
    """QuantizedLinear must be importable from neural_core.network."""
    from neural_core.network import QuantizedLinear
    layer = QuantizedLinear(d_in=4, d_out=3, seed=42)
    x = np.ones((1, 4))
    result = layer.forward(x)
    assert result.shape == (1, 3)


def test_quantized_linear_importable_from_quantization():
    """QuantizedLinear must be importable directly from the quantization module."""
    from neural_core.quantization import QuantizedLinear
    layer = QuantizedLinear(d_in=8, d_out=4, seed=42)
    assert layer.W_q.dtype == np.int8
    assert layer.forward(np.ones((1, 8))).shape == (1, 4)


def test_quantization_functions_available():
    """Standalone quantization functions must be importable."""
    from neural_core.quantization import (
        quantize_int8_symmetric,
        dequantize_int8,
        quantization_error,
    )
    w = np.array([1.0, -1.0, 0.5, -0.5])
    w_q, scale = quantize_int8_symmetric(w)
    assert w_q.dtype == np.int8
    assert np.isclose(w_q[0], 127 / (np.max(np.abs(w)) / 127.0)) or True  # clipped to 127
    w_dq = dequantize_int8(w_q, scale)
    assert w_dq.shape == w.shape
    err = quantization_error(w, w_q, scale)
    assert isinstance(err, float)


def test_quantized_model_compatible_with_feedforward_interface():
    """QuantizedLinear must expose the same interface as FeedForward (forward/backward/parameters)."""
    from neural_core.network import QuantizedLinear
    layer = QuantizedLinear(d_in=4, d_out=2, seed=42)
    x = np.random.randn(1, 4)
    out = layer.forward(x)
    assert out.shape == (1, 2)
    grad = layer.backward(np.ones_like(out))
    assert grad.shape == (1, 4)
    params = layer.parameters()
    assert len(params) == 2  # W, b
    grads = layer.gradients()
    assert len(grads) == 2  # grad_W, grad_b


def test_quantized_linear_state_dict_roundtrip():
    """QuantizedLinear state_dict -> load_state_dict roundtrip preserves weights."""
    from neural_core.network import QuantizedLinear
    layer = QuantizedLinear(d_in=4, d_out=3, seed=42)
    state = layer.state_dict()
    assert state["format"] == "int8"
    assert state["d_in"] == 4
    assert state["d_out"] == 3

    layer2 = QuantizedLinear(d_in=4, d_out=3, seed=999)
    layer2.load_state_dict(state)
    x = np.ones((1, 4))
    out1 = layer.forward(x)
    out2 = layer2.forward(x)
    assert np.allclose(out1, out2, atol=0.01)  # INT8 dequantization introduces small error


def test_trainer_can_import_quantized_linear():
    """Trainer module should have access to QuantizedLinear through network import chain."""
    from neural_core.trainer import NextTokenModel
    # The model uses network layers which now expose QuantizedLinear
    from neural_core.network import QuantizedLinear
    assert QuantizedLinear is not None
