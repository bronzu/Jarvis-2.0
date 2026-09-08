import sys
import pickle
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_core.trainer import load_model_from_weights
from neural_core.tokenizer import Tokenizer
from executive_system.router import ExecutiveRouter


def test_tokenizer_loads():
    path = Path(__file__).resolve().parents[1] / "neural_core" / "data" / "tokenizer.pkl"
    assert path.exists(), "tokenizer.pkl not found"
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.mode == "whitespace"
    assert len(tokenizer.vocabulary) == 8590
    print("test_tokenizer_loads: PASS")


def test_model_loads():
    path = Path(__file__).resolve().parents[1] / "neural_core" / "data" / "model.pkl"
    assert path.exists(), "model.pkl not found"
    tokenizer_path = Path(__file__).resolve().parents[1] / "neural_core" / "data" / "tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model_from_weights(str(path), vocab_size=len(tokenizer.vocabulary))
    assert model.parameter_count() == 2737806
    assert model.context_size == 256
    print("test_model_loads: PASS")


def test_forward_pass():
    import numpy as np
    path = Path(__file__).resolve().parents[1] / "neural_core" / "data" / "model.pkl"
    tokenizer_path = Path(__file__).resolve().parents[1] / "neural_core" / "data" / "tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    model = load_model_from_weights(str(path), vocab_size=len(tokenizer.vocabulary))
    tokens = np.array([[1, 2, 3, 4, 5]])
    logits = model.forward(tokens, training=False)
    assert logits.shape == (1, 5, 8590)
    print("test_forward_pass: PASS")


def test_generation():
    router = ExecutiveRouter()
    result = router.neural_interface.generate("Hello JARVIS")
    assert "text" in result
    assert result["model"] == "neural-core-internal"
    assert result["fallback"] is False
    assert result["context_window"] == 256
    print("test_generation: PASS")


def test_interface_status():
    router = ExecutiveRouter()
    status = router.neural_interface.status()
    assert status["model_available"] is True
    assert status["tokenizer_available"] is True
    assert status["model_architecture"] == "MicroTransformer"
    assert status["context_window"] == 256
    assert status["parameter_count"] == 2737806
    assert status["vocab_size"] == 8590
    assert status["tokenizer_mode"] == "whitespace"
    assert status["external_model_required"] is False
    print("test_interface_status: PASS")


def test_runtime_availability():
    router = ExecutiveRouter()
    assert router.neural_interface.model is not None
    assert router.neural_interface.tokenizer is not None
    assert router.neural_interface.is_internal() is True
    print("test_runtime_availability: PASS")


if __name__ == "__main__":
    test_tokenizer_loads()
    test_model_loads()
    test_forward_pass()
    test_generation()
    test_interface_status()
    test_runtime_availability()
    print("\nAll tests passed.")
