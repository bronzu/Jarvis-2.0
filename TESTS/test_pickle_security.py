import json
import os
import pickle
import tempfile

import numpy as np

from neural_core.trainer import _load_weights_safe, _load_checkpoint_safe, load_model_from_weights


class MaliciousPayload:
    """Payload that attempts arbitrary code execution via pickle __reduce__."""
    def __reduce__(self):
        return (os.system, ('echo HACKED > /tmp/pwned_by_pickle_test.txt',))


def test_safe_weights_loader_rejects_pickle_format(tmp_path):
    weights_path = tmp_path / "weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump({"model_class": "TestModel", "parameters": [np.array([1.0])]}, f)

    try:
        _load_weights_safe(str(weights_path))
        assert False, "Expected JSON decode error"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        pass


def test_safe_checkpoint_loader_rejects_pickle_format(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"format": "jarvis-neural-core-checkpoint", "parameters": [np.array([1.0])]}, f)

    try:
        _load_checkpoint_safe(str(checkpoint_path))
        assert False, "Expected JSON decode error"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        pass


def test_load_model_from_weights_rejects_malicious_pickle(tmp_path):
    weights_path = tmp_path / "weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(MaliciousPayload(), f)

    try:
        load_model_from_weights(str(weights_path))
        assert False, "Expected error"
    except Exception:
        pass


def test_load_checkpoint_rejects_malicious_pickle(tmp_path):
    from neural_core.trainer import Trainer, NextTokenModel
    from neural_core.loss import CrossEntropyLoss
    from neural_core.optimizer import Adam
    from neural_core.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    model = NextTokenModel(vocab_size=len(tokenizer.vocabulary), embedding_dim=8)
    trainer = Trainer(model, tokenizer, CrossEntropyLoss(), Adam())

    checkpoint_path = tmp_path / "checkpoint.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(MaliciousPayload(), f)

    try:
        trainer.load_checkpoint(str(checkpoint_path))
        assert False, "Expected error"
    except Exception:
        pass


def test_malicious_pickle_does_not_create_artifacts():
    if os.path.exists('/tmp/pwned_by_pickle_test.txt'):
        os.remove('/tmp/pwned_by_pickle_test.txt')
