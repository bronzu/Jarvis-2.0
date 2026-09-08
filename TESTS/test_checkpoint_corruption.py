"""
Checkpoint corruption and compatibility tests.
"""

import json
import os
import pickle
import tempfile

import numpy as np
import pytest

from neural_core.network import MicroTransformer
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer
from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam


def _make_trainer(tmp_path):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    tokenizer = Tokenizer(max_vocab_size=20, mode="whitespace")
    tokenizer.fit(text)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=32,
        seed=42,
    )
    ids = tokenizer.encode(text)
    ds = Dataset(ids, context_size=8, stride=8)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        loss=loss,
        optimizer=optimizer,
    )


def test_checkpoint_save_load_roundtrip(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "ckpt.pkl"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()

    trainer2 = _make_trainer(tmp_path)
    state = trainer2.load_checkpoint(ckpt)
    assert state["model_class"] == "MicroTransformer"
    assert state["model_config"]["vocab_size"] == trainer.model.vocab_size


def test_checkpoint_rejects_corrupted_file(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "corrupt.pkl"
    ckpt.write_bytes(b"not a pickle")

    with pytest.raises((pickle.UnpicklingError, EOFError, ValueError)):
        trainer.load_checkpoint(ckpt)


def test_checkpoint_rejects_invalid_format(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "bad_format.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump({"format": "unknown"}, f)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(ckpt)


def test_checkpoint_rejects_wrong_model_class(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "wrong_model.pkl"
    state = trainer.state_dict()
    state["model_class"] = "NonExistentModel"
    with open(ckpt, "wb") as f:
        pickle.dump(state, f)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(ckpt)


def test_checkpoint_rejects_config_mismatch(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "mismatch.pkl"
    state = trainer.state_dict()
    state["model_config"]["d_model"] = 999
    with open(ckpt, "wb") as f:
        pickle.dump(state, f)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(ckpt)


def test_checkpoint_rejects_tokenizer_mismatch(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "tok_mismatch.pkl"
    state = trainer.state_dict()
    state["tokenizer_state"] = {"format": "different", "version": 99, "mode": "technical", "special_tokens": [], "vocabulary": {}}
    with open(ckpt, "wb") as f:
        pickle.dump(state, f)

    with pytest.raises(ValueError):
        trainer.load_checkpoint(ckpt)


def test_resume_restores_training_state(tmp_path):
    trainer = _make_trainer(tmp_path)
    ckpt = tmp_path / "resume.pkl"
    trainer.global_step = 10
    trainer.epoch = 2
    trainer.best_validation_loss = 0.5
    trainer.save_checkpoint(ckpt)

    trainer2 = _make_trainer(tmp_path)
    state = trainer2.load_checkpoint(ckpt)
    assert trainer2.global_step == 10
    assert trainer2.epoch == 2
    assert trainer2.best_validation_loss == 0.5
