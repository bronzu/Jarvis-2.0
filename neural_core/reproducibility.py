"""
Reproducibility experiment for JARVIS 2.0.

Runs two training runs with identical configuration and compares:
- loss trajectory
- model parameters
- checkpoint checksums
- training history

Demonstrates bitwise or numerical reproducibility.
"""

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from neural_core.network import MicroTransformer
from neural_core.tokenizer import Tokenizer
from neural_core.trainer import Trainer
from neural_core.dataset import Dataset
from neural_core.loss import CrossEntropyLoss
from neural_core.optimizer import Adam


def _run_training(seed: int, steps: int = 10):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega " * 20
    tokenizer = Tokenizer(max_vocab_size=200, mode="whitespace")
    tokenizer.fit(text)
    model = MicroTransformer(
        vocab_size=len(tokenizer.vocabulary),
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=1,
        max_len=64,
        seed=seed,
    )
    ids = tokenizer.encode(text)
    ds = Dataset(ids, context_size=16, stride=16)
    loss = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(model=model, tokenizer=tokenizer, loss=loss, optimizer=optimizer)

    history = []
    for step in range(steps):
        batch = ds[step % len(ds)]
        loss_val = trainer.train_step(batch[0], batch[1])
        history.append(loss_val)

    params = model.get_parameters()
    param_checksum = hashlib.sha256(pickle.dumps(params)).hexdigest()
    return {
        "seed": seed,
        "steps": steps,
        "final_loss": history[-1],
        "loss_history": history,
        "param_checksum": param_checksum,
        "vocab_size": len(tokenizer.vocabulary),
    }


def run_reproducibility_experiment(steps: int = 10, tolerance: float = 1e-6):
    run_a = _run_training(seed=42, steps=steps)
    run_b = _run_training(seed=42, steps=steps)

    loss_match = all(abs(a - b) < tolerance for a, b in zip(run_a["loss_history"], run_b["loss_history"]))
    params_match = run_a["param_checksum"] == run_b["param_checksum"]

    return {
        "run_a": run_a,
        "run_b": run_b,
        "loss_match": loss_match,
        "params_match": params_match,
        "numerically_reproducible": loss_match and params_match,
        "tolerance": tolerance,
        "max_loss_diff": max(abs(a - b) for a, b in zip(run_a["loss_history"], run_b["loss_history"])),
    }


if __name__ == "__main__":
    result = run_reproducibility_experiment()
    print(json.dumps(result, indent=2, ensure_ascii=False))
