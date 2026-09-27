"""
Sonda di RSS di processo per il criterio (a).

Esegue UNA variante in un processo pulito e riporta il picco di RSS
(`ru_maxrss`). Serve perche' `tracemalloc` misura l'allocazione transitiva
durante `compute_fisher_matrix`, che e' dominata dalle attivazioni di
forward/backward e non distingue le due varianti; il RSS di picco del
processo, isolando una variante per processo, attribuisce correttamente la
memoria alle strutture Fisher/optimal effettivamente persistenti.

Uso: python3 rss_probe.py <lora_rank>      (lora_rank=0 -> full-parameter)
"""

import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_core.network import MicroTransformer  # noqa: E402
from neural_core.tokenizer import Tokenizer  # noqa: E402
from neural_core.loss import CrossEntropyLoss  # noqa: E402
from neural_core.optimizer import Adam  # noqa: E402
from neural_core.trainer import Trainer  # noqa: E402

PRODUCTION_CONFIG = dict(vocab_size=8590, d_model=128, num_heads=4, d_ff=256,
                         num_layers=4, max_len=256)
FISHER_SAMPLES = 50
CONTEXT_SIZE = 16
SEED = 42


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def main():
    rank = int(sys.argv[1])
    vocab = PRODUCTION_CONFIG["vocab_size"]

    baseline = peak_rss_mb()
    model = MicroTransformer(seed=SEED, lora_rank=rank, lora_alpha=16.0,
                             **PRODUCTION_CONFIG)
    trainer = Trainer(model=model, tokenizer=Tokenizer(max_vocab_size=vocab,
                                                       mode="whitespace"),
                      loss=CrossEntropyLoss(),
                      optimizer=Adam(learning_rate=0.01), seed=SEED)
    rng = np.random.default_rng(SEED)
    ds = (rng.integers(0, vocab, size=(FISHER_SAMPLES, CONTEXT_SIZE)),
          rng.integers(0, vocab, size=(FISHER_SAMPLES, CONTEXT_SIZE)))
    trainer.compute_fisher_matrix(ds, num_samples=FISHER_SAMPLES)

    after = peak_rss_mb()
    persistent = int(sum(np.asarray(v).nbytes for v in trainer.fisher_matrix.values())
                     + sum(np.asarray(v).nbytes for v in trainer.optimal_weights.values()))
    entries = sum(int(np.asarray(v).size) for v in trainer.fisher_matrix.values())
    print(f"{rank} {baseline:.4f} {after:.4f} {after - baseline:.4f} "
          f"{persistent} {entries}")


if __name__ == "__main__":
    main()
