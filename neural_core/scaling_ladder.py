"""
Phase 2B — Scaling ladder and context-length analysis.

This module declares candidate configurations for the model scaling preparation.
It does NOT execute training. The ladder is progressive:

    CURRENT  ~2.5M   (the protected baseline)
    SMALL    ~10M
    MEDIUM   ~25M
    LARGE    ~50M
    OPTIONAL ~100M

Targets are APPROXIMATE. Each configuration keeps the same decoder-only math
(width + depth + FFN size + context length vary; the fundamental math does not).
No heads value changes the parameter count, so heads are chosen for a clean
``d_k`` split (d_model / num_heads).

The context-length analysis isolates 256 / 512 / 1024. Context length does NOT
change the parameter count; it only changes attention/activation memory and
compute. 1024 is NOT selected as a new default here (would require authorization).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from neural_core.model_config import ModelConfig


#: Real vocabulary derived from the trained tokenizer (NOT 8000).
#: Documented discrepancy: tokenizer declares ``max_vocab_size=8000`` but the
#: fitted vocabulary is 8590. We keep the REAL vocabulary (8590) for scaling.
REAL_VOCAB_SIZE = 8590

#: Baseline context lengths already verified in Phase 1.
VERIFIED_CONTEXT_SIZES = (128, 256, 512, 1024)

#: Default context for the ladder (unchanged from baseline).
DEFAULT_CONTEXT_SIZE = 256


@dataclass
class LadderEntry:
    name: str
    target_params: str
    vocab_size: int
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    max_len: int
    dropout: float = 0.1
    dtype: str = "float64"
    seed: Optional[int] = None

    def to_config(self, backend: str = "cpu", device: str = "cpu") -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            max_len=self.max_len,
            dropout=self.dropout,
            dtype=self.dtype,
            seed=self.seed,
            backend=backend,
            device=device,
        )

    def as_dict(self) -> Dict:
        return asdict(self)


#: The scaling ladder. Heads chosen so d_model % num_heads == 0 and d_k is sane.
SCALING_LADDER: List[LadderEntry] = [
    LadderEntry(
        name="CURRENT", target_params="~2.5M",
        vocab_size=REAL_VOCAB_SIZE, d_model=128, num_heads=4, d_ff=256,
        num_layers=2, max_len=DEFAULT_CONTEXT_SIZE, dropout=0.1,
    ),
    LadderEntry(
        name="SMALL", target_params="~10M",
        vocab_size=REAL_VOCAB_SIZE, d_model=256, num_heads=8, d_ff=512,
        num_layers=11, max_len=DEFAULT_CONTEXT_SIZE, dropout=0.1,
    ),
    LadderEntry(
        name="MEDIUM", target_params="~25M",
        vocab_size=REAL_VOCAB_SIZE, d_model=384, num_heads=12, d_ff=768,
        num_layers=16, max_len=DEFAULT_CONTEXT_SIZE, dropout=0.1,
    ),
    LadderEntry(
        name="LARGE", target_params="~50M",
        vocab_size=REAL_VOCAB_SIZE, d_model=512, num_heads=16, d_ff=1024,
        num_layers=20, max_len=DEFAULT_CONTEXT_SIZE, dropout=0.1,
    ),
    LadderEntry(
        name="OPTIONAL", target_params="~100M",
        vocab_size=REAL_VOCAB_SIZE, d_model=768, num_heads=16, d_ff=1536,
        num_layers=18, max_len=DEFAULT_CONTEXT_SIZE, dropout=0.1,
    ),
]


def get_ladder() -> List[LadderEntry]:
    return list(SCALING_LADDER)


def get_ladder_entry(name: str) -> LadderEntry:
    for entry in SCALING_LADDER:
        if entry.name == name:
            return entry
    raise KeyError(f"Unknown ladder entry: {name}")


#: Context-length variants analyzed against the CURRENT width/depth.
CONTEXT_VARIANTS = [256, 512, 1024]


def context_analysis_base_config(max_len: int) -> ModelConfig:
    """CURRENT-shaped config at a given context length (for isolated analysis)."""
    return ModelConfig(
        vocab_size=REAL_VOCAB_SIZE,
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_len=max_len,
        dropout=0.1,
        dtype="float64",
        seed=None,
        backend="cpu",
        device="cpu",
    )
