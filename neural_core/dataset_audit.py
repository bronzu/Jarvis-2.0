"""
Dataset audit for JARVIS 2.0.

Produces a reproducible report with:
  - split sizes
  - token counts
  - length stats
  - duplicates (exact, normalized, sentence)
  - near-duplicates (Jaccard on token sets)
  - train/val/test overlap
  - token frequency / rare tokens
  - UNK frequency with current tokenizer
"""

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

from neural_core.tokenizer import Tokenizer


DATA_DIR = Path("neural_core/data")
TRAIN_FILE = DATA_DIR / "train.txt"
VAL_FILE = DATA_DIR / "validation.txt"
TEST_FILE = DATA_DIR / "test.txt"


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _sentence_hash(sentence: str) -> str:
    return hashlib.md5(_normalize(sentence).encode("utf-8")).hexdigest()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _near_duplicate_pairs(texts: List[str], threshold: float = 0.85, sample_size: int = 500) -> List[Tuple[int, int, float]]:
    if len(texts) > sample_size:
        rng = __import__("random").Random(42)
        texts = rng.sample(texts, sample_size)
    token_sets = [set(_tokenize(t)) for t in texts]
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = _jaccard(token_sets[i], token_sets[j])
            if score >= threshold:
                pairs.append((i, j, score))
    return pairs


def audit_dataset(
    train_path: Path = TRAIN_FILE,
    val_path: Path = VAL_FILE,
    test_path: Path = TEST_FILE,
    max_vocab_size: int = 8000,
    near_duplicate_threshold: float = 0.85,
) -> Dict[str, Any]:
    train_lines = _read_lines(train_path)
    val_lines = _read_lines(val_path)
    test_lines = _read_lines(test_path)

    train_text = "\n".join(train_lines)
    val_text = "\n".join(val_lines)
    test_text = "\n".join(test_lines)

    tokenizer = Tokenizer(max_vocab_size=max_vocab_size, mode="whitespace")
    tokenizer.fit(train_text)

    def _stats(lines: List[str], text: str) -> Dict[str, Any]:
        tokens = tokenizer.encode(text)
        lengths = [len(line.split()) for line in lines]
        unk_id = tokenizer.vocabulary[tokenizer.UNK_TOKEN]
        unk_count = sum(1 for t in tokens if t == unk_id)
        return {
            "examples": len(lines),
            "token_count": len(tokens),
            "mean_length": mean(lengths) if lengths else 0.0,
            "median_length": median(lengths) if lengths else 0.0,
            "std_length": stdev(lengths) if len(lengths) > 1 else 0.0,
            "unk_count": unk_count,
            "unk_rate": unk_count / max(1, len(tokens)),
        }

    train_stats = _stats(train_lines, train_text)
    val_stats = _stats(val_lines, val_text)
    test_stats = _stats(test_lines, test_text)

    train_set = set(train_lines)
    val_set = set(val_lines)
    test_set = set(test_lines)

    train_norm = set(_normalize(l) for l in train_lines)
    val_norm = set(_normalize(l) for l in val_lines)
    test_norm = set(_normalize(l) for l in test_lines)

    def _sentences(lines: List[str]) -> List[str]:
        sents = []
        for line in lines:
            for s in line.split("."):
                s = s.strip()
                if s and len(s) > 10:
                    sents.append(s)
        return sents

    train_sents = set(_sentences(train_lines))
    val_sents = set(_sentences(val_lines))
    test_sents = set(_sentences(test_lines))

    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    test_tokens = tokenizer.encode(test_text)

    train_ds = type("DS", (), {"get_inputs": lambda self: [t for t in train_tokens]})()
    val_ds = type("DS", (), {"get_inputs": lambda self: [t for t in val_tokens]})()
    test_ds = type("DS", (), {"get_inputs": lambda self: [t for t in test_tokens]})()

    token_overlap = {
        "train_val": len(set(train_ds.get_inputs()) & set(val_ds.get_inputs())),
        "train_test": len(set(train_ds.get_inputs()) & set(test_ds.get_inputs())),
        "val_test": len(set(val_ds.get_inputs()) & set(test_ds.get_inputs())),
    }

    all_lines = train_lines + val_lines + test_lines
    nd_pairs = _near_duplicate_pairs(all_lines, near_duplicate_threshold)
    near_duplicates = [
        {"i": i, "j": j, "score": score, "split_a": "train" if i < len(train_lines) else "val" if i < len(train_lines) + len(val_lines) else "test", "split_b": "train" if j < len(train_lines) else "val" if j < len(train_lines) + len(val_lines) else "test"}
        for i, j, score in nd_pairs
    ]

    freq = {}
    for t in train_tokens:
        freq[t] = freq.get(t, 0) + 1
    rare = sum(1 for c in freq.values() if c == 1)
    unk_freq = freq.get(tokenizer.vocabulary[tokenizer.UNK_TOKEN], 0)

    return {
        "splits": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        },
        "duplicates": {
            "exact": {
                "train_val": len(train_set & val_set),
                "train_test": len(train_set & test_set),
                "val_test": len(val_set & test_set),
            },
            "normalized": {
                "train_val": len(train_norm & val_norm),
                "train_test": len(train_norm & test_norm),
                "val_test": len(val_norm & test_norm),
            },
            "sentence": {
                "train_val": len(train_sents & val_sents),
                "train_test": len(train_sents & test_sents),
                "val_test": len(val_sents & test_sents),
            },
        },
        "token_overlap": token_overlap,
        "near_duplicates": {
            "count": len(near_duplicates),
            "threshold": near_duplicate_threshold,
            "pairs": near_duplicates[:50],
        },
        "vocabulary": {
            "size": tokenizer.vocab_size,
            "max_vocab_size": tokenizer.max_vocab_size,
            "rare_tokens": rare,
            "unk_frequency": unk_freq,
        },
        "valid": (
            len(train_set & val_set) == 0
            and len(train_set & test_set) == 0
            and len(val_set & test_set) == 0
            and len(train_norm & val_norm) == 0
            and len(train_norm & test_norm) == 0
            and len(val_norm & test_norm) == 0
            and len(train_sents & val_sents) == 0
            and len(train_sents & test_sents) == 0
            and len(val_sents & test_sents) == 0
        ),
        "provenance": {
            "manifest_path": "neural_core/data/provenance_manifest.json",
            "template_leakage_measured": False,
            "source_leakage_measured": False,
            "notes": "Template/source leakage metadata is not preserved in the current dataset pipeline. "
                     "Enable provenance tracking by generating datasets via build_dataset.py with provenance metadata.",
        },
    }


if __name__ == "__main__":
    report = audit_dataset()
    print(json.dumps(report, indent=2, ensure_ascii=False))
