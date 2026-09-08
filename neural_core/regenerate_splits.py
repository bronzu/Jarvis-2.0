"""
Regenerate dataset splits using the strict normalization rule.

Reads the existing train/val/test files, extracts sentences, applies strict
normalization, deduplicates globally, reassigns to splits via hash, and
writes back to the same files. The original `train.txt` / `validation.txt` /
`test.txt` are OVERWRITTEN in place.
"""

import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from neural_core.build_dataset import _strict_normalize


TRAIN = Path("neural_core/data/train.txt")
VAL = Path("neural_core/data/validation.txt")
TEST = Path("neural_core/data/test.txt")
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _paragraphs_to_sentences(paragraphs: List[str], min_len: int = 3) -> List[str]:
    out: List[str] = []
    for p in paragraphs:
        for s in re.split(r"(?<=[.!?])\s+", p):
            s = s.strip()
            if len(s) >= min_len:
                out.append(s)
    return out


def _load_split(path: Path) -> List[str]:
    if not path.exists():
        return []
    return _split_paragraphs(path.read_text(encoding="utf-8"))


def _dedup_paragraphs_within(paragraphs: List[str], sent_drop: set) -> List[str]:
    out: List[str] = []
    for p in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s.strip()]
        kept = [s for s in sents if _strict_normalize(s) not in sent_drop]
        if kept:
            out.append(" ".join(kept))
    return out


def _assign_split(normalized: str, seed: int = SEED) -> str:
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    ratio = (int(digest[:8], 16) % 10000) / 10000.0
    if ratio < TRAIN_RATIO:
        return "train"
    if ratio < TRAIN_RATIO + VAL_RATIO:
        return "validation"
    return "test"


def _build_paragraphs(sentences: List[str], per_paragraph: int = 10) -> List[str]:
    paragraphs = []
    for i in range(0, len(sentences), per_paragraph):
        chunk = sentences[i:i + per_paragraph]
        text = " ".join(chunk)
        if not text.endswith("."):
            text += "."
        paragraphs.append(text)
    return paragraphs


def regenerate() -> Dict[str, object]:
    all_sentences: List[str] = []
    for label, path in [("train", TRAIN), ("validation", VAL), ("test", TEST)]:
        paragraphs = _load_split(path)
        all_sentences.extend(_paragraphs_to_sentences(paragraphs))

    norm_to_original: Dict[str, str] = {}
    for s in all_sentences:
        n = _strict_normalize(s)
        if n and n not in norm_to_original:
            norm_to_original[n] = s

    split_sentences: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    for n, s in norm_to_original.items():
        split = _assign_split(n)
        split_sentences[split].append(s)

    rng = random.Random(SEED)
    for k in split_sentences:
        rng.shuffle(split_sentences[k])

    for label, path in [("train", TRAIN), ("validation", VAL), ("test", TEST)]:
        paragraphs = _build_paragraphs(split_sentences[label])
        path.write_text("\n\n".join(paragraphs), encoding="utf-8")

    return {
        "total_unique_sentences": len(norm_to_original),
        "per_split_counts": {k: len(v) for k, v in split_sentences.items()},
    }


if __name__ == "__main__":
    import json
    result = regenerate()
    print(json.dumps(result, indent=2, ensure_ascii=False))
