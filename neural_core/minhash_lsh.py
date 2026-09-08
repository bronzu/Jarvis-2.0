"""
MinHash/LSH near-duplicate detection for the JARVIS dataset.

Replaces the previous 500-row Jaccard sampling approach. Uses a deterministic
hash family + banding (LSH) to find candidate pairs, then a strict Jaccard
verification step.

Methodology:
    1. Tokenize each document into word shingles (n-gram over tokens).
    2. Compute `num_perm` MinHash signatures using stable hash functions.
    3. Apply LSH banding with `b` bands x `r` rows to generate candidate pairs.
    4. Verify candidates with exact Jaccard similarity.
    5. Report the distribution of similarities and the count of pairs above
       the threshold.

This is reproducible (seeded), scalable (O(N) signature computation, O(N)
candidate pairs via LSH), and stdlib-only.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


MAX_HASH = (1 << 64) - 1


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text.lower())


def _shingles(tokens: List[str], n: int = 3) -> Set[str]:
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _hash_token(token: str, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & MAX_HASH


def minhash_signature(shingles: Set[str], num_perm: int = 64) -> List[int]:
    """Compute the MinHash signature for a set of shingles.

    Each of the `num_perm` hash functions picks the minimum hash value across
    all shingles. This gives a (1/sqrt(num_perm)) approximation of Jaccard.
    """
    if not shingles:
        return [MAX_HASH] * num_perm
    sig = [MAX_HASH] * num_perm
    for sh in shingles:
        for i in range(num_perm):
            h = _hash_token(sh, seed=i)
            if h < sig[i]:
                sig[i] = h
    return sig


def lsh_bands(signature: List[int], num_bands: int, num_rows: int) -> List[Tuple[int, int]]:
    """Split a MinHash signature into `num_bands` bands of `num_rows` hashes.

    Each band is hashed as a tuple to produce a bucket key.
    """
    assert len(signature) == num_bands * num_rows, "signature length must equal bands*rows"
    keys = []
    for b in range(num_bands):
        band = signature[b * num_rows:(b + 1) * num_rows]
        band_key = hash(tuple(band))
        keys.append((b, band_key))
    return keys


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicates(
    documents: List[str],
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
    bands: Optional[int] = None,
    rows: Optional[int] = None,
) -> Dict[str, object]:
    """Find near-duplicate pairs across `documents`.

    Returns a structured report with:
      - num_documents
      - num_perm, ngram, threshold, bands, rows (parameters used)
      - num_candidate_pairs (LSH)
      - num_verified_pairs (Jaccard >= threshold)
      - pairs (list of (i, j, jaccard))
      - similarity_histogram (bucketed jaccard distribution for verified pairs)
    """
    n = len(documents)
    if bands is None or rows is None:
        bands, rows = _optimal_lsh_params(num_perm, threshold)
    assert bands * rows == num_perm, f"bands*rows ({bands*rows}) must equal num_perm ({num_perm})"

    shingles_list = [_shingles(_tokenize(d), n=ngram) for d in documents]
    signatures = [minhash_signature(s, num_perm=num_perm) for s in shingles_list]

    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, sig in enumerate(signatures):
        for b_idx, key in enumerate(lsh_bands(sig, bands, rows)):
            buckets[(b_idx, key)].append(idx)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                candidate_pairs.add((bucket_docs[i], bucket_docs[j]))

    verified_pairs: List[Tuple[int, int, float]] = []
    for i, j in candidate_pairs:
        sim = _jaccard(shingles_list[i], shingles_list[j])
        if sim >= threshold:
            verified_pairs.append((i, j, sim))
    verified_pairs.sort(key=lambda x: -x[2])

    histogram = _build_histogram(verified_pairs)

    return {
        "num_documents": n,
        "num_perm": num_perm,
        "ngram": ngram,
        "threshold": threshold,
        "bands": bands,
        "rows": rows,
        "num_candidate_pairs": len(candidate_pairs),
        "num_verified_pairs": len(verified_pairs),
        "pairs": verified_pairs[:100],
        "similarity_histogram": histogram,
        "method": "MinHash + LSH (banding) + exact Jaccard verification",
    }


def _optimal_lsh_params(num_perm: int, threshold: float) -> Tuple[int, int]:
    """Pick (bands, rows) maximizing probability of catching a pair at the
    given threshold while keeping bands*rows == num_perm.

    Standard formula: choose b ~ num_perm / max(1, log2(1/threshold)).
    """
    if threshold <= 0 or threshold >= 1:
        return (1, num_perm)
    target = max(2, int(round(1.0 / max(0.01, threshold))))
    rows = 1
    while num_perm // rows > target and rows < num_perm:
        rows *= 2
    bands = num_perm // rows
    return bands, rows


def _build_histogram(pairs: List[Tuple[int, int, float]]) -> Dict[str, int]:
    bins = {"0.85-0.90": 0, "0.90-0.95": 0, "0.95-0.99": 0, "0.99-1.00": 0, "1.00": 0}
    for _, _, sim in pairs:
        if sim >= 1.0:
            bins["1.00"] += 1
        elif sim >= 0.99:
            bins["0.99-1.00"] += 1
        elif sim >= 0.95:
            bins["0.95-0.99"] += 1
        elif sim >= 0.90:
            bins["0.90-0.95"] += 1
        else:
            bins["0.85-0.90"] += 1
    return bins


def audit_dataset_near_duplicates(
    train_path: Path = Path("neural_core/data/train.txt"),
    val_path: Path = Path("neural_core/data/validation.txt"),
    test_path: Path = Path("neural_core/data/test.txt"),
    threshold: float = 0.85,
    num_perm: int = 64,
    ngram: int = 3,
) -> Dict[str, object]:
    docs: List[str] = []
    boundaries: Dict[str, Tuple[int, int]] = {}
    for label, path in [("train", train_path), ("validation", val_path), ("test", test_path)]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
        start = len(docs)
        docs.extend(sentences)
        boundaries[label] = (start, len(docs))

    overall = find_near_duplicates(docs, threshold=threshold, num_perm=num_perm, ngram=ngram)

    cross_split = []
    for i, j, sim in overall["pairs"]:
        split_i = _split_of(i, boundaries)
        split_j = _split_of(j, boundaries)
        if split_i != split_j:
            cross_split.append({
                "i": i, "j": j, "jaccard": sim,
                "split_i": split_i, "split_j": split_j,
                "doc_i_preview": docs[i][:80],
                "doc_j_preview": docs[j][:80],
            })

    return {
        "params": {
            "method": overall["method"],
            "num_perm": overall["num_perm"],
            "ngram": overall["ngram"],
            "threshold": overall["threshold"],
            "bands": overall["bands"],
            "rows": overall["rows"],
        },
        "totals": {
            "num_documents": overall["num_documents"],
            "num_candidate_pairs": overall["num_candidate_pairs"],
            "num_verified_pairs": overall["num_verified_pairs"],
        },
        "cross_split_pairs": cross_split[:50],
        "cross_split_count": len(cross_split),
        "similarity_histogram": overall["similarity_histogram"],
    }


def _split_of(idx: int, boundaries: Dict[str, Tuple[int, int]]) -> str:
    for label, (s, e) in boundaries.items():
        if s <= idx < e:
            return label
    return "unknown"


if __name__ == "__main__":
    report = audit_dataset_near_duplicates()
    print(json.dumps(report, indent=2, ensure_ascii=False))
