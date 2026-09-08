"""
Phase 2B — Dataset verification for Kaggle reproducibility.

Prepares the mechanism to VERIFY the dataset on a remote machine (e.g. Kaggle)
without regenerating, cleaning, or modifying the corpus. Verification covers:

    - dataset identification (path + sha256 checksum);
    - token-count verification (encode with the real tokenizer);
    - tokenizer identity/version verification;
    - vocabulary verification (real vocab size = 8590).

The tokenizer declares ``max_vocab_size=8000`` but the FITTED vocabulary is
8590. This discrepancy is DOCUMENTED, not "corrected" automatically. Scaling
keeps vocab_size = 8590 (the real tokenizer vocabulary).
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


#: Real vocabulary size derived from the trained tokenizer (NOT 8000).
EXPECTED_VOCAB_SIZE = 8590

#: Documented discrepancy: tokenizer declares this, but fitted vocab is 8590.
DECLARED_MAX_VOCAB_SIZE = 8000


def sha256_of_file(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def count_tokens(text: str, tokenizer) -> int:
    """Token count of ``text`` using the (already fitted) tokenizer."""
    return len(tokenizer.encode(text))


def verify_tokenizer(tokenizer, expected_vocab_size: int = EXPECTED_VOCAB_SIZE,
                     declared_max_vocab_size: int = DECLARED_MAX_VOCAB_SIZE) -> Dict[str, Any]:
    """Document and verify tokenizer identity, including the 8000/8590 gap."""
    actual = int(tokenizer.vocab_size)
    declared = getattr(tokenizer, "max_vocab_size", None)
    discrepancy = declared is not None and actual != declared
    return {
        "vocab_size": actual,
        "expected_vocab_size": expected_vocab_size,
        "vocab_size_matches_expected": actual == expected_vocab_size,
        "declared_max_vocab_size": declared,
        "documented_discrepancy": bool(discrepancy),
        "discrepancy_detail": (
            f"tokenizer declares max_vocab_size={declared} but fitted "
            f"vocabulary is {actual} (kept as-is for scaling; not auto-corrected)"
            if discrepancy else ""
        ),
        "mode": getattr(tokenizer, "mode", None),
        "version": getattr(tokenizer, "state_dict", lambda: {}).__call__().get("version"),
    }


def verify_dataset(path, tokenizer=None, expected_sha256: Optional[str] = None,
                   expected_token_count: Optional[int] = None,
                   expected_vocab_size: int = EXPECTED_VOCAB_SIZE) -> Dict[str, Any]:
    """
    Verify a dataset file. Does NOT modify or regenerate anything.

    Returns a dict with checksum, token count, and (if a tokenizer is supplied)
    vocabulary verification. ``status`` is "OK" only when all provided expected
    values match.
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False, "status": "MISSING"}

    sha = sha256_of_file(path)
    token_count = None
    vocab_check = None
    if tokenizer is not None:
        text = path.read_text(encoding="utf-8")
        token_count = count_tokens(text, tokenizer)
        vocab_check = verify_tokenizer(tokenizer, expected_vocab_size)

    checks = []
    if expected_sha256 is not None:
        checks.append(sha == expected_sha256)
    if expected_token_count is not None:
        checks.append(token_count == expected_token_count)
    if vocab_check is not None:
        checks.append(vocab_check["vocab_size_matches_expected"])

    status = "OK" if (not checks or all(checks)) else "MISMATCH"

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha,
        "expected_sha256": expected_sha256,
        "sha256_matches": None if expected_sha256 is None else (sha == expected_sha256),
        "token_count": token_count,
        "expected_token_count": expected_token_count,
        "token_count_matches": None if expected_token_count is None else (token_count == expected_token_count),
        "vocab_verification": vocab_check,
        "status": status,
    }


def dataset_identity(path, tokenizer=None) -> Dict[str, Any]:
    """Build a portable dataset identity record (sha256 + token count + vocab)."""
    res = verify_dataset(path, tokenizer=tokenizer)
    return {
        "path": res["path"],
        "sha256": res["sha256"],
        "token_count": res["token_count"],
        "vocab_verification": res.get("vocab_verification"),
    }
